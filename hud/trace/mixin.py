"""TraceMixin - Adds trace() method to Environment.

This mixin provides the trace() context manager that creates TraceContext
instances for recording agent runs, with optional parallel execution and
variant-based A/B testing.
"""

from __future__ import annotations

import inspect
import itertools
import logging
import uuid
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

from hud.trace.context import TraceContext
from hud.trace.parallel import (
    ASTExtractionError,
    _get_with_block_body,
    run_parallel_traces,
)

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from hud.types import MCPToolResult

logger = logging.getLogger(__name__)


def _expand_variants(
    variants: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """Expand variants dict into all combinations.
    
    Args:
        variants: Dict where values can be:
            - Single value: {"model": "gpt-4o"} → fixed
            - List: {"model": ["gpt-4o", "claude"]} → expand
    
    Returns:
        List of variant assignments, one per combination.
        
    Examples:
        >>> _expand_variants(None)
        [{}]
        >>> _expand_variants({"model": "gpt-4o"})
        [{"model": "gpt-4o"}]
        >>> _expand_variants({"model": ["gpt-4o", "claude"]})
        [{"model": "gpt-4o"}, {"model": "claude"}]
        >>> _expand_variants({"model": ["a", "b"], "temp": [0.0, 0.7]})
        [{"model": "a", "temp": 0.0}, {"model": "a", "temp": 0.7},
         {"model": "b", "temp": 0.0}, {"model": "b", "temp": 0.7}]
    """
    if not variants:
        return [{}]
    
    # Normalize: single values become single-element lists
    expanded: dict[str, list[Any]] = {}
    for key, value in variants.items():
        if isinstance(value, list):
            expanded[key] = value
        else:
            expanded[key] = [value]
    
    # Generate all combinations
    keys = list(expanded.keys())
    value_lists = [expanded[k] for k in keys]
    
    return [
        dict(zip(keys, combo, strict=True))
        for combo in itertools.product(*value_lists)
    ]


class TraceMixin:
    """Mixin that adds trace capabilities to Environment.
    
    This mixin provides:
    - trace(): Create a TraceContext for recording agent runs
    - Parallel execution with group=N parameter
    - A/B testing with variants parameter
    
    Example:
        ```python
        class Environment(TraceMixin, MCPServer):
            ...
        
        env = Environment("my-env")
        
        # Single trace
        async with env.trace("task") as tc:
            await tc.call_tool("navigate", {"url": "..."})
            tc.reward = 0.9
        
        # Parallel traces (runs 4 times)
        async with env.trace("task", group=4) as tc:
            await tc.call_tool("navigate", {"url": "..."})
            tc.reward = 0.9
        
        # A/B testing (2 variants x 3 runs = 6 traces)
        async with env.trace("task",
            variants={"model": ["gpt-4o", "claude"]},
            group=3,
        ) as tc:
            model = tc.variants["model"]
            response = await call_llm(model=model)
            tc.reward = evaluate(response)
        
        # Access results
        for t in tc.results:
            print(f"{t.variants} run {t.index}: reward={t.reward}")
        ```
    """
    
    # These will be provided by the Environment class
    name: str
    
    # Store last parallel results (list of completed TraceContext objects)
    _last_traces: list[TraceContext] | None = None
    
    async def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> MCPToolResult:
        """Placeholder - implemented by Environment."""
        raise NotImplementedError
    
    @property
    def last_traces(self) -> list[TraceContext] | None:
        """Get TraceContext objects from the last parallel execution.
        
        Each TraceContext has: trace_id, index, reward, duration, error, success
        """
        return self._last_traces
    
    @asynccontextmanager
    async def trace(
        self,
        name: str,
        *,
        variants: dict[str, Any] | None = None,
        group: int = 1,
        group_ids: list[str] | None = None,
        job_id: str | None = None,
        trace_id: str | None = None,
        api_key: str | None = None,
    ) -> AsyncGenerator[TraceContext, None]:
        """Create a trace context for recording an agent run.
        
        The trace context provides:
        - Unique trace identification
        - Task name linking (for training data construction)
        - Headers for gateway integration (auto-injected to inference.hud.ai)
        - Tool call delegation
        - Reward setting
        - Metrics logging
        
        A/B Testing:
            Use `variants` to define experiment variables. Each list value
            creates a variant; single values are fixed. All combinations
            are expanded and run.
            
        Parallel Execution:
            Use `group` to run multiple times per variant for statistical
            significance. Total traces = len(variants combinations) x group.
        
        Args:
            name: Task name for this trace (used for task construction)
            variants: A/B test configuration. Dict where:
                - List values are expanded: {"model": ["gpt-4o", "claude"]}
                - Single values are fixed: {"temp": 0.7}
                - All combinations are run
            group: Runs per variant (default: 1) for statistical significance.
            group_ids: Optional list of group IDs for each trace.
                       Length must match (variants x group). If not provided,
                       a single shared group_id is auto-generated.
            job_id: Optional job ID to link this trace to. If not provided,
                    auto-detects from current `hud.job()` context.
            trace_id: Optional trace ID (auto-generated if not provided).
                      For parallel execution, each trace gets a unique ID.
            api_key: Optional API key for backend calls (defaults to settings.api_key)
            
        Yields:
            TraceContext for this trace. Inside the body:
            - `tc.variants` = current variant assignment (e.g., {"model": "gpt-4o"})
            - `tc.index` = local run index (for debugging)
            - `tc.group_id` = links all traces in this parallel execution
            
            After execution (for variants/group > 1):
            - `tc.results` = list of all TraceContext objects
            - `tc.reward` = mean reward across all traces
            
        Example:
            ```python
            # Single execution
            async with env.trace("task") as tc:
                await tc.call_tool("search", {"query": "..."})
                tc.reward = 1.0
            
            # A/B test: 2 variants x 3 runs = 6 traces
            async with env.trace("task",
                variants={"model": ["gpt-4o", "claude"]},
                group=3,
            ) as tc:
                model = tc.variants["model"]  # Assigned per-trace
                response = await call_llm(model=model)
                tc.reward = evaluate(response)
            
            # Access results
            for t in tc.results:
                print(f"{t.variants} run {t.index}: reward={t.reward}")
            ```
            
        Limitations (for variants/group > 1):
            - Requires source file (won't work in REPL/Jupyter)
            - Outer variables captured at enter time, changes don't propagate back
            - Modifying mutable objects causes race conditions
            - Cannot use yield/generators inside body
        """
        if group <= 0:
            raise ValueError("group must be >= 1")
        
        # Expand variants into all combinations
        variant_combos = _expand_variants(variants)
        total_traces = len(variant_combos) * group
        
        # Validate parallelization - only remote connections allowed for group > 1
        if total_traces > 1 and not self.is_parallelizable:  # type: ignore[attr-defined]
            local_conns = self.local_connections  # type: ignore[attr-defined]
            raise ValueError(
                f"Cannot run parallel traces (group={group}) with local connections.\n"
                f"  Local connections: {local_conns}\n"
                f"  Local connections (stdio/Docker) can only run one instance.\n"
                f"  Use remote connections (HTTP/URL) for parallel execution."
            )
        
        if total_traces == 1:
            # Simple case: single trace
            # TraceContext enters FIRST (sets headers in contextvar)
            # Environment enters SECOND (can inject headers into connections)
            tc = TraceContext(
                env=self,  # type: ignore[arg-type]
                name=name,
                trace_id=trace_id,
                api_key=api_key,
                job_id=job_id,
                _variants=variant_combos[0],
            )
            async with tc:
                async with self:  # type: ignore[attr-defined]
                    yield tc
        else:
            # Parallel execution: each trace gets its own environment instance
            # Parent environment NOT entered - each child connects independently
            completed = await self._run_parallel_trace(
                name=name,
                variant_combos=variant_combos,
                group=group,
                group_ids=group_ids,
                job_id=job_id,
                api_key=api_key,
            )
            
            # Create parent tc with results injected
            tc = TraceContext(
                env=self,  # type: ignore[arg-type]
                name=name,
                trace_id=trace_id,
                api_key=api_key,
                job_id=job_id,
            )
            tc.results = completed
            self._last_traces = completed
            
            # Compute aggregate reward (mean of non-None rewards)
            rewards = [t.reward for t in completed if t.reward is not None]
            if rewards:
                tc.reward = sum(rewards) / len(rewards)
            
            yield tc
    
    async def _run_parallel_trace(
        self,
        name: str,
        variant_combos: list[dict[str, Any]],
        group: int,
        group_ids: list[str] | None,
        job_id: str | None,
        api_key: str | None,
    ) -> list[TraceContext]:
        """Run parallel trace execution using AST extraction.
        
        This method:
        1. Captures the caller's frame
        2. Extracts the with-block body via AST
        3. Creates (variants x group) TraceContext instances
        4. Runs the body in parallel
        5. Stores results in self._last_traces
        
        Args:
            name: Task name
            variant_combos: List of variant assignments (one per combination)
            group: Runs per variant
            group_ids: Optional list of group IDs (one per total trace)
            job_id: Optional job ID (auto-detected from current job if not provided)
            api_key: Optional API key
        """
        # Get the caller's frame (skip this method and the trace method)
        frame = inspect.currentframe()
        if frame is None:
            raise ASTExtractionError("Cannot get current frame")
        
        try:
            # Go up: _run_parallel_trace -> trace -> user code
            caller_frame = frame.f_back
            if caller_frame is not None:
                caller_frame = caller_frame.f_back
            if caller_frame is None:
                raise ASTExtractionError("Cannot get caller frame")
            
            # Extract the with-block body
            body_source, captured_locals = _get_with_block_body(caller_frame)
            
        finally:
            del frame  # Avoid reference cycles
        
        # Calculate total traces
        total_traces = len(variant_combos) * group
        
        # Use provided group_ids or generate one shared group_id
        if group_ids:
            if len(group_ids) != total_traces:
                raise ValueError(
                    f"group_ids length ({len(group_ids)}) must match "
                    f"total traces ({total_traces} = {len(variant_combos)} variants x {group} runs)"
                )
            resolved_group_ids = group_ids
        else:
            # All traces share one auto-generated group_id
            shared_group_id = str(uuid.uuid4())
            resolved_group_ids = [shared_group_id] * total_traces
        
        # Create TraceContext for each (variant, run) combination
        trace_contexts: list[TraceContext] = []
        idx = 0
        for variant in variant_combos:
            for _ in range(group):
                tc = TraceContext(
                    env=self,  # type: ignore[arg-type]
                    name=name,
                    api_key=api_key,
                    job_id=job_id,
                    _group_id=resolved_group_ids[idx],
                    _index=idx,
                    _variants=variant,
                )
                trace_contexts.append(tc)
                idx += 1
        
        # Run in parallel
        total = len(trace_contexts)
        logger.info(
            "Running %d traces for task '%s' (%d variants x %d runs)",
            total, name, len(variant_combos), group,
        )
        completed = await run_parallel_traces(trace_contexts, body_source, captured_locals)
        
        # Store results
        self._last_traces = completed
        
        # Calculate stats
        rewards = [tc.reward for tc in completed if tc.reward is not None]
        mean_reward = sum(rewards) / len(rewards) if rewards else 0.0
        success_count = sum(1 for tc in completed if tc.success)
        
        logger.info(
            "Traces complete: %d/%d succeeded, mean_reward=%.3f",
            success_count,
            len(completed),
            mean_reward,
        )
        
        return completed
