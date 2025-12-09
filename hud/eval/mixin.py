"""EvalMixin - Adds eval() method to Environment.

This mixin provides the eval() context manager that creates EvalContext
instances for recording agent runs, with optional parallel execution and
variant-based A/B testing.
"""

from __future__ import annotations

import inspect
import logging
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

from hud.eval.parallel import (
    ASTExtractionError,
    execute_parallel_evals,
    expand_variants,
    get_with_block_body,
    resolve_group_ids,
)

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from hud.eval.context import EvalContext
    from hud.types import MCPToolResult

logger = logging.getLogger(__name__)


class EvalMixin:
    """Mixin that adds eval capabilities to Environment.

    This mixin provides:
    - eval(): Create an EvalContext for recording agent runs
    - Parallel execution with group=N parameter
    - A/B testing with variants parameter

    Example:
        ```python
        class Environment(EvalMixin, MCPServer): ...


        env = Environment("my-env")

        # Single eval - yields EvalContext (which has Environment capabilities)
        async with env.eval("task") as ctx:
            await ctx.call_tool("navigate", {"url": "..."})
            ctx.reward = 0.9

        # Parallel evals (runs 4 times)
        async with env.eval("task", group=4) as ctx:
            await ctx.call_tool("navigate", {"url": "..."})
            ctx.reward = 0.9

        # A/B testing (2 variants x 3 runs = 6 evals)
        async with env.eval(
            "task",
            variants={"model": ["gpt-4o", "claude"]},
            group=3,
        ) as ctx:
            model = ctx.variants["model"]
            response = await call_llm(model=model)
            ctx.reward = evaluate(response)

        # Access results
        for e in ctx.results:
            print(f"{e.variants} run {e.index}: reward={e.reward}")
        ```
    """

    # These will be provided by the Environment class
    name: str

    # Store last parallel results
    _last_evals: list[EvalContext] | None = None

    async def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> MCPToolResult:
        """Placeholder - implemented by Environment."""
        raise NotImplementedError

    def _capture_code_snippet(self) -> str | None:
        """Capture the code inside the eval() with-block (best effort).

        Returns None if source cannot be extracted (e.g., REPL, Jupyter).
        """
        frame = inspect.currentframe()
        if frame is None:
            return None

        try:
            # Go up: _capture_code_snippet -> eval -> user code
            caller = frame.f_back
            if caller is not None:
                caller = caller.f_back
            if caller is None:
                return None

            body_source, _ = get_with_block_body(caller)
            return body_source
        except ASTExtractionError:
            # Can't extract from REPL/Jupyter - that's OK
            return None
        except Exception as e:
            logger.debug("Failed to capture code snippet: %s", e)
            return None
        finally:
            del frame

    def _get_env_config(self) -> dict[str, Any] | None:
        """Get serializable environment configuration.

        Returns dict with connections and local tools.
        """
        # This will be overridden by Environment with actual implementation
        return None

    @property
    def last_evals(self) -> list[EvalContext] | None:
        """Get EvalContext objects from the last parallel execution.

        Each EvalContext has: trace_id, index, reward, duration, error, success
        """
        return self._last_evals

    @asynccontextmanager
    async def eval(
        self,
        name: str,
        *,
        variants: dict[str, Any] | None = None,
        group: int = 1,
        group_ids: list[str] | None = None,
        job_id: str | None = None,
        trace_id: str | None = None,
        api_key: str | None = None,
    ) -> AsyncGenerator[EvalContext, None]:
        """Create an eval context for recording an agent run.

        The eval context provides:
        - Unique trace identification
        - Task name linking (for training data construction)
        - Headers for gateway integration (auto-injected to inference.hud.ai)
        - Tool call capabilities (call_tool, as_openai_chat_tools, etc.)
        - Reward setting
        - Metrics logging

        A/B Testing:
            Use `variants` to define experiment variables. Each list value
            creates a variant; single values are fixed. All combinations
            are expanded and run.

        Parallel Execution:
            Use `group` to run multiple times per variant for statistical
            significance. Total evals = len(variants combinations) x group.

        Args:
            name: Task name for this eval (used for task construction)
            variants: A/B test configuration. Dict where:
                - List values are expanded: {"model": ["gpt-4o", "claude"]}
                - Single values are fixed: {"temp": 0.7}
                - All combinations are run
            group: Runs per variant (default: 1) for statistical significance.
            group_ids: Optional list of group IDs for each eval.
                       Length must match (variants x group). If not provided,
                       a single shared group_id is auto-generated.
            job_id: Optional job ID to link this eval to. If not provided,
                    auto-detects from current `hud.job()` context.
            trace_id: Optional trace ID (auto-generated if not provided).
                      For parallel execution, each eval gets a unique ID.
            api_key: Optional API key for backend calls (defaults to settings.api_key)

        Yields:
            EvalContext for this evaluation. Inside the body:
            - `ctx.variants` = current variant assignment (e.g., {"model": "gpt-4o"})
            - `ctx.index` = local run index (for debugging)
            - `ctx.group_id` = links all evals in this parallel execution
            - `ctx.call_tool(...)` = call tools on the environment
            - `ctx.reward = ...` = set reward

            After execution (for variants/group > 1):
            - `ctx.results` = list of all EvalContext objects
            - `ctx.reward` = mean reward across all evals

        Example:
            ```python
            # Single execution
            async with env.eval("task") as ctx:
                await ctx.call_tool("search", {"query": "..."})
                ctx.reward = 1.0

            # A/B test: 2 variants x 3 runs = 6 evals
            async with env.eval(
                "task",
                variants={"model": ["gpt-4o", "claude"]},
                group=3,
            ) as ctx:
                model = ctx.variants["model"]  # Assigned per-eval
                response = await call_llm(model=model)
                ctx.reward = evaluate(response)

            # Access results
            for e in ctx.results:
                print(f"{e.variants} run {e.index}: reward={e.reward}")
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
        variant_combos = expand_variants(variants)
        total_evals = len(variant_combos) * group

        # Capture code snippet (best effort - won't work in REPL/Jupyter)
        code_snippet = self._capture_code_snippet()

        # Get environment config
        env_config = self._get_env_config()

        # Validate parallelization - only remote connections allowed for group > 1
        if total_evals > 1 and not self.is_parallelizable:  # type: ignore[attr-defined]
            local_conns = self.local_connections  # type: ignore[attr-defined]
            raise ValueError(
                f"Cannot run parallel evals (group={group}) with local connections.\n"
                f"  Local connections: {local_conns}\n"
                f"  Local connections (stdio/Docker) can only run one instance.\n"
                f"  Use remote connections (HTTP/URL) for parallel execution."
            )

        # Lazy import to avoid circular dependency
        from hud.eval.context import EvalContext

        if total_evals == 1:
            # Simple case: single eval
            # Create EvalContext from parent environment
            ctx = EvalContext.from_environment(
                env=self,  # type: ignore[arg-type]
                name=name,
                trace_id=trace_id,
                api_key=api_key,
                job_id=job_id,
                variants=variant_combos[0],
                code_snippet=code_snippet,
                env_config=env_config,
            )
            async with ctx:
                yield ctx
        else:
            # Parallel execution: each eval gets its own environment instance
            completed = await self._run_parallel_eval(
                name=name,
                variant_combos=variant_combos,
                group=group,
                group_ids=group_ids,
                job_id=job_id,
                api_key=api_key,
                code_snippet=code_snippet,
                env_config=env_config,
            )

            # Create parent ctx with results injected
            ctx = EvalContext.from_environment(
                env=self,  # type: ignore[arg-type]
                name=name,
                trace_id=trace_id,
                api_key=api_key,
                job_id=job_id,
                code_snippet=code_snippet,
                env_config=env_config,
            )
            ctx.results = completed
            self._last_evals = completed

            # Compute aggregate reward (mean of non-None rewards)
            rewards = [e.reward for e in completed if e.reward is not None]
            if rewards:
                ctx.reward = sum(rewards) / len(rewards)

            yield ctx

    async def _run_parallel_eval(
        self,
        name: str,
        variant_combos: list[dict[str, Any]],
        group: int,
        group_ids: list[str] | None,
        job_id: str | None,
        api_key: str | None,
        code_snippet: str | None,
        env_config: dict[str, Any] | None,
    ) -> list[EvalContext]:
        """Run parallel eval execution.

        Creates EvalContexts from parent environment and runs them in parallel.
        """
        # Lazy import to avoid circular dependency
        from hud.eval.context import EvalContext

        # Calculate total evals and resolve group IDs
        total_evals = len(variant_combos) * group
        resolved_group_ids = resolve_group_ids(group_ids, total_evals)

        # Create EvalContext for each (variant, run) combination
        eval_contexts: list[EvalContext] = []
        idx = 0
        for variant in variant_combos:
            for _ in range(group):
                ctx = EvalContext.from_environment(
                    env=self,  # type: ignore[arg-type]
                    name=name,
                    api_key=api_key,
                    job_id=job_id,
                    group_id=resolved_group_ids[idx],
                    index=idx,
                    variants=variant,
                    code_snippet=code_snippet,
                    env_config=env_config,
                )
                eval_contexts.append(ctx)
                idx += 1

        # Run in parallel (frame depth: _run_parallel_eval -> eval -> user code)
        completed = await execute_parallel_evals(eval_contexts, caller_frame_depth=3)

        # Store results
        self._last_evals = completed
        return completed


__all__ = ["EvalMixin"]

