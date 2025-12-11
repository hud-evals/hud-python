"""EvalContext - Environment with evaluation tracking.

EvalContext IS an Environment, with additional evaluation tracking
capabilities (trace_id, reward, backend reporting).

This makes `async with env.eval("task") as env` natural - you get
a full Environment that you can call tools on directly.
"""

from __future__ import annotations

import contextvars
import logging
import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Self

from hud.environment import Environment
from hud.environment.types import EnvConfig
from hud.settings import settings
from hud.shared import make_request
from hud.telemetry.job import get_current_job

if TYPE_CHECKING:
    from types import TracebackType

    from hud.types import Task

from hud.eval.types import EvalExitPayload, EvalPayload, ParallelEvalComplete

logger = logging.getLogger(__name__)

# Contextvar to store current trace headers (for httpx auto-instrumentation)
_current_trace_headers: contextvars.ContextVar[dict[str, str] | None] = contextvars.ContextVar(
    "current_trace_headers", default=None
)


def get_current_trace_headers() -> dict[str, str] | None:
    """Get the current trace headers from context."""
    return _current_trace_headers.get()


# =============================================================================
# EvalContext
# =============================================================================


class EvalContext(Environment):
    """Environment with evaluation tracking capabilities.

    Attributes:
        trace_id: Unique identifier for this evaluation
        eval_name: Task/evaluation name (separate from env name)
        job_id: Links to parent job (auto-detected from hud.job() context)
        group_id: Links parallel evaluations together
        variants: Variant assignment dict (for A/B testing)
        reward: Reward value (user-settable)
        error: Exception if failed
        results: All eval results (for parallel execution)
        task: Task definition (if loaded from slug)

    Example:
        ```python
        # From existing environment
        async with env.eval("task") as ctx:
            await ctx.call_tool("navigate", url="...")
            ctx.reward = 0.9

        # Standalone with slug
        async with hud.eval("my-org/task:1") as ctx:
            await agent.run(ctx)
            ctx.reward = result.reward

        # Blank eval
        async with hud.eval() as ctx:
            ctx.reward = compute_reward()
        ```
    """

    def __init__(
        self,
        name: str = "eval",
        *,
        trace_id: str | None = None,
        api_key: str | None = None,
        job_id: str | None = None,
        group_id: str | None = None,
        index: int = 0,
        variants: dict[str, Any] | None = None,
        code_snippet: str | None = None,
        env_config: dict[str, Any] | None = None,
        task: Task | None = None,
        trace: bool = True,
        quiet: bool = False,
        **env_kwargs: Any,
    ) -> None:
        """Initialize EvalContext.

        Args:
            name: Environment/evaluation name
            trace_id: Unique trace ID (auto-generated if not provided)
            api_key: API key for backend calls
            job_id: Job ID to link to (auto-detected if not provided)
            group_id: Group ID for parallel evaluations
            index: Index in parallel execution
            variants: Variant assignment for A/B testing
            code_snippet: Code being evaluated (for reproducibility)
            env_config: Environment configuration dict
            task: Task definition (if loaded from slug)
            trace: Whether to send trace data to backend (default True)
            quiet: Whether to suppress printing links (default False)
            **env_kwargs: Additional kwargs passed to Environment.__init__
        """
        # Initialize Environment
        super().__init__(name=name, **env_kwargs)

        # === Evaluation tracking (not in Environment) ===

        # Identity
        self.trace_id: str = trace_id or str(uuid.uuid4())
        self.eval_name: str = name  # Separate from self.name for clarity

        # Job linkage
        if job_id is None:
            current_job = get_current_job()
            self.job_id: str | None = current_job.id if current_job else None
        else:
            self.job_id = job_id

        self.group_id: str | None = group_id
        self.index: int = index

        # Variant assignment
        self.variants: dict[str, Any] = variants or {}

        # User-settable (per-run values, override Environment defaults)
        self.prompt: str | None = None  # From script setup or task
        self.reward: float | None = None
        self.answer: str | None = None  # Agent's submitted answer

        # Error tracking
        self.error: BaseException | None = None

        # Parallel results
        self.results: list[EvalContext] | None = None

        # Code and config
        self.code_snippet: str | None = code_snippet
        self._eval_env_config: dict[str, Any] | None = env_config

        # Task definition (if loaded from slug)
        self.task: Task | None = task

        # Apply task configuration
        if task:
            self._apply_task(task)

        # Private state for eval tracking
        self._eval_api_key = api_key
        self._started_at: datetime | None = None
        self._completed_at: datetime | None = None
        self._token: contextvars.Token[dict[str, str] | None] | None = None
        self._is_summary: bool = False  # True for summary contexts (skip trace)
        self._suppress_link: bool = quiet  # True to suppress printing eval link
        self._trace_enabled: bool = trace  # Whether to send trace data to backend
        self._script_name: str | None = None  # Current script name (for submit)
        self._source_env_name: str | None = None  # Source env name for remote lookups

    def _apply_task(self, task: Task) -> None:
        """Apply a Task definition to this environment."""
        # Set prompt
        if task.prompt:
            self.prompt = task.prompt

        # Connect MCP servers
        if task.mcp_config:
            self.connect_mcp_config(task.mcp_config)

        # Configure setup tool calls
        if task.setup_tool:
            setup_calls = task.setup_tool
            if not isinstance(setup_calls, list):
                setup_calls = [setup_calls]
            for call in setup_calls:
                self.setup_tool(call.name, **(call.arguments or {}))

        # Configure evaluate tool calls
        if task.evaluate_tool:
            eval_calls = task.evaluate_tool
            if not isinstance(eval_calls, list):
                eval_calls = [eval_calls]
            for call in eval_calls:
                self.evaluate_tool(call.name, **(call.arguments or {}))

    @classmethod
    def from_environment(
        cls,
        env: Environment,
        name: str,
        *,
        trace_id: str | None = None,
        api_key: str | None = None,
        job_id: str | None = None,
        group_id: str | None = None,
        index: int = 0,
        variants: dict[str, Any] | None = None,
        code_snippet: str | None = None,
        env_config: dict[str, Any] | None = None,
        trace: bool = True,
        quiet: bool = False,
    ) -> EvalContext:
        """Create an EvalContext that copies configuration from an existing Environment.

        This creates a new EvalContext with the same connections as the parent.
        Used by env.eval() to create evaluation contexts.

        Args:
            env: Parent environment to copy from
            name: Evaluation name
            trace_id: Unique trace ID
            api_key: API key for backend calls
            job_id: Job ID to link to
            group_id: Group ID for parallel evaluations
            index: Index in parallel execution
            variants: Variant assignment
            code_snippet: Code being evaluated
            env_config: Environment configuration
        """
        ctx = cls(
            name=name,
            trace_id=trace_id,
            api_key=api_key,
            job_id=job_id,
            group_id=group_id,
            index=index,
            variants=variants,
            code_snippet=code_snippet,
            env_config=env_config,
            trace=trace,
            quiet=quiet,
        )

        # Copy connections from parent - each connector is copied so parallel
        # execution gets fresh client instances
        ctx._connections = {name: connector.copy() for name, connector in env._connections.items()}
        ctx._hub_configs = getattr(env, "_hub_configs", []).copy()
        ctx._setup_calls = env._setup_calls.copy()
        ctx._evaluate_calls = env._evaluate_calls.copy()

        # Copy scripts (definitions) by reference - they don't change
        ctx._scripts = getattr(env, "_scripts", {})
        # Create fresh session state for this eval (parallel evals each need their own)
        ctx._script_sessions = {}
        ctx._script_latest = {}
        ctx._script_answers = {}

        # Store source env name for remote script lookups
        ctx._source_env_name = env.name

        # Copy managers by reference (they hold local tools, prompts, resources)
        # This allows ctx.call_tool(), ctx.get_prompt(), ctx.read_resource() to work
        # for locally defined tools/scripts
        ctx._tool_manager = env._tool_manager
        ctx._prompt_manager = env._prompt_manager
        ctx._resource_manager = env._resource_manager

        # Copy prompt
        if env.prompt:
            ctx.prompt = env.prompt

        return ctx

    @classmethod
    def from_task(
        cls,
        task: Task,
        name: str | None = None,
        *,
        trace_id: str | None = None,
        api_key: str | None = None,
        job_id: str | None = None,
        group_id: str | None = None,
        index: int = 0,
        variants: dict[str, Any] | None = None,
        code_snippet: str | None = None,
        trace: bool = True,
        quiet: bool = False,
    ) -> EvalContext:
        """Create an EvalContext from a Task definition.

        .. deprecated:: 0.5.0
            Use Eval objects from env() instead of Task objects.

        Args:
            task: Task definition
            name: Evaluation name (defaults to task.id or "eval")
            trace_id: Unique trace ID
            api_key: API key for backend calls
            job_id: Job ID to link to
            group_id: Group ID for parallel evaluations
            index: Index in parallel execution
            variants: Variant assignment
            code_snippet: Code being evaluated
            trace: Whether to send trace data to backend
            quiet: Whether to suppress printing links
        """
        import warnings

        warnings.warn(
            "EvalContext.from_task() is deprecated. Use Eval objects from env() instead.",
            DeprecationWarning,
            stacklevel=2,
        )

        eval_name = name or task.id or "eval"

        return cls(
            name=eval_name,
            trace_id=trace_id,
            api_key=api_key,
            job_id=job_id,
            group_id=group_id,
            index=index,
            variants=variants,
            code_snippet=code_snippet,
            task=task,
            trace=trace,
            quiet=quiet,
        )

    # =========================================================================
    # Summary Context - Attribute Access Control
    # =========================================================================

    # Attributes accessible on summary context (everything else raises ParallelEvalComplete)
    _SUMMARY_ALLOWED = frozenset(
        {
            # Results and metadata
            "results",
            "reward",
            "error",
            "success",
            # IDs
            "trace_id",
            "job_id",
            "group_id",
            "index",
            # Private attrs
            "_is_summary",
            "_suppress_link",
            "__class__",
            "__dict__",
        }
    )

    def __getattribute__(self, name: str) -> Any:
        """Block most attribute access on summary contexts."""
        # Always allow private/dunder and whitelisted attrs
        if name.startswith("_") or name in EvalContext._SUMMARY_ALLOWED:
            return super().__getattribute__(name)

        # Check if this is a summary context
        try:
            is_summary = super().__getattribute__("_is_summary")
        except AttributeError:
            is_summary = False

        if is_summary:
            raise ParallelEvalComplete

        return super().__getattribute__(name)

    # =========================================================================
    # Computed Properties (eval-specific)
    # =========================================================================

    @property
    def headers(self) -> dict[str, str]:
        """Headers for gateway integration."""
        return {"Trace-Id": self.trace_id}

    @property
    def duration(self) -> float:
        """Execution duration in seconds."""
        if self._started_at is None:
            return 0.0
        end = self._completed_at or datetime.now(UTC)
        return (end - self._started_at).total_seconds()

    @property
    def success(self) -> bool:
        """True if no error occurred."""
        return self.error is None

    @property
    def done(self) -> bool:
        """True if execution completed."""
        return self._completed_at is not None

    # =========================================================================
    # Backend Integration
    # =========================================================================

    def _get_eval_api_key(self) -> str | None:
        return self._eval_api_key or settings.api_key

    def _build_base_payload(self) -> EvalPayload:
        """Build the base payload for enter/exit."""
        env_config_model: EnvConfig | None = None
        if self._eval_env_config:
            env_config_model = EnvConfig(**self._eval_env_config)

        return EvalPayload(
            job_name=self.eval_name,
            prompt=self.prompt,
            code_snippet=self.code_snippet,
            env_config=env_config_model,
            job_id=self.job_id,
            group_id=self.group_id,
            variants=self.variants if self.variants else None,
        )

    async def log(self, metrics: dict[str, Any]) -> None:
        """Log metrics to the backend."""
        api_key = self._get_eval_api_key()
        if not settings.telemetry_enabled or not api_key:
            return

        try:
            await make_request(
                method="POST",
                url=f"{settings.hud_telemetry_url}/traces/{self.trace_id}/log",
                json={"metrics": metrics},
                api_key=api_key,
            )
        except Exception as e:
            logger.warning("Failed to log metrics: %s", e)

    async def submit(self, answer: str) -> None:
        """Submit the agent's answer for script evaluation.

        Delegates to Environment.submit() with the current script name.
        The answer will be passed to the script's evaluate phase via
        `yield`, e.g.: `answer = yield "Do the task"`

        Args:
            answer: The agent's final answer/result to submit

        Example:
            async with env("checkout", product="laptop") as ctx:
                response = await agent.run(ctx.prompt)
                await ctx.submit(response)
            # On exit, script's evaluate phase receives the answer
        """
        if not self._script_name:
            logger.warning("submit() called but no script is running")
            return

        # Store answer on context for display
        self.answer = answer

        # Delegate to Environment.submit() which handles storage + broadcast
        await super().submit(self._script_name, answer)

    async def _eval_enter(self) -> None:
        """Notify backend that eval has started."""
        if not self._trace_enabled:
            return
        api_key = self._get_eval_api_key()
        if not settings.telemetry_enabled or not api_key:
            return

        try:
            payload = self._build_base_payload()
            await make_request(
                method="POST",
                url=f"{settings.hud_api_url}/trace/{self.trace_id}/enter",
                json=payload.model_dump(exclude_none=True),
                api_key=api_key,
            )
        except Exception as e:
            logger.warning("Failed to send eval enter: %s", e)

    async def _eval_exit(self, error_message: str | None = None) -> None:
        """Notify backend that eval has completed."""
        if not self._trace_enabled:
            return
        api_key = self._get_eval_api_key()
        if not settings.telemetry_enabled or not api_key:
            return

        # Use evaluate tool reward if not manually set
        reward = self.reward
        if reward is None:
            reward = getattr(self, "_evaluate_reward", None)

        try:
            payload = EvalExitPayload(
                **self._build_base_payload().model_dump(),
                reward=reward,
                success=self.success,
                error_message=error_message,
            )
            await make_request(
                method="POST",
                url=f"{settings.hud_api_url}/trace/{self.trace_id}/exit",
                json=payload.model_dump(exclude_none=True),
                api_key=api_key,
            )
        except Exception as e:
            logger.warning("Failed to send eval exit: %s", e)

    # =========================================================================
    # Context Manager (override Environment)
    # =========================================================================

    async def __aenter__(self) -> Self:
        """Enter eval context - connect environment and set trace headers."""
        if self._is_summary:
            return self

        # Start tracking
        self._started_at = datetime.now(UTC)
        self._token = _current_trace_headers.set(self.headers)

        # Connect environment (MCP servers, tools)
        await super().__aenter__()

        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> bool:
        """Exit eval context - disconnect and report."""
        # Summary contexts skip trace tracking (parallel results already tracked)
        # Suppress ParallelEvalComplete - it's expected for skipping body re-execution
        if self._is_summary:
            return exc_type is ParallelEvalComplete

        self._completed_at = datetime.now(UTC)

        # Track error
        error_msg: str | None = None
        if exc_type is not None:
            self.error = exc_val
            error_msg = str(exc_val) if exc_val else "Unknown error"

        # Disconnect environment (parent class)
        await super().__aexit__(exc_type, exc_val, exc_tb)

        # Reset context var
        if self._token is not None:
            _current_trace_headers.reset(self._token)
            self._token = None

        # Notify backend
        await self._eval_exit(error_msg)
        return False

    def __repr__(self) -> str:
        return f"EvalContext({self.trace_id[:8]}..., name={self.eval_name!r}, reward={self.reward})"

    def _print_eval_link(self) -> None:
        """Print a nicely formatted eval link."""
        # Skip if link printing is suppressed (e.g., parallel child traces)
        if self._suppress_link:
            return

        from hud.eval.display import print_link

        trace_url = f"https://hud.ai/trace/{self.trace_id}"
        print_link(trace_url, "🔗 Eval Started")


# Re-export for backwards compatibility with trace module
__all__ = ["EvalContext", "get_current_trace_headers"]
