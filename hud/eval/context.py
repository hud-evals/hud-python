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
from typing import TYPE_CHECKING, Any, Self

from hud.environment import Environment
from hud.settings import settings
from hud.shared import make_request
from hud.telemetry import flush, instrument

if TYPE_CHECKING:
    from types import TracebackType

    from hud.eval.task import Task
    from hud.types import MCPToolResult


from hud.eval.types import EvalExitPayload, EvalPayload, ParallelEvalComplete

logger = logging.getLogger(__name__)

# Contextvar to store current trace headers (for httpx auto-instrumentation)
_current_trace_headers: contextvars.ContextVar[dict[str, str] | None] = contextvars.ContextVar(
    "current_trace_headers", default=None
)

# Contextvar to store current api_key override (for telemetry exporter)
_current_api_key: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "current_api_key", default=None
)


def get_current_trace_headers() -> dict[str, str] | None:
    """Get the current trace headers from context."""
    return _current_trace_headers.get()


def get_current_trace_id() -> str | None:
    """Get the current trace ID (task_run_id) from context.

    Returns the Trace-Id if inside an eval context, None otherwise.
    Used by @instrument decorator to know where to send telemetry.
    """
    headers = _current_trace_headers.get()
    if headers:
        return headers.get("Trace-Id")
    return None


def get_current_api_key() -> str | None:
    """Get the current API key override from context.

    Returns the api_key if one was passed to hud.eval(), otherwise None.
    Falls back to settings.api_key if not in an eval context.
    Used by telemetry exporter for uploads.
    """
    return _current_api_key.get()


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
        results: All eval results (populated for parallel execution, empty for single)
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
        self.job_id: str | None = job_id

        self.group_id: str | None = group_id
        self.index: int = index

        # Variant assignment
        self.variants: dict[str, Any] = variants or {}

        # User-settable (per-run values, override Environment defaults)
        self.prompt: str | None = None  # From scenario setup or task
        self.reward: float | None = None
        self.answer: str | None = None  # Agent's submitted answer
        self.system_prompt: str | None = None  # From task.agent_config, passed to agent

        # Error tracking
        self.error: BaseException | None = None

        # User metadata (arbitrary key-value pairs)
        self.metadata: dict[str, Any] = {}

        # Parallel results (empty list for single evals, populated for parallel)
        self.results: list[EvalContext] = []

        # Code snippet for reproducibility
        self.code_snippet: str | None = code_snippet

        # Private state for eval tracking
        self._eval_api_key = api_key
        self._token: contextvars.Token[dict[str, str] | None] | None = None
        self._api_key_token: contextvars.Token[str | None] | None = None
        self._is_summary: bool = False  # True for summary contexts (skip trace)
        self._suppress_link: bool = quiet  # True to suppress printing eval link
        self._trace_enabled: bool = trace  # Whether to send trace data to backend
        self._source_env_name: str | None = None  # Source env name for remote lookups
        self._task: Task | None = None  # Task config (set by from_task)

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
            trace=trace,
            quiet=quiet,
        )

        # Copy connections from parent - each connector is copied so parallel
        # execution gets fresh client instances
        ctx._connections = {name: connector.copy() for name, connector in env._connections.items()}
        ctx._setup_calls = env._setup_calls.copy()
        ctx._evaluate_calls = env._evaluate_calls.copy()

        # Copy scenarios (definitions) by reference - they don't change
        ctx._scenarios = getattr(env, "_scenarios", {})
        # Create fresh session state for this eval (parallel evals each need their own)
        ctx._scenario_sessions = {}
        ctx._scenario_latest = {}
        ctx._scenario_answers = {}

        # Store source env name for remote scenario lookups
        ctx._source_env_name = env.name

        # Copy managers by reference (they hold local tools, prompts, resources)
        # This allows ctx.call_tool(), ctx.get_prompt(), ctx.read_resource() to work
        # for locally defined tools/scenarios
        ctx._tool_manager = env._tool_manager
        ctx._prompt_manager = env._prompt_manager
        ctx._resource_manager = env._resource_manager

        # Copy prompt
        if env.prompt:
            ctx.prompt = env.prompt

        # Copy agent-level tool filters (allowed_tools/disallowed_tools)
        ctx._agent_include = getattr(env, "_agent_include", None)
        ctx._agent_exclude = getattr(env, "_agent_exclude", None)

        # Copy router's conflict resolution strategy
        ctx._router.conflict_resolution = env._router.conflict_resolution

        # Copy mock mode settings (for testing)
        ctx._mock_mode = getattr(env, "_mock_mode", False)
        ctx._mock_outputs = getattr(env, "_mock_outputs", {}).copy()
        ctx._mock_tool_schemas = getattr(env, "_mock_tool_schemas", {}).copy()

        # Copy hub config (needed to detect remote hub for telemetry)
        ctx._hub_config = getattr(env, "_hub_config", None)

        # Copy mcp config (needed to detect remote HUD MCP for telemetry)
        ctx._mcp_config = getattr(env, "_mcp_config", None)

        return ctx

    @classmethod
    def from_task(
        cls,
        task: Task,
        *,
        name: str | None = None,
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
        """Create an EvalContext from a Task config.

        Args:
            task: Task config (env, scenario, args)
            name: Override for eval/trace name (defaults to task scenario/args)
            trace_id: Unique trace ID
            api_key: API key for backend calls
            job_id: Job ID to link to
            group_id: Group ID for parallel evaluations
            index: Index in parallel execution
            variants: Variant assignment
            code_snippet: Code being evaluated
            trace: Whether to send traces to backend
            quiet: Whether to suppress output
        """
        from hud.environment import Environment
        from hud.eval.task import build_eval_name

        eval_name = name or build_eval_name(task.scenario, task.args)

        # task.env is guaranteed to be Environment after Task.__post_init__
        assert isinstance(task.env, Environment), "Task.env should be Environment"

        ctx = cls.from_environment(
            env=task.env,
            name=eval_name,
            trace_id=trace_id,
            api_key=api_key,
            job_id=job_id,
            group_id=group_id,
            index=index,
            variants=variants,
            code_snippet=code_snippet,
            trace=trace,
            quiet=quiet,
        )

        # Store task info for scenario execution
        ctx._task = task

        # Set system_prompt from task.agent_config
        if task.agent_config:
            if isinstance(task.agent_config, dict):
                if task.agent_config.get("system_prompt"):
                    ctx.system_prompt = task.agent_config["system_prompt"]
            elif task.agent_config.system_prompt:
                ctx.system_prompt = task.agent_config.system_prompt

        return ctx

    async def _run_task_scenario_setup(self) -> None:
        """Run the task's scenario setup phase (if scenario provided)."""
        if self._task is None or self._task.scenario is None:
            return

        prompt = await self.run_scenario_setup(self._task.scenario, self._task.args)
        if prompt:
            self.prompt = prompt

    async def _run_task_scenario_evaluate(self) -> None:
        """Run the task's scenario evaluate phase (if scenario provided)."""
        if self._task is None or self._task.scenario is None:
            return

        reward = await self.run_scenario_evaluate(self._task.scenario)
        if reward is not None:
            self.reward = reward

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
    def success(self) -> bool:
        """True if no error occurred."""
        return self.error is None

    @property
    def has_scenario(self) -> bool:
        """True if a scenario is running and can accept submissions."""
        return self._task is not None and self._task.scenario is not None

    # =========================================================================
    # Backend Integration
    # =========================================================================

    def _get_eval_api_key(self) -> str | None:
        return self._eval_api_key or settings.api_key

    def _build_base_payload(self) -> EvalPayload:
        """Build the base payload for enter/exit."""
        return EvalPayload(
            prompt=self.prompt,
            code_snippet=self.code_snippet,
            job_id=self.job_id,
            group_id=self.group_id,
            variants=self.variants if self.variants else None,
            # Only send task_version_id for v5 tasks (those with scenarios).
            # v4 tasks have client-side IDs that shouldn't be sent to backend.
            task_version_id=self._task.id if self._task and self._task.scenario else None,
            metadata=self.metadata if self.metadata else None,
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
        """Submit the agent's answer for scenario evaluation.

        Delegates to Environment.submit() with the current scenario name.
        The answer will be passed to the scenario's evaluate phase via
        `yield`, e.g.: `answer = yield "Do the task"`

        Args:
            answer: The agent's final answer/result to submit

        Example:
            async with env("checkout", product="laptop") as ctx:
                response = await agent.run(ctx.prompt)
                await ctx.submit(response)
            # On exit, scenario's evaluate phase receives the answer
        """
        if not self._task or not self._task.scenario:
            return

        # Store answer on context for display
        self.answer = answer

        # Delegate to Environment.submit() which handles storage + broadcast
        await super().submit(self._task.scenario, answer)

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
        self._token = _current_trace_headers.set(self.headers)
        self._api_key_token = _current_api_key.set(self._eval_api_key)

        # Connect environment (MCP servers, tools)
        await super().__aenter__()

        # Run task scenario setup (if created from_task with scenario)
        await self._run_task_scenario_setup()

        # Notify backend and print link
        await self._eval_enter()
        self._print_eval_link()

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

        # Run task scenario evaluate (if no error and has scenario)
        if exc_type is None:
            await self._run_task_scenario_evaluate()

        # Track error
        error_msg: str | None = None
        if exc_type is not None:
            self.error = exc_val
            error_msg = str(exc_val) if exc_val else "Unknown error"

        # Flush any pending telemetry spans for this trace
        flush(self.trace_id)

        # Disconnect environment (parent class)
        await super().__aexit__(exc_type, exc_val, exc_tb)

        # Reset context vars
        if self._token is not None:
            _current_trace_headers.reset(self._token)
            self._token = None
        if self._api_key_token is not None:
            _current_api_key.reset(self._api_key_token)
            self._api_key_token = None

        # Notify backend
        await self._eval_exit(error_msg)

        # Print single eval result summary (unless suppressed for parallel evals)
        self._print_single_result(error_msg)

        return False

    # =========================================================================
    # Tool Call Instrumentation
    # =========================================================================

    async def _execute_tool(self, name: str, arguments: dict[str, Any]) -> MCPToolResult:
        """Execute a tool with automatic telemetry recording.

        Overrides Environment._execute_tool to record MCP spans for the eval context.
        Instrumentation is disabled when connected to a remote HUD server (telemetry is
        recorded server-side in that case).
        """
        # Skip instrumentation when connected to a remote hub - telemetry is handled server-side
        if self._hub_config is not None:
            return await super()._execute_tool(name, arguments)

        # Skip instrumentation for v4 tasks with HUD MCP config (remote server)
        if self._mcp_config is not None:
            from hud.utils.mcp import _is_hud_server

            for server_cfg in self._mcp_config.values():
                if isinstance(server_cfg, dict):
                    url = server_cfg.get("url", "")
                    if url and _is_hud_server(url):
                        return await super()._execute_tool(name, arguments)

        # For local environments, record MCP spans
        return await self._execute_tool_instrumented(name, arguments)

    @instrument(category="mcp")
    async def _execute_tool_instrumented(
        self, name: str, arguments: dict[str, Any]
    ) -> MCPToolResult:
        """Instrumented version of _execute_tool for local environments."""
        return await super()._execute_tool(name, arguments)

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

    def _print_single_result(self, error_msg: str | None) -> None:
        """Print a single eval result summary."""
        # Skip if link printing is suppressed (e.g., parallel child traces)
        if self._suppress_link:
            return

        from hud.eval.display import print_single_result

        print_single_result(
            trace_id=self.trace_id,
            name=self.eval_name,
            reward=self.reward,
            error=error_msg,
        )


# Re-export for backwards compatibility with trace module
__all__ = [
    "EvalContext",
    "get_current_api_key",
    "get_current_trace_headers",
    "get_current_trace_id",
]
