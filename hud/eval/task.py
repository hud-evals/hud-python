"""Task - A runnable evaluation unit (data class).

A Task holds the configuration needed to run an evaluation:
- Environment configuration (how to create/connect)
- Optional scenario name and args

When entered as a context manager, it creates an EvalContext.

Usage:
    env = Environment("my-env").connect_hub("browser")

    # Empty - just env
    async with env() as ctx:
        await ctx.call_tool("navigate", url="...")

    # With scenario
    async with env("checkout", user_id="alice") as ctx:
        await agent.run(ctx.prompt)

    # Orchestrated via hud.eval
    tasks = [env("checkout", user_id="alice"), env("checkout", user_id="bob")]
    async with hud.eval(tasks, variants={"model": ["gpt-4o"]}, group=4) as ctx:
        ...
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from hud.types import MCPToolCall

if TYPE_CHECKING:
    from types import TracebackType

    from hud.environment import Environment
    from hud.eval.context import EvalContext

__all__ = ["Task", "build_eval_name"]

logger = logging.getLogger(__name__)


def _warn_local_mcp(mcp_config: dict[str, Any] | None) -> None:
    """Warn if mcp_config uses local MCP servers (command without url).

    Local MCP servers can cause port conflicts when running tasks concurrently.
    """
    if not mcp_config:
        return

    has_local = any(
        isinstance(server_cfg, dict) and "command" in server_cfg and not server_cfg.get("url")
        for server_cfg in mcp_config.values()
        if isinstance(server_cfg, dict)
    )

    if has_local:
        import warnings

        warnings.warn(
            "Task uses local MCP configuration (command without url). "
            "This may cause port conflicts when running tasks concurrently. "
            "Consider using remote MCP servers for parallel execution.",
            UserWarning,
            stacklevel=4,  # Skip through from_v4 -> _warn_local_mcp -> warn
        )


def build_eval_name(scenario: str | None, args: dict[str, Any] | None) -> str:
    """Build descriptive name: 'scenario with val1, val2, ...'"""
    if not scenario:
        return "eval"
    if not args:
        return scenario

    val_parts = []
    for v in list(args.values())[:3]:  # Max 3 values
        v_str = repr(v) if isinstance(v, str) else str(v)
        if len(v_str) > 25:
            v_str = v_str[:22] + "..."
        val_parts.append(v_str)

    if val_parts:
        return f"{scenario} with {', '.join(val_parts)}"
    return scenario


@dataclass
class Task:
    """A runnable evaluation unit (data class).

    Simplified v5 Task format:
    - env: Environment instance OR EnvConfig with hub name + filters
    - scenario: Scenario name to run
    - args: Scenario arguments
    - validation: Optional list of tool calls representing successful completion

    When entered as a context manager, creates an EvalContext.

        Attributes:
        id: Optional task identifier for filtering/tracking
        env: Environment instance (auto-created from dict/EnvConfig in __post_init__)
        scenario: Scenario name to run (from @env.scenario)
        args: Scenario arguments
        validation: Optional list of MCPToolCall objects representing successful completion

    Example (v5 format):
        ```python
        from hud.eval import Task

        # Pass dict - auto-converts to Environment
        task = Task(
            env={"name": "browser", "include": ["navigate", "screenshot"]},
            scenario="checkout",
            args={"user_id": "alice"},
            validation=[{"name": "check_cart", "arguments": {}}],
        )
        # task.env is now Environment connected to browser hub!

        # Or pass live Environment directly
        env = Environment("my-env").connect_hub("browser")
        task = Task(env=env, scenario="checkout", args={"user_id": "alice"})
        ```

    Migration from v4:
        Use Task.from_v4() to convert LegacyTask objects:

        ```python
        task = Task.from_v4(legacy_task)
        # or
        task = Task.from_v4({"prompt": "...", "mcp_config": {...}, ...})
        ```
    """

    # Core v5 task definition
    id: str | None = None
    env: Environment | None = None
    scenario: str | None = None
    args: dict[str, Any] = field(default_factory=dict)
    validation: list[MCPToolCall] | None = None

    # EvalContext creation params (set by hud.eval for parallel execution)
    trace_id: str | None = field(default=None, repr=False)
    api_key: str | None = field(default=None, repr=False)
    job_id: str | None = field(default=None, repr=False)
    group_id: str | None = field(default=None, repr=False)
    index: int = field(default=0, repr=False)
    variants: dict[str, Any] = field(default_factory=dict, repr=False)
    code_snippet: str | None = field(default=None, repr=False)
    _suppress_link: bool = field(default=False, repr=False)
    _trace: bool = field(default=True, repr=False)
    _quiet: bool = field(default=False, repr=False)

    # Runtime state
    _ctx: EvalContext | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        """Validate and normalize env and validation fields after initialization.

        Auto-converts dict or EnvConfig to Environment by connecting to the hub.
        Auto-converts validation dicts to MCPToolCall objects.
        """
        from hud.environment import Environment
        from hud.environment.types import EnvConfig

        # Convert env field
        if not isinstance(self.env, (Environment, type(None))):
            # Convert dict to EnvConfig first (with validation)
            if isinstance(self.env, dict):
                try:
                    config = EnvConfig(**self.env)
                except Exception as e:
                    raise ValueError(
                        f"Invalid env config: {e}. Expected fields: name (str), "
                        f"include (list[str] | None), exclude (list[str] | None)"
                    ) from e
            elif isinstance(self.env, EnvConfig):
                config = self.env
            else:
                raise TypeError(
                    f"Task.env must be Environment, EnvConfig, dict, or None. "
                    f"Got {type(self.env).__name__}"
                )

            # Convert EnvConfig to Environment
            env = Environment(config.name)
            env.connect_hub(config.name, include=config.include, exclude=config.exclude)
            self.env = env

        # Convert validation dicts to MCPToolCall objects
        if self.validation and isinstance(self.validation, list):
            converted_validation = []
            for item in self.validation:
                if isinstance(item, dict):
                    converted_validation.append(MCPToolCall(**item))
                elif isinstance(item, MCPToolCall):
                    converted_validation.append(item)
                else:
                    raise TypeError(
                        f"validation items must be dict or MCPToolCall, got {type(item).__name__}"
                    )
            self.validation = converted_validation

    @classmethod
    def from_v4(
        cls,
        source: Any,  # LegacyTask | dict[str, Any] | str
    ) -> Task:
        """Convert a v4 LegacyTask to a v5 Task.

        This is the recommended migration path for existing v4 code. The returned
        Task automatically runs setup_tool at the start and evaluate_tool at the
        end, matching the old LegacyTask behavior.

        Args:
            source: One of:
                - LegacyTask object
                - dict with LegacyTask fields (prompt, mcp_config, etc.)
                - JSON string of LegacyTask fields

        Returns:
            Task with Environment configured to mimic LegacyTask behavior.

        Example:
            ```python
            from hud.eval import Task

            # From existing LegacyTask
            task = Task.from_v4(legacy_task)

            # From dict (e.g., loaded from JSON file)
            task = Task.from_v4(
                {
                    "prompt": "Navigate to google.com",
                    "mcp_config": {"hud": {...}},
                    "setup_tool": {"name": "navigate", "arguments": {"url": "..."}},
                    "evaluate_tool": {"name": "check_url", "arguments": {}},
                }
            )

            # Use with hud.eval() or as context manager
            async with task as ctx:
                result = await agent.run(ctx)
            ```

        Note:
            For new code, prefer using @env.scenario() instead:
            - setup_tool code goes BEFORE the first yield
            - evaluate_tool code goes AFTER the first yield
            See https://docs.hud.ai/migration for the full migration guide.
        """
        import json as json_module

        from hud.environment import Environment
        from hud.types import LegacyTask

        # Parse JSON string
        if isinstance(source, str):
            try:
                source = json_module.loads(source)
            except json_module.JSONDecodeError as e:
                from hud.shared.exceptions import HudConfigError

                raise HudConfigError(f"Invalid JSON string for Task.from_v4: {e}") from e

        # Convert dict to LegacyTask (suppress the deprecation warning since we're migrating)
        if isinstance(source, dict):
            import warnings

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=DeprecationWarning)
                legacy_task = LegacyTask(**source)
        elif isinstance(source, LegacyTask):
            legacy_task = source
        else:
            raise TypeError(
                f"Task.from_v4() expects LegacyTask, dict, or JSON string, "
                f"got {type(source).__name__}"
            )

        # Warn if using local MCP configs (command without url)
        _warn_local_mcp(legacy_task.mcp_config)

        # Create Environment and connect via mcp_config
        env = Environment(legacy_task.id or "v4-legacy")
        env.connect_mcp_config(legacy_task.mcp_config)

        # Set the prompt
        env.prompt = legacy_task.prompt

        # Add setup_tool calls (run after connection via Environment._setup_calls)
        if legacy_task.setup_tool:
            setup_calls = legacy_task.setup_tool
            if not isinstance(setup_calls, list):
                setup_calls = [setup_calls]
            for call in setup_calls:
                env.setup_tool(call.name, **(call.arguments or {}))

        # Add evaluate_tool calls (run before disconnection via Environment._evaluate_calls)
        if legacy_task.evaluate_tool:
            evaluate_calls = legacy_task.evaluate_tool
            if not isinstance(evaluate_calls, list):
                evaluate_calls = [evaluate_calls]
            for call in evaluate_calls:
                env.evaluate_tool(call.name, **(call.arguments or {}))

        logger.debug(
            "Created Task from v4 LegacyTask: %s",
            legacy_task.prompt[:50] if legacy_task.prompt else "no prompt",
        )

        return cls(
            id=legacy_task.id,
            env=env,  # Live Environment with mcp_config, setup_tool, evaluate_tool
            scenario=None,  # No scenario - uses prompt directly
            args={},
            validation=None,
        )

    # Backwards compat alias

    def copy(self) -> Task:
        """Create a copy of this Task for parallel execution."""
        return Task(
            env=self.env,  # Share reference - from_environment handles copying
            scenario=self.scenario,
            args=self.args.copy(),
            trace_id=None,  # Each copy gets unique trace_id
            api_key=self.api_key,
            job_id=self.job_id,
            group_id=self.group_id,
            index=self.index,
            variants=self.variants.copy(),
            code_snippet=self.code_snippet,
            _suppress_link=self._suppress_link,
            _trace=self._trace,
            _quiet=self._quiet,
        )

    def to_eval_context(self) -> EvalContext:
        """Convert this Task to an EvalContext.

        Creates an EvalContext from the environment (live or from config).
        If env is EnvConfig or dict, creates Environment by connecting to the hub.
        """
        from hud.environment import Environment
        from hud.eval.context import EvalContext

        # Get environment (or create blank if None)
        source_env = self.env if self.env is not None else Environment("eval")

        eval_name = build_eval_name(self.scenario, self.args)

        # Create EvalContext from environment
        ctx = EvalContext.from_environment(
            env=source_env,
            name=eval_name,
            trace_id=self.trace_id,
            api_key=self.api_key,
            job_id=self.job_id,
            group_id=self.group_id,
            index=self.index,
            variants=self.variants,
            code_snippet=self.code_snippet,
        )
        ctx._suppress_link = self._suppress_link
        ctx._trace_enabled = self._trace

        return ctx

    async def __aenter__(self) -> EvalContext:
        """Enter eval context.

        Order of operations:
        1. Create EvalContext from environment config
        2. Connect environment (MCP servers, etc.)
        3. Run scenario setup (if scenario) → sets ctx.prompt
        4. Notify backend (with prompt now set)
        5. Print trace link
        """
        self._ctx = self.to_eval_context()
        await self._ctx.__aenter__()  # Connect env, set trace headers

        # Run scenario setup (sets prompt)
        if self.scenario:
            await self._run_scenario_setup()

        # Notify backend with prompt included
        await self._ctx._eval_enter()
        self._ctx._print_eval_link()

        return self._ctx

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Exit eval context - run scenario evaluate and exit EvalContext."""
        if self._ctx is None:
            return

        # If we have a scenario and no error, run its evaluate phase
        if self.scenario and exc_type is None:
            await self._run_scenario_evaluate()

        # Exit the EvalContext
        await self._ctx.__aexit__(exc_type, exc_val, exc_tb)
        self._ctx = None

    async def _run_scenario_setup(self) -> None:
        """Run the scenario's setup phase (get prompt)."""
        if self._ctx is None or self.scenario is None:
            return

        # Store scenario name on context for ctx.submit()
        self._ctx._scenario_name = self.scenario

        # Delegate to ScenarioMixin.run_scenario_setup
        prompt = await self._ctx.run_scenario_setup(self.scenario, self.args)
        if prompt:
            self._ctx.prompt = prompt

    async def _run_scenario_evaluate(self) -> None:
        """Run the scenario's evaluate phase (get reward)."""
        if self._ctx is None or self.scenario is None:
            return

        # Delegate to ScenarioMixin.run_scenario_evaluate
        reward = await self._ctx.run_scenario_evaluate(self.scenario)
        if reward is not None:
            self._ctx.reward = reward
