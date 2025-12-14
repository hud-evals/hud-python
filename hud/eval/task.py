"""Task - A runnable evaluation unit (Pydantic model).

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
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from hud.types import MCPToolCall

if TYPE_CHECKING:
    from hud.environment import Environment
    from hud.environment.types import EnvConfig

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


class Task(BaseModel):
    """A runnable evaluation unit (Pydantic model).

    Simplified v5 Task format:
    - env: Environment instance OR EnvConfig with hub name + filters
    - scenario: Scenario name to run
    - args: Scenario arguments
    - validation: Optional list of tool calls representing successful completion

    When entered as a context manager, creates an EvalContext.

    Attributes:
        id: Optional task identifier for filtering/tracking
        env: Environment instance (auto-created from dict/EnvConfig via validator)
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

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Fields - env accepts Environment | EnvConfig | dict, auto-converts to Environment
    env: Any = Field(default=None)  # Typed as Any for input flexibility, validated below
    scenario: str | None = None
    id: str | None = None
    args: dict[str, Any] = Field(default_factory=dict)
    validation: list[MCPToolCall] | None = None

    @field_validator("env", mode="before")
    @classmethod
    def convert_env(
        cls, v: Environment | EnvConfig | dict[str, Any] | None
    ) -> Environment | None:
        """Auto-convert dict/EnvConfig to Environment."""
        from hud.environment import Environment
        from hud.environment.types import EnvConfig

        if v is None:
            return None
        if isinstance(v, Environment):
            return v
        if isinstance(v, dict):
            try:
                v = EnvConfig(**v)
            except Exception as e:
                raise ValueError(
                    f"Invalid env config: {e}. Expected fields: name (str), "
                    f"include (list[str] | None), exclude (list[str] | None)"
                ) from e
        if isinstance(v, EnvConfig):
            env = Environment(v.name)
            env.connect_hub(v.name, include=v.include, exclude=v.exclude)
            return env
        raise TypeError(
            f"Task.env must be Environment, EnvConfig, or dict. Got {type(v).__name__}"
        )

    @field_validator("validation", mode="before")
    @classmethod
    def convert_validation(
        cls, v: list[MCPToolCall | dict[str, Any]] | None
    ) -> list[MCPToolCall] | None:
        """Auto-convert validation dicts to MCPToolCall objects."""
        if v is None:
            return None
        if not isinstance(v, list):
            raise TypeError(f"validation must be a list, got {type(v).__name__}")

        converted = []
        for item in v:
            if isinstance(item, dict):
                converted.append(MCPToolCall(**item))
            elif isinstance(item, MCPToolCall):
                converted.append(item)
            else:
                raise TypeError(
                    f"validation items must be dict or MCPToolCall, got {type(item).__name__}"
                )
        return converted

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
            env=env,  # Live Environment with mcp_config, setup_tool, evaluate_tool
            scenario=None,  # v4 tasks use prompt directly, not scenarios
            id=legacy_task.id,
            args={},
            validation=None,
        )

    def copy(self) -> Task:
        """Create a copy of this Task config.

        Note: env is shared (not deep copied) since Environment instances
        should be reused. Args and validation are deep copied.
        """
        return Task(
            id=self.id,
            env=self.env,  # Share reference
            scenario=self.scenario,
            args=self.args.copy() if self.args else {},
            validation=self.validation.copy() if self.validation else None,
        )
