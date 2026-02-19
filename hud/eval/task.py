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
from copy import deepcopy
from typing import TYPE_CHECKING, Any, cast

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_serializer,
    field_validator,
    model_serializer,
    model_validator,
)

from hud.types import MCPToolCall

if TYPE_CHECKING:
    from hud.environment import Environment
    from hud.environment.types import EnvConfig

__all__ = ["Task", "TaskAgentConfig", "build_eval_name"]

logger = logging.getLogger(__name__)


class TaskAgentConfig(BaseModel):
    """Agent configuration for a Task.

    Contains settings that should be passed to the agent when running this task.

    Note: allowed_tools/disallowed_tools are handled at the Environment level
    (via env.include()/env.exclude() for v5, or extracted by build_env_from_v4() for v4).
    """

    model_config = ConfigDict(extra="ignore")

    system_prompt: str | None = Field(
        default=None,
        description="Custom system prompt to pass to the agent",
    )

    # Agent behavior settings (from v4 agent_config, applied by EvalContext)
    append_setup_output: bool = Field(
        default=False,
        description="Append setup tool output to the agent's initial prompt",
    )
    append_setup_tool: bool = Field(
        default=False,
        description="Alias for append_setup_output (backwards compat)",
    )

    @model_validator(mode="before")
    @classmethod
    def warn_extra_fields(cls, data: Any) -> Any:
        """Warn about extra fields that will be ignored."""
        if isinstance(data, dict):
            known_fields = {
                "system_prompt",
                "append_setup_output",
                "append_setup_tool",
            }
            extra = set(data.keys()) - known_fields
            if extra:
                logger.warning(
                    "Deprecated or unknown fields in agent_config will be ignored: %s",
                    ", ".join(sorted(extra)),
                )
        return data


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
        id: Internal platform task version identifier
        slug: Stable user-defined task identifier for filtering and sync
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
    id: str | None = Field(
        default=None,
        description="Internal platform task version ID. Reserved for platform-assigned values.",
    )
    slug: str | None = Field(
        default=None,
        max_length=20,
        description="Stable task slug for task filtering and sync workflows.",
    )
    args: dict[str, Any] | None = Field(
        default=None,
        description="Scenario arguments. None indicates a template (args filled in later).",
    )
    validation: list[MCPToolCall] | None = None

    # Agent config - settings passed to agent (system_prompt, etc.)
    # Accepts TaskAgentConfig or dict (auto-converted via validator)
    agent_config: TaskAgentConfig | dict[str, Any] | None = None

    # Task metadata - for tracking/filtering, not used by agent
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("agent_config", mode="before")
    @classmethod
    def convert_agent_config(
        cls, v: TaskAgentConfig | dict[str, Any] | None
    ) -> TaskAgentConfig | None:
        """Auto-convert dict to TaskAgentConfig."""
        if v is None:
            return None
        if isinstance(v, TaskAgentConfig):
            return v
        if isinstance(v, dict):
            return TaskAgentConfig(**v)
        raise TypeError(
            f"Task.agent_config must be TaskAgentConfig or dict. Got {type(v).__name__}"
        )

    @model_validator(mode="before")
    @classmethod
    def detect_v4_format(cls, data: Any) -> Any:
        """Auto-detect v4 LegacyTask format and convert to v5 Task format.

        If the input dict is a valid v4 format (has prompt, mcp_config, evaluate_tool),
        it's converted using build_env_from_v4().

        This allows Task(**v4_dict) to work seamlessly.
        """
        from hud.eval.utils import build_env_from_v4, is_v4_format, validate_v4_task

        if not isinstance(data, dict):
            return data

        if is_v4_format(data):
            # Validate completeness before conversion
            validate_v4_task(data)
            # build_env_from_v4 returns a dict with all Task fields
            return build_env_from_v4(data)

        return data

    @field_validator("env", mode="before")
    @classmethod
    def convert_env(cls, v: Environment | EnvConfig | dict[str, Any] | None) -> Environment | None:
        """Auto-convert dict/EnvConfig to Environment.

        Format: {"name": "browser", "include": [...], "exclude": [...]}
        """
        from hud.environment import Environment
        from hud.environment.types import EnvConfig

        if v is None:
            return None
        if isinstance(v, Environment):
            return v
        if isinstance(v, dict):
            try:
                config = EnvConfig(**v)
            except Exception as e:
                raise ValueError(
                    f"Invalid env config: {e}. Expected fields: name (str), "
                    f"include (list[str] | None), exclude (list[str] | None)"
                ) from e
            env = Environment(config.name)
            env.connect_hub(config.name, include=config.include, exclude=config.exclude)
            return env
        if isinstance(v, EnvConfig):
            env = Environment(v.name)
            env.connect_hub(v.name, include=v.include, exclude=v.exclude)
            return env
        raise TypeError(f"Task.env must be Environment, EnvConfig, or dict. Got {type(v).__name__}")

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

    @field_serializer("env")
    def serialize_env(self, env: Environment | None) -> dict[str, Any] | None:
        """Serialize Environment to config dict via to_config()."""
        if env is None:
            return None
        return env.to_config()

    @model_serializer(mode="wrap")
    def _serialize_task(
        self,
        handler: Any,  # SerializerFunctionWrapHandler
    ) -> dict[str, Any]:
        """Custom serializer for v4 format flattening.

        For v5 tasks: uses default serialization (env field handled by field_serializer)
        For v4 tasks: flattens {"prompt": ..., "mcp_config": ..., "evaluate_tool": ...}
        """
        # Get default serialization (env is already converted by field_serializer)
        data = handler(self)

        # Check if this is a v4 task (env config has mcp_config)
        env_config = data.get("env")
        if env_config and isinstance(env_config, dict) and "mcp_config" in env_config:
            # v4 format - flatten into top-level dict
            result = env_config.copy()

            # Map validation → integration_test_tool
            if self.validation:
                result["integration_test_tool"] = [
                    {"name": v.name, "arguments": v.arguments or {}} for v in self.validation
                ]

            # Preserve agent_config
            agent_config: dict[str, Any] = {}
            if data.get("agent_config"):
                agent_config.update(data["agent_config"])
            # Restore tool filters from Environment (they were extracted during v4 conversion)
            if self.env is not None:
                if getattr(self.env, "_agent_include", None) is not None:
                    agent_config["allowed_tools"] = self.env._agent_include
                elif "allowed_tools" not in agent_config:
                    # ["*"] was converted to None, restore it for serialization
                    agent_config["allowed_tools"] = ["*"]
                if getattr(self.env, "_agent_exclude", None) is not None:
                    agent_config["disallowed_tools"] = self.env._agent_exclude
            if agent_config:
                result["agent_config"] = agent_config

            # Preserve metadata
            if data.get("metadata"):
                result["metadata"] = data["metadata"]

            # Preserve id
            if data.get("id"):
                result["id"] = data["id"]
            # Preserve slug
            if data.get("slug"):
                result["slug"] = data["slug"]

            return result

        return data

    @classmethod
    def from_v4(cls, source: Any) -> Task:
        """Convert v4 LegacyTask format to v5 Task.

        This is a convenience wrapper. You can also use Task(**dict) directly
        since the model validator auto-detects v4 format.

        Args:
            source: LegacyTask, dict, or JSON string with v4 fields

        Returns:
            Task configured for v4 behavior
        """
        import json as json_module

        # JSON string → dict
        if isinstance(source, str):
            source = json_module.loads(source)

        # LegacyTask → dict (import only when needed)
        if hasattr(source, "model_dump"):
            source = source.model_dump()

        # Model validator handles v4 detection and conversion
        return cls(**source)

    def copy(
        self,
        *,
        include: Any = None,
        exclude: Any = None,
        update: dict[str, Any] | None = None,
        deep: bool = False,
    ) -> Task:
        """Create a copy of this Task config.

        Note: env is shared (not deep copied) since Environment instances
        should be reused. Args and validation are deep copied.
        Task identity fields (id, slug) are reset unless explicitly provided
        in ``update``.
        """
        update_data = dict(update or {})
        update_data.setdefault("id", None)
        update_data.setdefault("slug", None)

        if include is not None or exclude is not None:
            # BaseModel.model_copy() does not support include/exclude. Build
            # through dump+validate to preserve callers that rely on filtering.
            data = self.model_dump(mode="python", include=include, exclude=exclude)
            if deep:
                if isinstance(data.get("args"), dict):
                    data["args"] = deepcopy(data["args"])
                if isinstance(data.get("validation"), list):
                    data["validation"] = deepcopy(data["validation"])
            data.update(update_data)
            return cast("Task", type(self).model_validate(data))

        if update is not None or deep:
            # Preserve validation and env-sharing semantics.
            data = self.model_dump(mode="python")
            # Keep the existing Environment reference unless explicitly overridden.
            if "env" not in update_data:
                data["env"] = self.env
            if isinstance(data.get("args"), dict):
                data["args"] = deepcopy(data["args"]) if deep else data["args"].copy()
            if isinstance(data.get("validation"), list):
                data["validation"] = (
                    deepcopy(data["validation"]) if deep else data["validation"].copy()
                )
            data.update(update_data)
            return cast("Task", type(self).model_validate(data))

        return Task(
            id=None,
            slug=None,
            env=self.env,  # Share reference
            scenario=self.scenario,
            args=self.args.copy() if self.args is not None else None,
            validation=self.validation.copy() if self.validation else None,
        )
