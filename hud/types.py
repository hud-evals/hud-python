from __future__ import annotations

import json
import logging
import uuid
from enum import Enum
from typing import Any, Literal

import mcp.types as types
from mcp.types import CallToolRequestParams, CallToolResult
from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator

from hud.settings import settings
from hud.utils.env import resolve_env_vars as _resolve_env_vars
from hud.utils.tool_shorthand import normalize_to_tool_call_dict

logger = logging.getLogger(__name__)

# Guard to ensure we only log missing HUD_API_KEY once
_missing_api_key_error_logged: bool = False


class AgentType(str, Enum):
    CLAUDE = "claude"
    OPENAI = "openai"
    OPERATOR = "operator"
    GEMINI = "gemini"
    GEMINI_CUA = "gemini_cua"
    OPENAI_COMPATIBLE = "openai_compatible"
    INTEGRATION_TEST = "integration_test"

    @property
    def cls(self) -> type:
        from hud.agents import OpenAIAgent, OperatorAgent
        from hud.agents.claude import ClaudeAgent
        from hud.agents.gemini import GeminiAgent
        from hud.agents.gemini_cua import GeminiCUAAgent
        from hud.agents.openai_chat import OpenAIChatAgent

        mapping: dict[AgentType, type] = {
            AgentType.CLAUDE: ClaudeAgent,
            AgentType.OPENAI: OpenAIAgent,
            AgentType.OPERATOR: OperatorAgent,
            AgentType.GEMINI: GeminiAgent,
            AgentType.GEMINI_CUA: GeminiCUAAgent,
            AgentType.OPENAI_COMPATIBLE: OpenAIChatAgent,
        }
        if self == AgentType.INTEGRATION_TEST:
            from hud.agents.misc.integration_test_agent import IntegrationTestRunner

            return IntegrationTestRunner
        if self not in mapping:
            raise ValueError(f"Unsupported agent type: {self}")
        return mapping[self]


class BaseAgentConfig(BaseModel):
    """Agent configuration for LLM-specific settings.

    Note: allowed_tools, disallowed_tools, append_setup_output, and initial_screenshot
    are kept for backwards compatibility with v4 task configs but are no longer applied
    at the agent level. These should be configured on the Environment/Task instead.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid", populate_by_name=True)

    # Model identifier - use 'model' (preferred) or 'checkpoint_name' (alias)
    model: str | None = Field(
        default=None, validation_alias=AliasChoices("model", "checkpoint_name")
    )
    model_name: str = "Agent"  # Human-readable display name

    # LLM-specific setting
    system_prompt: str | None = None

    # Deprecated: kept for backwards compat with v4 task configs, not applied by agent
    allowed_tools: list[str] | None = None
    disallowed_tools: list[str] | None = None
    append_setup_output: bool = True
    append_setup_tool: bool = True  # Alias for append_setup_output (backwards compat)
    initial_screenshot: bool = True

    @property
    def checkpoint_name(self) -> str | None:
        """Alias for model (for backwards compatibility)."""
        return self.model


class LegacyTask(BaseModel):
    """
    DEPRECATED: Use Task from env() instead.

    A task configuration that can be used to create a task.

    The mcp_config field supports environment variable substitution using
    template placeholders in the format ${VAR_NAME} or ${VAR_NAME:default_value}.

    .. deprecated:: 0.5.0
        LegacyTask is deprecated in v0.5.0 and will be removed in v0.6.0
        (no earlier than March 1st, 2026).

        Use one of these migration paths:

        1. Quick conversion: ``Task.from_v4(legacy_task)`` converts LegacyTask to Task
        2. Full migration: Use ``@env.scenario()`` with setup code before first yield
           and evaluate code after first yield

        See https://docs.hud.ai/migration for the full migration guide.

    Example (deprecated):
        mcp_config: {
            "hud": {
                "url": "${HUD_MCP_URL:https://mcp.hud.ai/v3/mcp}",
                "headers": {
                    "Authorization": "Bearer ${HUD_API_KEY}",
                    "Mcp-Image": "your-mcp-image"
                }
            }
        }
    """

    id: str | None = None
    prompt: str
    mcp_config: dict[str, Any]
    setup_tool: MCPToolCall | list[MCPToolCall] | None = None
    evaluate_tool: MCPToolCall | list[MCPToolCall] | None = None
    integration_test_tool: MCPToolCall | list[MCPToolCall] | None = None
    agent_config: BaseAgentConfig | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    def __init__(self, **data: Any) -> None:
        """Initialize LegacyTask with deprecation warning."""
        import warnings

        warnings.warn(
            "LegacyTask is deprecated in v0.5.0 and will be removed in v0.6.0 "
            "(no earlier than March 1st, 2026). "
            "Use Task.from_v4() for quick conversion, or migrate to @env.scenario(). "
            "See https://docs.hud.ai/migration for details.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(**data)

    @field_validator("mcp_config", "metadata", mode="before")
    @classmethod
    def parse_json_strings(cls, v: Any) -> Any:
        """Parse JSON strings into dictionaries."""
        if isinstance(v, str):
            try:
                return json.loads(v)
            except json.JSONDecodeError as e:
                from hud.shared.exceptions import HudConfigError

                raise HudConfigError(f"Invalid JSON string: {e}") from e
        return v

    @field_validator("agent_config", mode="before")
    @classmethod
    def parse_agent_config(cls, v: Any) -> BaseAgentConfig | None:
        """Parse agent_config into BaseAgentConfig."""
        if v is None:
            return None
        if isinstance(v, BaseAgentConfig):
            return v
        if isinstance(v, str):
            try:
                v = json.loads(v)
            except json.JSONDecodeError as e:
                from hud.shared.exceptions import HudConfigError

                raise HudConfigError(f"Invalid JSON string for agent_config: {e}") from e
        if isinstance(v, dict):
            return BaseAgentConfig(**v)
        return v

    @field_validator("setup_tool", "evaluate_tool", "integration_test_tool", mode="before")
    @classmethod
    def convert_dict_to_tool_call(cls, v: Any, info: Any) -> Any:
        """Convert dict (with shorthands) to MCPToolCall instance.

        Supports nested forms by walking to the deepest tool name and its arguments.
        Examples:
        - {"name": "navigate", "arguments": {...}} -> name=navigate
        - {"navigate": {...}} -> name=navigate
        - {"setup": {"navigate": {...}}} -> name=navigate
        - {"name": "setup", "arguments": {"name": "navigate", "arguments": {...}}}
          -> name=navigate
        - Lists are normalized element-wise
        """
        if v is None:
            return None

        # Parse JSON string if needed
        if isinstance(v, str):
            try:
                v = json.loads(v)
            except json.JSONDecodeError as e:
                from hud.shared.exceptions import HudConfigError

                raise HudConfigError(f"Invalid JSON string: {e}") from e

        normalized = normalize_to_tool_call_dict(v)

        if isinstance(normalized, dict):
            return MCPToolCall(**normalized)
        if isinstance(normalized, list):
            return [MCPToolCall(**item) if isinstance(item, dict) else item for item in normalized]
        return v

    @field_validator("mcp_config", mode="before")
    @classmethod
    def resolve_env_vars(cls, v: dict[str, Any]) -> dict[str, Any]:
        """
        Automatically resolve environment variables in mcp_config.

        Supports ${VAR_NAME} syntax with variable substitution from
        system environment variables and settings (including HUD_API_KEY, etc.)

        Missing variables resolve to empty strings.
        """
        # Warn once if HUD_API_KEY is not set
        if not settings.api_key:
            global _missing_api_key_error_logged
            if not _missing_api_key_error_logged:
                logger.error("HUD_API_KEY is not set, tracing and remote training will not work")
                _missing_api_key_error_logged = True

        return _resolve_env_vars(v)


class MCPToolCall(CallToolRequestParams):
    """A tool call."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))  # Unique identifier for reference

    def __str__(self) -> str:
        """Format tool call as plain text."""
        args_str = ""
        if self.arguments:
            try:
                args_str = json.dumps(self.arguments, separators=(",", ":"))
                if len(args_str) > 60:
                    args_str = args_str[:57] + "..."
            except (TypeError, ValueError):
                args_str = str(self.arguments)[:60]

        return f"→ {self.name}({args_str})"

    def __rich__(self) -> str:
        """Rich representation with color formatting."""
        from hud.utils.hud_console import hud_console

        return hud_console.format_tool_call(self.name, self.arguments)


class MCPToolResult(CallToolResult):
    """A tool result with optional call_id for correlation."""

    call_id: str | None = None  # For correlating with provider-specific tool call IDs

    def _get_content_summary(self) -> str:
        """Extract a summary of the content."""
        # Extract content summary
        content_summary = ""
        if self.content:
            for block in self.content:
                if isinstance(block, types.TextContent):
                    # Get first line or truncate
                    text = block.text.strip()
                    first_line = text.split("\n")[0] if "\n" in text else text
                    content_summary = first_line
                    break
                elif isinstance(block, types.ImageContent):
                    content_summary = "📷 Image"
                    break

        # Or use structured content if no text content
        if not content_summary and self.structuredContent:
            try:
                content_summary = json.dumps(self.structuredContent, separators=(",", ":"))
            except (TypeError, ValueError):
                content_summary = str(self.structuredContent)

        return content_summary

    def __str__(self) -> str:
        """Format tool result as plain text for compatibility."""
        content_summary = self._get_content_summary()

        # Plain text format with unicode symbols
        if self.isError:
            return f"✗ {content_summary}"
        else:
            return f"✓ {content_summary}"

    def __rich__(self) -> str:
        """Rich representation with color formatting."""
        from hud.utils.hud_console import hud_console

        content_summary = self._get_content_summary()
        return hud_console.format_tool_result(content_summary, self.isError)


class AgentResponse(BaseModel):
    """A model response in the conversation."""

    # --- FUNCTIONAL ---
    tool_calls: list[MCPToolCall] = Field(default_factory=list)
    done: bool = Field(default=False)

    # --- TELEMETRY [hud.ai] ---
    # Responses
    content: str | None = Field(default=None)
    reasoning: str | None = Field(default=None)
    info: dict[str, Any] = Field(default_factory=dict)
    isError: bool = Field(default=False)
    raw: Any | None = Field(default=None)  # Include raw response for access to Choice objects

    # Timestamps
    start_timestamp: str | None = None
    end_timestamp: str | None = None

    def __str__(self) -> str:
        response = ""
        if self.reasoning:
            response += f"Reasoning: {self.reasoning}\n"
        if self.content:
            response += f"Content: {self.content}\n"
        if self.tool_calls:
            response += f"""Tool Calls: {
                ", ".join([f"{tc.name}: {tc.arguments}" for tc in self.tool_calls])
            }"""
        if self.raw:
            response += f"Raw: {self.raw}"
        return response


class TraceStep(BaseModel):
    """Canonical data for a single span (shared with telemetry)."""

    # HUD identifiers
    task_run_id: str | None = Field(default=None)
    job_id: str | None = Field(default=None)

    # Span category - can be any string, but "mcp" and "agent" are privileged on the platform
    category: Literal["mcp", "agent"] | str = Field(default="mcp")  # noqa: PYI051

    # Generic I/O fields - works for any category
    request: Any | None = None
    result: Any | None = None

    # Generic span info
    type: str = Field(default="CLIENT")

    # Timestamps (optional, for local tracking)
    start_timestamp: str | None = None
    end_timestamp: str | None = None

    model_config = ConfigDict(populate_by_name=True, extra="allow")


class HudSpan(BaseModel):
    """A telemetry span ready for export to HUD API."""

    name: str
    trace_id: str = Field(pattern=r"^[0-9a-fA-F]{32}$")
    span_id: str = Field(pattern=r"^[0-9a-fA-F]{16}$")
    parent_span_id: str | None = Field(default=None, pattern=r"^[0-9a-fA-F]{16}$")

    start_time: str  # ISO format
    end_time: str  # ISO format

    status_code: str  # "UNSET", "OK", "ERROR"
    status_message: str | None = None

    attributes: TraceStep
    exceptions: list[dict[str, Any]] | None = None
    internal_type: str | None = None

    model_config = ConfigDict(extra="forbid")


class Trace(BaseModel):
    """Unified result from agent execution (task or prompt).

    Fields:
    - done: Whether the run is complete
    - reward: The reward for the run
    - info: Additional metadata for the run
    - content: The final content/response from the agent
    - isError: Whether the execution resulted in an error
    - trace: The steps taken in the run (empty if not tracing)
    """

    reward: float = Field(default=0.0)
    done: bool = Field(default=True)
    info: dict[str, Any] = Field(default_factory=dict)
    content: str | None = Field(default=None)
    isError: bool = Field(default=False)

    # Metadata
    task: LegacyTask | None = Field(default=None)

    # Trace
    trace: list[TraceStep] = Field(default_factory=list)
    messages: list[Any] = Field(default_factory=list)

    def __len__(self) -> int:
        return len(self.trace)

    @property
    def num_messages(self) -> int:
        return len(self.messages)

    def append(self, step: TraceStep) -> None:
        self.trace.append(step)


# Re-export Task for backwards compatibility (after module defs to avoid circular import)
from hud.eval.task import Task  # noqa: E402

# Type alias for functions that accept v5 Task, v4 LegacyTask, or raw dicts
TaskInput = Task | LegacyTask | dict[str, Any]

__all__ = [
    "AgentResponse",
    "AgentType",
    "HudSpan",
    "LegacyTask",
    "MCPToolCall",
    "MCPToolResult",
    "Task",
    "TaskInput",
    "Trace",
    "TraceStep",
]
