from __future__ import annotations

import contextlib
import json
import logging
import uuid
from collections import defaultdict
from enum import Enum
from string import Template
from typing import Any, Literal

import mcp.types as types
from mcp.types import CallToolRequestParams, CallToolResult
from pydantic import BaseModel, ConfigDict, Field, field_validator

from hud.settings import settings
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
        from hud.agents import ClaudeAgent, GeminiAgent, OpenAIAgent, OperatorAgent
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
    """Standard agent configuration that tasks can override.
    Provider-specific configs should not be included here.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    allowed_tools: list[str] | None = None
    disallowed_tools: list[str] | None = None
    response_tool_name: str | None = None
    system_prompt: str | None = None
    append_setup_output: bool = True
    initial_screenshot: bool = True


class Task(BaseModel):
    """
    A task configuration that can be used to create a task.

    The mcp_config field supports environment variable substitution using
    template placeholders in the format ${VAR_NAME} or ${VAR_NAME:default_value}.

    Example:
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
        Automatically resolve environment variables in mcp_config using Template.

        Supports ${VAR_NAME} syntax with variable substitution from
        System environment variables (including HUD_API_KEY, etc.)

        Missing variables resolve to empty strings.
        """
        import os

        # Start with current environment variables
        mapping = dict(os.environ)
        # Include settings (from process env, project .env, and user .env)
        settings_dict = settings.model_dump()
        mapping.update(settings_dict)
        # Add UPPERCASE aliases for settings keys
        for _key, _val in settings_dict.items():
            with contextlib.suppress(Exception):
                mapping[_key.upper()] = _val

        if settings.api_key:
            mapping["HUD_API_KEY"] = settings.api_key
        else:
            global _missing_api_key_error_logged
            if not _missing_api_key_error_logged:
                logger.error("HUD_API_KEY is not set, tracing and remote training will not work")
                _missing_api_key_error_logged = True

        def substitute_in_value(obj: Any) -> Any:
            """Recursively substitute variables in nested structures."""
            if isinstance(obj, str):
                # Use Template's substitute with defaultdict - missing vars become empty strings
                safe_mapping = defaultdict(str, mapping)
                return Template(obj).substitute(safe_mapping)
            elif isinstance(obj, dict):
                return {k: substitute_in_value(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [substitute_in_value(item) for item in obj]
            else:
                return obj

        return substitute_in_value(v)


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
    """A tool result."""

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
    task: Task | None = Field(default=None)

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

    def populate_from_context(self) -> None:
        """Populate trace steps from the current trace context if available.

        This checks if we're executing within a hud.trace() context and
        automatically populates the trace field with collected steps.
        """
        from hud.otel.context import get_current_task_run_id
        from hud.telemetry.replay import get_trace

        task_run_id = get_current_task_run_id()
        if task_run_id:
            collected_trace = get_trace(task_run_id)
            if collected_trace:
                self.trace = collected_trace.trace


__all__ = [
    "AgentResponse",
    "AgentType",
    "MCPToolCall",
    "MCPToolResult",
    "Trace",
    "TraceStep",
]
