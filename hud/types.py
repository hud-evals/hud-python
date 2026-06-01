from __future__ import annotations

import json
import uuid
from enum import Enum
from typing import TYPE_CHECKING, Any, Literal, TypeAlias

import mcp.types as types
from mcp.types import CallToolRequestParams, CallToolResult
from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from hud.agents.claude import ClaudeAgent
    from hud.agents.gemini import GeminiAgent
    from hud.agents.openai import OpenAIAgent
    from hud.agents.openai_compatible import OpenAIChatAgent
    from hud.agents.types import ClaudeConfig, GeminiConfig, OpenAIChatConfig, OpenAIConfig

    AgentClass: TypeAlias = type[ClaudeAgent | GeminiAgent | OpenAIAgent | OpenAIChatAgent]
    AgentConfigClass: TypeAlias = type[
        ClaudeConfig | GeminiConfig | OpenAIConfig | OpenAIChatConfig
    ]

# JSON-compatible scalar/container values.
JsonValue: TypeAlias = str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]
JsonObject: TypeAlias = dict[str, JsonValue]


class AgentType(str, Enum):
    CLAUDE = "claude"
    OPENAI = "openai"
    GEMINI = "gemini"
    OPENAI_COMPATIBLE = "openai_compatible"

    @property
    def cls(self) -> AgentClass:
        match self:
            case AgentType.CLAUDE:
                from hud.agents.claude import ClaudeAgent

                return ClaudeAgent
            case AgentType.OPENAI:
                from hud.agents.openai import OpenAIAgent

                return OpenAIAgent
            case AgentType.GEMINI:
                from hud.agents.gemini import GeminiAgent

                return GeminiAgent
            case AgentType.OPENAI_COMPATIBLE:
                from hud.agents.openai_compatible import OpenAIChatAgent

                return OpenAIChatAgent

    @property
    def config_cls(self) -> AgentConfigClass:
        """Get config class without importing agent (avoids SDK dependency)."""
        from hud.agents.types import ClaudeConfig, GeminiConfig, OpenAIChatConfig, OpenAIConfig

        match self:
            case AgentType.CLAUDE:
                return ClaudeConfig
            case AgentType.OPENAI:
                return OpenAIConfig
            case AgentType.GEMINI:
                return GeminiConfig
            case AgentType.OPENAI_COMPATIBLE:
                return OpenAIChatConfig

    @property
    def gateway_provider(self) -> str:
        """Default provider client used when this agent type is a gateway shortcut."""
        match self:
            case AgentType.CLAUDE:
                return "anthropic"
            case AgentType.OPENAI:
                return "openai"
            case AgentType.GEMINI:
                return "gemini"
            case AgentType.OPENAI_COMPATIBLE:
                return "openai"


class MCPToolCall(CallToolRequestParams):
    """A tool call."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))  # Unique identifier for reference
    annotation: str | None = None  # Optional explanation of why this action is taken
    provider_name: str | None = None  # Original provider tool name when it differs from MCP name

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

        s = f"→ {self.name}({args_str})"
        if self.annotation:
            s += f"  # {self.annotation}"
        return s

    def __rich__(self) -> str:
        """Rich representation with color formatting."""
        from rich.markup import escape

        from hud.utils.hud_console import hud_console

        s = hud_console.format_tool_call(self.name, self.arguments)
        if self.annotation:
            s += f"  [bright_black]# {escape(self.annotation)}[/bright_black]"
        return s


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
    """Result of a single agent inference call.

    Returned by provider agents' ``get_response()`` methods.  Carries the
    model's text output, any tool calls it wants to make, and provider-
    specific metadata like reasoning traces and citations.
    """

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

    # --- RESPONSE METADATA ---
    # Populated by provider agents when citations are available.
    # Uses dict form of Citation (provider-normalized) so AgentResponse
    # doesn't depend on hud.tools.types at import time.
    citations: list[dict[str, Any]] = Field(default_factory=list)

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
    """The recorded outcome of one task rollout — a pure, serializable datum.

    A ``Trace`` is what a rollout *produces*: the prompt the env handed out, the
    agent's trajectory (``messages``), its final ``content``, and the env-assigned
    ``reward``. It is the unit of training data — held by the thousands, dumped
    for telemetry, collected by ``asyncio.gather``. The live connection and the
    run lifecycle live on ``Rollout`` (hud.client), not here.

    Fields:
    - prompt: The task prompt produced by ``tasks.start``
    - reward: The reward assigned by the env's ``tasks.evaluate``
    - info: Additional metadata for the run
    - content: The final content/response from the agent
    - isError: Whether the execution resulted in an error
    - citations: Provider-normalized citations from the final inference
    - trace: The steps taken in the run (empty if not tracing)
    """

    prompt: str | None = Field(default=None)
    reward: float = Field(default=0.0)
    done: bool = Field(default=True)
    info: dict[str, Any] = Field(default_factory=dict)
    content: str | None = Field(default=None)
    isError: bool = Field(default=False)

    # Response metadata carried from the final AgentResponse
    citations: list[dict[str, Any]] = Field(default_factory=list)

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


# Re-export Task for backwards compatibility (after module defs to avoid circular import)
from hud.eval.task import Task  # noqa: E402

# Resolve Trace.task's forward reference now that Task is available.
Trace.model_rebuild()

# Type alias for functions that accept Task objects or raw task dicts.
TaskInput = Task | dict[str, Any]

__all__ = [
    "AgentResponse",
    "AgentType",
    "HudSpan",
    "JsonObject",
    "JsonValue",
    "MCPToolCall",
    "MCPToolResult",
    "Task",
    "TaskInput",
    "Trace",
    "TraceStep",
]
