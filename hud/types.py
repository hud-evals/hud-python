from __future__ import annotations

import json
import uuid
from enum import Enum
from typing import TYPE_CHECKING, Any, Literal, TypeAlias

import mcp.types as types
from mcp.types import CallToolRequestParams, CallToolResult
from pydantic import BaseModel, ConfigDict, Field, field_serializer

from hud.utils.serialization import json_safe_value

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
                from hud.agents import ClaudeAgent

                return ClaudeAgent
            case AgentType.OPENAI:
                from hud.agents import OpenAIAgent

                return OpenAIAgent
            case AgentType.GEMINI:
                from hud.agents import GeminiAgent

                return GeminiAgent
            case AgentType.OPENAI_COMPATIBLE:
                from hud.agents import OpenAIChatAgent

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


class Sample(BaseModel):
    """One model generation in a rollout: tokens conditioned on + tokens produced.

    Token-level data for RL training (Tinker-shaped). ``output_logprobs`` are the
    per-output-token logprobs under the *sampling* policy (q). Populated only when
    the model backend is trainable (returns token ids + logprobs); closed/eval-only
    backends leave it empty.
    """

    prompt_token_ids: list[int] = Field(default_factory=list)
    output_token_ids: list[int] = Field(default_factory=list)
    output_logprobs: list[float] = Field(default_factory=list)


class AgentResponse(BaseModel):
    """Result of a single LLM inference call.

    Returned by provider agents' ``get_response()`` methods.  Carries the
    model's text output, any tool calls it wants to make, and provider-
    specific metadata like reasoning traces and citations.
    """

    model_config = ConfigDict(populate_by_name=True)

    # --- FUNCTIONAL ---
    tool_calls: list[MCPToolCall] = Field(default_factory=list)
    done: bool = Field(default=False)

    # --- TRAINING ---
    # Token-level data for THIS turn; present iff the model backend is trainable.
    sample: Sample | None = Field(default=None)

    # --- RESPONSE ---
    content: str | None = Field(default=None)
    reasoning: str | None = Field(default=None)
    finish_reason: str | None = Field(default=None)
    citations: list[dict[str, Any]] = Field(default_factory=list)
    refusal: str | None = Field(default=None)
    isError: bool = Field(default=False)
    raw: Any | None = Field(default=None)

    # Timestamps
    start_timestamp: str | None = None
    end_timestamp: str | None = None

    @field_serializer("raw", when_used="json")
    def _serialize_raw(self, raw: Any | None) -> Any:
        return json_safe_value(raw)

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
    """The agent's trajectory for one rollout — a pure, serializable datum.

    Everything the *agent* collects while running: ``messages``, token-level
    ``samples``, final ``content`` (the graded answer), ``citations``, and whether it
    errored. The unit of training data. The task lifecycle (prompt, reward, evaluation)
    and the live connection live on ``Run``, not here.
    """

    done: bool = Field(default=True)
    info: dict[str, Any] = Field(default_factory=dict)
    content: str | None = Field(default=None)
    isError: bool = Field(default=False)

    # Response metadata carried from the final AgentResponse
    citations: list[dict[str, Any]] = Field(default_factory=list)

    # Trace
    trace: list[TraceStep] = Field(default_factory=list)
    messages: list[Any] = Field(default_factory=list)

    # Token-level samples for RL training — one per model call; empty for
    # eval-only runs. Inline mode (Mode A) fills these; server-side mode (Mode B)
    # leaves them empty and keys the trajectory by ``trace_id`` instead.
    # Inline token-level samples (Mode A); empty for eval-only runs.
    samples: list[Sample] = Field(default_factory=list)
    # Keys server-side-collected logprobs (Mode B); None for eval-only runs.
    trace_id: str | None = Field(default=None)

    def __len__(self) -> int:
        return len(self.trace)

    @property
    def num_messages(self) -> int:
        return len(self.messages)

    def append(self, step: TraceStep) -> None:
        self.trace.append(step)


__all__ = [
    "AgentResponse",
    "AgentType",
    "JsonObject",
    "JsonValue",
    "MCPToolCall",
    "MCPToolResult",
    "Trace",
    "TraceStep",
]
