from __future__ import annotations

from typing import Any, Literal

from mcp.types import CallToolRequestParams, CallToolResult
from pydantic import BaseModel, ConfigDict, Field


class MCPToolCall(CallToolRequestParams):
    """A tool call."""

    def __str__(self) -> str:
        response = f"Tool: {self.name}"
        if self.arguments:
            response += f"\nArguments: {self.arguments}"
        return response


class MCPToolResult(CallToolResult):
    """A tool result."""

    def __str__(self) -> str:
        response = f"Content: {self.content}"
        if self.structuredContent:
            response += f"\nStructured Content: {self.structuredContent}"
        if self.isError:
            response += f"\nError: {self.isError}"
        return response


class AgentResponse(BaseModel):
    """A model response in the conversation."""

    # --- FUNCTIONAL ---
    tool_calls: list[MCPToolCall] = Field(default_factory=list)
    done: bool = Field(default=False)

    # --- TELEMETRY [app.hud.so] ---
    # Responses
    content: str | None = Field(default=None)
    reasoning: str | None = Field(default=None)
    info: dict[str, Any] = Field(default_factory=dict)
    isError: bool = Field(default=False)

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
                ', '.join([f'{tc.name}: {tc.arguments}' for tc in self.tool_calls])
            }"""
        return response


class TraceStep(BaseModel):
    """Canonical data for a single agent/MCP span (shared with telemetry)."""

    # HUD identifiers
    task_run_id: str | None = Field(default=None)
    job_id: str | None = Field(default=None)

    # Span category
    category: Literal["mcp", "agent"] = Field(default="mcp")

    # === MCP fields ===
    mcp_request: Any | None = None  # ClientRequest
    mcp_result: Any | None = None  # ServerResult

    # === Agent fields ===
    agent_request: dict[str, Any] | None = None
    agent_response: dict[str, Any] | None = None

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

    done: bool = Field(default=True)
    reward: float = Field(default=0.0)
    info: dict[str, Any] = Field(default_factory=dict)
    content: str | None = Field(default=None)
    isError: bool = Field(default=False)
    trace: list[TraceStep] = Field(default_factory=list)

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
    "MCPToolCall",
    "MCPToolResult",
    "Trace",
    "TraceStep",
]
