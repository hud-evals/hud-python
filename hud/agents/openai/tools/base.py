"""Common agent-side OpenAI tool support."""

from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING, Any

from mcp.types import TextContent

from hud.agents import tools as _agent_tools
from hud.agents.tools import AgentTool, AgentToolSpec, CallTool

if TYPE_CHECKING:
    from openai.types.responses import ToolParam

    from hud.types import MCPToolCall, MCPToolResult
else:
    ToolParam = Any

OpenAIToolSpec = AgentToolSpec
call_tool = _agent_tools.call_tool


class OpenAITool(AgentTool["ToolParam"], ABC):
    """Agent-side OpenAI provider tool backed by an environment tool."""

    def format_result(self, call: MCPToolCall, result: MCPToolResult) -> dict[str, Any]:
        """Format a generic provider tool result for the OpenAI Responses API."""
        return {
            "type": "function_call_output",
            "call_id": call.id,
            "output": result_text(result),
        }


def result_text(result: MCPToolResult) -> str:
    """Return text content from an MCP tool result."""
    parts = [block.text for block in result.content if isinstance(block, TextContent)]
    return "\n".join(part for part in parts if part)


__all__ = ["CallTool", "OpenAITool", "OpenAIToolSpec", "call_tool", "result_text"]
