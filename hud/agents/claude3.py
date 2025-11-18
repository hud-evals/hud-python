"""Claude3 MCP Agent implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import mcp.types as types

from hud.types import AgentResponse, MCPToolCall, MCPToolResult

if TYPE_CHECKING:
    from hud.datasets import Task

from .base import MCPAgent


class Claude3Agent(MCPAgent):
    """
    Claude3 agent that uses MCP servers for tool execution.

    This agent uses Claude's native tool calling capabilities but executes
    tools through MCP servers instead of direct implementation.
    """

    def __init__(self, **kwargs: Any) -> None:
        """Initialize Claude3 MCP agent."""
        super().__init__(**kwargs)
        self.model_name = "Claude3"
        self.checkpoint_name = kwargs.get("model", "claude-sonnet-4-20250514")

    async def get_system_messages(self) -> list[types.ContentBlock]:
        """Get the system prompt."""
        return []

    async def get_response(self, messages: list[Any]) -> AgentResponse:
        """Get response from the model including any tool calls."""
        raise NotImplementedError("Claude3Agent.get_response not yet implemented")

    async def format_blocks(self, blocks: list[types.ContentBlock]) -> list[Any]:
        """Format a list of content blocks into a list of messages."""
        raise NotImplementedError("Claude3Agent.format_blocks not yet implemented")

    async def format_tool_results(
        self, tool_calls: list[MCPToolCall], tool_results: list[MCPToolResult]
    ) -> list[Any]:
        """Format tool results into messages for the model."""
        raise NotImplementedError("Claude3Agent.format_tool_results not yet implemented")

