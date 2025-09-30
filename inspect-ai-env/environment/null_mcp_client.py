"""
Null MCP Client for Inspect AI integration.

This is a minimal implementation of the AgentMCPClient protocol that does nothing.
It's used when the HUD agent is running inside Inspect AI, where Inspect AI itself
manages the tool execution loop, and we only need the agent for generate() calls.
"""

from typing import Any
import mcp.types as types
from hud.types import MCPToolCall, MCPToolResult


class NullMCPClient:
    """
    A null implementation of AgentMCPClient that satisfies the protocol
    but doesn't actually connect to any MCP servers.

    This is used in Inspect AI contexts where tools are managed by Inspect AI,
    not through MCP.
    """

    def __init__(self):
        self._initialized = False
        self._mcp_config = {}

    @property
    def mcp_config(self) -> dict[str, dict[str, Any]]:
        """Get the MCP config (empty for null client)."""
        return self._mcp_config

    @property
    def is_connected(self) -> bool:
        """Check if client is connected (always False for null client)."""
        return self._initialized

    async def initialize(self, mcp_config: dict[str, dict[str, Any]] | None = None) -> None:
        """Initialize the client (no-op for null client)."""
        if mcp_config:
            self._mcp_config = mcp_config
        self._initialized = True

    async def list_tools(self) -> list[types.Tool]:
        """List all available tools (empty for null client)."""
        return []

    async def call_tool(self, tool_call: MCPToolCall) -> MCPToolResult:
        """Execute a tool (raises error for null client)."""
        raise NotImplementedError(
            "NullMCPClient cannot execute tools. Tools should be executed by Inspect AI."
        )

    async def shutdown(self) -> None:
        """Shutdown the client (no-op for null client)."""
        self._initialized = False
