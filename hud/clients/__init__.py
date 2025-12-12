"""HUD MCP client implementations."""

from __future__ import annotations

from .base import AgentMCPClient, BaseHUDClient
from .fastmcp import FastMCPHUDClient
from .mcp_use import MCPUseHUDClient

# Default to MCP-use for agents (has multi-server session support)
MCPClient = MCPUseHUDClient

__all__ = [
    "AgentMCPClient",
    "BaseHUDClient",
    "FastMCPHUDClient",
    "MCPClient",
    "MCPUseHUDClient",
]
