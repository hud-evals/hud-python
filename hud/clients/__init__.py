"""HUD MCP client implementations."""

from __future__ import annotations

from .base import AgentMCPClient, BaseHUDClient
from .environment import EnvironmentClient
from .fastmcp import FastMCPHUDClient
from .mcp_use import MCPUseHUDClient

# Default to MCP-use for agents (has multi-server session support)
MCPClient = MCPUseHUDClient

__all__ = [
    "AgentMCPClient",
    "BaseHUDClient",
    "EnvironmentClient",
    "FastMCPHUDClient",
    "MCPClient",
    "MCPUseHUDClient",
]
