"""HUD MCP client implementations."""

from __future__ import annotations

from .base import AgentMCPClient, BaseHUDClient
from .environment import EnvironmentClient
from .fastmcp import FastMCPHUDClient

# Default to FastMCP client (no optional dependencies)
MCPClient = FastMCPHUDClient

# Note: MCPUseHUDClient requires mcp-use (optional dependency in [agents]).
# Import directly if needed:
#   from hud.clients.mcp_use import MCPUseHUDClient

__all__ = [
    "AgentMCPClient",
    "BaseHUDClient",
    "EnvironmentClient",
    "FastMCPHUDClient",
    "MCPClient",
]
