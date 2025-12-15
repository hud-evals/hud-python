"""HUD MCP client implementations."""

from __future__ import annotations

from .base import AgentMCPClient, BaseHUDClient
from .environment import EnvironmentClient
from .fastmcp import FastMCPHUDClient

# Default to FastMCP client (no optional dependencies)
MCPClient = FastMCPHUDClient

__all__ = [
    "AgentMCPClient",
    "BaseHUDClient",
    "EnvironmentClient",
    "FastMCPHUDClient",
    "MCPClient",
]
