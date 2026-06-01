from __future__ import annotations

from hud._runtime import activate_runtime

from .router import MCPRouter
from .server import MCPServer

# The MCP server runtime needs HUD's compatibility patches.
activate_runtime()

__all__ = ["MCPRouter", "MCPServer"]
