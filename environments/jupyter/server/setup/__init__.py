"""Setup tools for the sheet environment."""

from hud.server import MCPRouter

# Not setup tool here
router = MCPRouter(name="setup")

__all__ = ["router"]
