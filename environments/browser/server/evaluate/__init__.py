"""Evaluation tools for browser environment."""

from hud.server import MCPRouter

# Create combined router for all evaluation tools
router = MCPRouter(name="evaluate")

# Import and include sub-routers
from .game_2048 import router as game_2048_router
from .todo import router as todo_router

router.include_router(game_2048_router)
router.include_router(todo_router)

__all__ = ["router"]
