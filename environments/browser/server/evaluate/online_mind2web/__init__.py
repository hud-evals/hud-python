"""Evaluation tools for browser environment."""

from hud.server import MCPRouter

router = MCPRouter()

from .autonomous_eval import router as autonomous_eval_router
from .webjudge import router as webjudge_router

router.include_router(autonomous_eval_router)
router.include_router(webjudge_router)


__all__ = ["router"]
