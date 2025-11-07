from hud.server import MCPRouter
from .eval_all import router as eval_all_router

router = MCPRouter(name="evaluate")
router.include_router(eval_all_router)

__all__ = ["router"]
