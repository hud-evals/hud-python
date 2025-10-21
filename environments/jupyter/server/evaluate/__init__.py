from hud.server import MCPRouter
from .dumb import router as dumb_router
from .eval_single import router as eval_single_router
from .eval_all import router as eval_all_router

router = MCPRouter(name="evaluate")
router.include_router(dumb_router)
router.include_router(eval_single_router)
router.include_router(eval_all_router)

__all__ = ["router"]
