"""Setup tools for the sheet environment."""

from hud.server import MCPRouter
from .load_spreadsheet import router as load_spreadsheet_router

router = MCPRouter(name="setup")
router.include_router(load_spreadsheet_router)

__all__ = ["router"]
