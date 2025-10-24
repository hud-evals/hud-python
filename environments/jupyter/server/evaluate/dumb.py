import os, json, logging
from hud.tools.types import EvaluationResult
from fastmcp import Context
from hud.server import MCPRouter

logger = logging.getLogger(__name__)
router = MCPRouter()


@router.tool("dumb")
async def dumb(ctx: Context):
    logging.info("Dumb evaluate function")
    return EvaluationResult()
