import os, json, logging
from hud.tools.types import EvaluationResult
from . import evaluate
from fastmcp import Context

logger = logging.getLogger(__name__)


@evaluate.tool("dumb")
async def dumb(ctx: Context):
    logging.info("Dumb evaluate function")
    return EvaluationResult()
