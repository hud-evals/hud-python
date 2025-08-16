"""History length evaluator for remote browser environment."""

import logging
from typing import Optional
from fastmcp import Context
from hud.tools.types import EvaluationResult
from . import evaluate

logger = logging.getLogger(__name__)


@evaluate.tool("history_length")
async def history_length(
    ctx: Context, min_length: Optional[int] = None, max_length: Optional[int] = None
):
    """Check if action history has specific length.

    Args:
        min_length: Minimum required length
        max_length: Maximum allowed length

    Returns:
        Evaluation result
    """
    logger.info(f"Evaluating history length - min: {min_length}, max: {max_length}")

    # Get the playwright tool from the environment
    playwright_tool = evaluate.env
    if not playwright_tool:
        logger.error("No playwright tool available")
        return EvaluationResult(
            reward=0.0,
            done=False,
            content="No playwright tool available",
            info={"error": "No playwright tool available"},
        )

    # Get action history from PlaywrightToolWithMemory
    history_length = (
        len(playwright_tool.action_history) if hasattr(playwright_tool, "action_history") else 0
    )
    logger.info(f"Current history length: {history_length}")

    in_range = True
    if min_length is not None and history_length < min_length:
        in_range = False
        logger.info(f"❌ History too short: {history_length} < {min_length}")
    if max_length is not None and history_length > max_length:
        in_range = False
        logger.info(f"❌ History too long: {history_length} > {max_length}")

    if in_range:
        logger.info(f"✅ History length in range: {history_length}")

    # Calculate reward based on how close we are to the target
    if min_length is not None and max_length is not None:
        target = (min_length + max_length) / 2
        reward = max(0, 1 - abs(history_length - target) / target)
    else:
        reward = 1.0 if in_range else 0.0

    # Build content message
    if in_range:
        content = f"History length in range: {history_length}"
    else:
        if min_length is not None and history_length < min_length:
            content = f"History too short: {history_length} < {min_length}"
        else:
            content = f"History too long: {history_length} > {max_length}"

    return EvaluationResult(
        reward=float(reward),
        done=in_range,
        content=content,
        info={
            "history_length": history_length,
            "min_length": min_length,
            "max_length": max_length,
            "in_range": in_range,
        },
    )
