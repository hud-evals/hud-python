"""Element exists evaluator for remote browser environment."""

import logging
from fastmcp import Context
from hud.tools.types import EvaluationResult
from . import evaluate

logger = logging.getLogger(__name__)


@evaluate.tool("element_exists")
async def element_exists(ctx: Context, selector: str):
    """Check if an element exists on the page.

    Args:
        selector: CSS selector for the element

    Returns:
        Evaluation result
    """
    logger.info(f"Checking if element exists: {selector}")

    # Get the playwright tool from the environment
    playwright_tool = evaluate.env
    if not playwright_tool or not hasattr(playwright_tool, "page") or not playwright_tool.page:
        logger.error("No browser page available")
        return EvaluationResult(
            reward=0.0, done=False, content="No browser page available", info={"success": False}
        )

    try:
        # Check if element exists
        element = await playwright_tool.page.query_selector(selector)

        if element:
            logger.info(f"✅ Element found: {selector}")
            return EvaluationResult(
                reward=1.0,
                done=True,
                content=f"Element found: {selector}",
                info={"success": True, "selector": selector},
            )
        else:
            logger.info(f"❌ Element not found: {selector}")
            return EvaluationResult(
                reward=0.0,
                done=False,
                content=f"Element not found: {selector}",
                info={"success": False, "selector": selector},
            )

    except Exception as e:
        logger.error(f"Error checking element: {e}")
        return EvaluationResult(
            reward=0.0,
            done=False,
            content=f"Error checking element: {str(e)}",
            info={"success": False, "error": str(e)},
        )
