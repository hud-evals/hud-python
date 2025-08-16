"""Cookie exists evaluator for remote browser environment."""

import logging
from fastmcp import Context
from hud.tools.types import EvaluationResult
from . import evaluate

logger = logging.getLogger(__name__)


@evaluate.tool("cookie_exists")
async def cookie_exists(ctx: Context, cookie_name: str):
    """Check if a cookie exists in the browser.

    Args:
        cookie_name: Name of the cookie to check for

    Returns:
        Evaluation result
    """
    logger.info(f"Checking if cookie exists: {cookie_name}")

    # Get the playwright tool from the environment
    playwright_tool = evaluate.env
    if not playwright_tool or not hasattr(playwright_tool, "page") or not playwright_tool.page:
        logger.error("No browser page available")
        return EvaluationResult(
            reward=0.0, done=False, content="No browser page available", info={"success": False}
        )

    try:
        # Get all cookies
        cookies = await playwright_tool.page.context.cookies()

        # Check if cookie exists
        cookie = next((c for c in cookies if c.get("name") == cookie_name), None)

        if cookie:
            logger.info(f"✅ Cookie found: {cookie_name}")
            return EvaluationResult(
                reward=1.0,
                done=True,
                content=f"Cookie found: {cookie_name}",
                info={
                    "success": True,
                    "cookie_name": cookie_name,
                    "cookie_value": cookie.get("value", ""),
                    "domain": cookie.get("domain", ""),
                },
            )
        else:
            logger.info(f"❌ Cookie not found: {cookie_name}")
            return EvaluationResult(
                reward=0.0,
                done=False,
                content=f"Cookie not found: {cookie_name}",
                info={"success": False, "cookie_name": cookie_name, "total_cookies": len(cookies)},
            )

    except Exception as e:
        logger.error(f"Error checking cookie: {e}")
        return EvaluationResult(
            reward=0.0,
            done=False,
            content=f"Error checking cookie: {str(e)}",
            info={"success": False, "error": str(e)},
        )
