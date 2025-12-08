"""Cookie match evaluator for remote browser environment."""

import logging
from fastmcp import Context
from hud.tools.types import EvaluationResult
from . import evaluate

logger = logging.getLogger(__name__)


@evaluate.tool("cookie_match")
async def cookie_match(ctx: Context, cookie_name: str, expected_value: str):
    """Check if a cookie value matches expected value.

    Args:
        cookie_name: Name of the cookie to check
        expected_value: Expected value of the cookie

    Returns:
        Evaluation result
    """
    logger.info(f"Checking cookie {cookie_name} for value: {expected_value}")

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

        # Find the cookie
        cookie = next((c for c in cookies if c.get("name") == cookie_name), None)

        if not cookie:
            logger.info(f"❌ Cookie not found: {cookie_name}")
            return EvaluationResult(
                reward=0.0,
                done=False,
                content=f"Cookie not found: {cookie_name}",
                info={"success": False, "cookie_name": cookie_name},
            )

        actual_value = cookie.get("value", "")
        if actual_value == expected_value:
            logger.info(f"✅ Cookie value matches: {cookie_name}={expected_value}")
            return EvaluationResult(
                reward=1.0,
                done=True,
                content=f"Cookie value matches",
                info={"success": True, "cookie_name": cookie_name, "value": actual_value},
            )
        else:
            logger.info(
                f"❌ Cookie value mismatch: expected '{expected_value}', got '{actual_value}'"
            )
            return EvaluationResult(
                reward=0.0,
                done=False,
                content=f"Cookie value mismatch",
                info={
                    "success": False,
                    "cookie_name": cookie_name,
                    "expected": expected_value,
                    "actual": actual_value,
                },
            )

    except Exception as e:
        logger.error(f"Error checking cookie: {e}")
        return EvaluationResult(
            reward=0.0,
            done=False,
            content=f"Error checking cookie: {str(e)}",
            info={"success": False, "error": str(e)},
        )
