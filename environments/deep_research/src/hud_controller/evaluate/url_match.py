"""URL match evaluator for remote browser environment."""

import logging
from fastmcp import Context
from hud.tools.types import EvaluationResult
from . import evaluate

logger = logging.getLogger(__name__)


@evaluate.tool("url_match")
async def url_match(ctx: Context, target_url: str):
    """Check if the current URL contains a target string.

    Args:
        target_url: The target URL string to look for

    Returns:
        Evaluation result with reward between 0.0 and 1.0
    """
    logger.info(f"Evaluating URL match for target: '{target_url}'")

    # Get the playwright tool from the environment
    playwright_tool = evaluate.env
    if not playwright_tool or not hasattr(playwright_tool, "page") or not playwright_tool.page:
        logger.error("No browser page available")
        return EvaluationResult(
            reward=0.0, done=False, content="No browser page available", info={"success": False}
        )

    # Get the current URL
    current_url = playwright_tool.page.url
    logger.info(f"Current page URL: '{current_url}'")

    # Check if target URL is in current URL
    if target_url in current_url:
        logger.info(f"✅ URL match successful: '{target_url}' found in '{current_url}'")
        return EvaluationResult(
            reward=1.0,
            done=True,
            content=f"URL match successful",
            info={"success": True, "current_url": current_url, "target_url": target_url},
        )
    else:
        logger.info(f"❌ URL match failed: '{target_url}' not found in '{current_url}'")

        # Provide helpful debugging information
        info = {"success": False, "current_url": current_url, "target_url": target_url}

        # Check for partial matches
        if target_url.lower() in current_url.lower():
            info["note"] = "Case-insensitive match found"

        # Check for protocol differences
        if current_url.startswith("https://") and not target_url.startswith("https://"):
            alt_target = "https://" + target_url
            if alt_target in current_url:
                info["note"] = "Match found with https:// prefix"

        return EvaluationResult(reward=0.0, done=False, content=f"URL match failed", info=info)
