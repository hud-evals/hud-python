"""Cookie setup functions for remote browser environment."""

import logging
from typing import List, Dict, Any
from fastmcp import Context
from hud.tools.types import SetupResult
from . import setup

logger = logging.getLogger(__name__)


@setup.tool("set_cookies")
async def set_cookies(ctx: Context, cookies: List[Dict[str, Any]]):
    """Set cookies in the browser.

    Args:
        cookies: List of cookie dictionaries with name, value, and optional properties

    Returns:
        Setup result with status
    """
    logger.info(f"Setting {len(cookies)} cookies")

    # Get the playwright tool from the environment
    playwright_tool = setup.env
    if not playwright_tool or not hasattr(playwright_tool, "page") or not playwright_tool.page:
        logger.error("No browser page available")
        return SetupResult(content="No browser page available", isError=True)

    try:
        # Add cookies to the context
        await playwright_tool.page.context.add_cookies(cookies)

        logger.info(f"Successfully set {len(cookies)} cookies")
        return SetupResult(
            content=f"Set {len(cookies)} cookies",
            info={"cookies_set": [c.get("name") for c in cookies]},
        )
    except Exception as e:
        logger.error(f"Failed to set cookies: {e}")
        return SetupResult(content=f"Failed to set cookies: {str(e)}", isError=True)


@setup.tool("clear_cookies")
async def clear_cookies(ctx: Context):
    """Clear all cookies from the browser.

    Returns:
        Setup result with status
    """
    logger.info("Clearing all cookies")

    # Get the playwright tool from the environment
    playwright_tool = setup.env
    if not playwright_tool or not hasattr(playwright_tool, "page") or not playwright_tool.page:
        logger.error("No browser page available")
        return SetupResult(content="No browser page available", isError=True)

    try:
        # Clear all cookies
        await playwright_tool.page.context.clear_cookies()

        logger.info("Successfully cleared all cookies")
        return SetupResult(content="Cleared all cookies")
    except Exception as e:
        logger.error(f"Failed to clear cookies: {e}")
        return SetupResult(content=f"Failed to clear cookies: {str(e)}", isError=True)
