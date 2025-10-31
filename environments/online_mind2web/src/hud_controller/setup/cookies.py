"""Cookie setup functions for remote browser environment."""

import logging
from typing import List, Dict, Any
from fastmcp import Context
from mcp.types import TextContent
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

    # Get the playwright tool from the environment context
    persistent_ctx = setup.env
    playwright_tool = getattr(persistent_ctx, "playwright_tool", None)
    if not playwright_tool or not hasattr(playwright_tool, "page") or not playwright_tool.page:
        logger.error("No browser page available")
        return TextContent(text="No browser page available", type="text")

    try:
        # Add cookies to the context
        await playwright_tool.page.context.add_cookies(cookies)

        logger.info(f"Successfully set {len(cookies)} cookies")
        return TextContent(
            text=f"Set {len(cookies)} cookies: {', '.join([c.get('name', 'unnamed') for c in cookies])}",
            type="text",
        )
    except Exception as e:
        logger.error(f"Failed to set cookies: {e}")
        return TextContent(text=f"Failed to set cookies: {str(e)}", type="text")


@setup.tool("clear_cookies")
async def clear_cookies(ctx: Context):
    """Clear all cookies from the browser.

    Returns:
        Setup result with status
    """
    logger.info("Clearing all cookies")

    # Get the playwright tool from the environment context
    persistent_ctx = setup.env
    playwright_tool = getattr(persistent_ctx, "playwright_tool", None)
    if not playwright_tool or not hasattr(playwright_tool, "page") or not playwright_tool.page:
        logger.error("No browser page available")
        return TextContent(text="No browser page available", type="text")

    try:
        # Clear all cookies
        await playwright_tool.page.context.clear_cookies()

        logger.info("Successfully cleared all cookies")
        return TextContent(text="Cleared all cookies", type="text")
    except Exception as e:
        logger.error(f"Failed to clear cookies: {e}")
        return TextContent(text=f"Failed to clear cookies: {str(e)}", type="text")
