"""Navigation setup for remote browser environment."""

import logging
from fastmcp import Context
from hud.tools.types import SetupResult
from . import setup

logger = logging.getLogger(__name__)


@setup.tool("navigate_to_url")
async def navigate_to_url(ctx: Context, url: str, wait_for_load_state: str = "networkidle"):
    """Navigate browser to a specific URL.

    Args:
        url: The URL to navigate to
        wait_for_load_state: State to wait for after navigation

    Returns:
        Setup result with navigation status
    """
    logger.info(f"Navigating to URL: {url}")

    # Get the playwright tool from the environment
    playwright_tool = setup.env
    if not playwright_tool or not hasattr(playwright_tool, "page") or not playwright_tool.page:
        logger.error("No playwright tool available")
        return SetupResult(content="No browser available for navigation", isError=True)

    # Navigate using the playwright tool
    result = await playwright_tool.navigate(url, wait_for_load_state)

    if result.get("success"):
        logger.info(f"Successfully navigated to {url}")
        return SetupResult(
            content=f"Navigated to {url}",
            info={"url": result.get("url"), "title": result.get("title")},
        )
    else:
        logger.error(f"Failed to navigate: {result.get('error')}")
        return SetupResult(content=f"Navigation failed: {result.get('error')}", isError=True)
