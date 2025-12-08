"""Setup function to load custom HTML content."""

import logging
from fastmcp import Context
from hud.tools.types import SetupResult
from . import setup

logger = logging.getLogger(__name__)


@setup.tool("load_html_content")
async def load_html_content(ctx: Context, html: str):
    """Load custom HTML content directly into the browser.

    Args:
        html: HTML content to load

    Returns:
        Setup result with status
    """
    logger.info("Loading custom HTML content into browser")

    # Get the playwright tool from the environment
    playwright_tool = setup.env
    if not playwright_tool or not hasattr(playwright_tool, "page") or not playwright_tool.page:
        logger.error("No browser page available")
        return SetupResult(content="No browser page available", isError=True)

    try:
        # Create a data URL with the HTML content
        data_url = f"data:text/html,{html}"

        # Navigate to the data URL
        await playwright_tool.page.goto(data_url)
        logger.info("Successfully loaded custom HTML content")

        return SetupResult(
            content="Custom HTML content loaded",
            info={"html_length": len(html), "current_url": playwright_tool.page.url},
        )
    except Exception as e:
        logger.error(f"Failed to load HTML content: {e}")
        return SetupResult(content=f"Failed to load HTML content: {str(e)}", isError=True)
