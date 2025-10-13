"""Interaction setup functions for remote browser environment."""

import logging
from fastmcp import Context
from mcp.types import TextContent
from . import setup

logger = logging.getLogger(__name__)


@setup.tool("click_element")
async def click_element(ctx: Context, selector: str, timeout: int = 30000):
    """Click on an element by selector.

    Args:
        selector: CSS selector for the element
        timeout: Maximum time to wait for element (ms)

    Returns:
        Setup result with status
    """
    logger.info(f"Clicking element with selector: {selector}")

    # Get the playwright tool from the environment context
    persistent_ctx = setup.env
    playwright_tool = getattr(persistent_ctx, "playwright_tool", None)
    if not playwright_tool or not hasattr(playwright_tool, "page") or not playwright_tool.page:
        logger.error("No browser page available")
        return TextContent(text="No browser page available", type="text")

    try:
        # Wait for element and click
        element = await playwright_tool.page.wait_for_selector(selector, timeout=timeout)
        await element.click()

        logger.info(f"Successfully clicked element: {selector}")
        return TextContent(text=f"Clicked element: {selector}", type="text")
    except Exception as e:
        logger.error(f"Failed to click element: {e}")
        return TextContent(text=f"Failed to click element: {str(e)}", type="text")


@setup.tool("fill_input")
async def fill_input(ctx: Context, selector: str, text: str, timeout: int = 30000):
    """Fill an input field with text.

    Args:
        selector: CSS selector for the input element
        text: Text to fill in the input
        timeout: Maximum time to wait for element (ms)

    Returns:
        Setup result with status
    """
    logger.info(f"Filling input {selector} with text")

    # Get the playwright tool from the environment context
    persistent_ctx = setup.env
    playwright_tool = getattr(persistent_ctx, "playwright_tool", None)
    if not playwright_tool or not hasattr(playwright_tool, "page") or not playwright_tool.page:
        logger.error("No browser page available")
        return TextContent(text="No browser page available", type="text")

    try:
        # Wait for element and fill
        element = await playwright_tool.page.wait_for_selector(selector, timeout=timeout)
        await element.fill(text)

        logger.info(f"Successfully filled input: {selector}")
        return TextContent(text=f"Filled input {selector} with {len(text)} characters", type="text")
    except Exception as e:
        logger.error(f"Failed to fill input: {e}")
        return TextContent(text=f"Failed to fill input: {str(e)}", type="text")


@setup.tool("select_option")
async def select_option(ctx: Context, selector: str, value: str, timeout: int = 30000):
    """Select an option in a dropdown.

    Args:
        selector: CSS selector for the select element
        value: Value of the option to select
        timeout: Maximum time to wait for element (ms)

    Returns:
        Setup result with status
    """
    logger.info(f"Selecting option {value} in {selector}")

    # Get the playwright tool from the environment context
    persistent_ctx = setup.env
    playwright_tool = getattr(persistent_ctx, "playwright_tool", None)
    if not playwright_tool or not hasattr(playwright_tool, "page") or not playwright_tool.page:
        logger.error("No browser page available")
        return TextContent(text="No browser page available", type="text")

    try:
        # Wait for element and select option
        await playwright_tool.page.select_option(selector, value, timeout=timeout)

        logger.info(f"Successfully selected option: {value}")
        return TextContent(text=f"Selected option '{value}' in {selector}", type="text")
    except Exception as e:
        logger.error(f"Failed to select option: {e}")
        return TextContent(text=f"Failed to select option: {str(e)}", type="text")
