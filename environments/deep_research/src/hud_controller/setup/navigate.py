"""Setup tools for navigation (Wikipedia-focused)."""

import logging
from typing import Optional

from fastmcp import Context

from . import setup

logger = logging.getLogger(__name__)


@setup.tool("navigate")
async def navigate(ctx: Context, url: str):
    """Navigate to a URL."""
    tool = setup.env
    await tool.navigate(url)
    return {"success": True, "message": f"Navigated to {url}"}


@setup.tool("open_wikipedia_page")
async def open_wikipedia_page(ctx: Context, title: str, lang: Optional[str] = "en"):
    """Open a Wikipedia page by article title.

    Args:
        title: Article title (e.g., "Alan Turing")
        lang: Language code (default: en)
    """
    slug = title.strip().replace(" ", "_")
    url = f"https://{lang}.wikipedia.org/wiki/{slug}"
    tool = setup.env
    await tool.navigate(url)
    return {"success": True, "message": f"Opened Wikipedia page: {title}", "url": url}

