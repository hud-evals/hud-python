"""MCP server for deep research (Wikipedia-focused) environment."""

import logging
import os
import sys
from datetime import datetime
from typing import Optional, TypedDict

logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="[%(levelname)s] %(asctime)s | %(name)s | %(message)s",
    force=True,
)
logger = logging.getLogger(__name__)

from hud.server import MCPServer

from .tools.playwright import PlaywrightToolWithMemory
from .tools.executor import BrowserExecutor

from .setup import setup as setup_hub
from .evaluate import evaluate as evaluate_hub


class Telemetry(TypedDict):
    provider: str
    status: str
    live_url: str | None
    timestamp: str
    cdp_url: str | None
    instance_id: str | None


mcp = MCPServer(
    name="HUD Deep Research",
    instructions="""
    This environment launches a local Playwright browser tailored for reading and
    analyzing Wikipedia pages. It exposes setup and evaluation tools plus a Playwright
    automation tool with navigation/action history.
    """,
)

playwright_tool: Optional[PlaywrightToolWithMemory] = None
browser_executor: Optional[BrowserExecutor] = None


@mcp.resource("telemetry://live")
async def get_telemetry_resource() -> Telemetry:
    status = "running" if playwright_tool and playwright_tool.page else "not_initialized"
    return Telemetry(
        provider="local-playwright",
        status=status,
        live_url=None,
        timestamp=datetime.now().isoformat(),
        cdp_url=None,
        instance_id=None,
    )


@mcp.initialize
async def initialize_environment(ctx):
    global playwright_tool, browser_executor

    metadata = ctx.meta
    progress_token = metadata.get("progressToken", None)

    async def send_progress(progress: int, message: str):
        if progress_token:
            await ctx.session.send_progress_notification(
                progress_token=progress_token,
                progress=progress,
                total=100,
                message=message,
            )
        logger.info(f"[{progress}%] {message}")

    try:
        await send_progress(10, "Starting deep_research initialization...")

        skip_browser = os.getenv("SKIP_BROWSER") in {"1", "true", "True"}

        # Initialize local Playwright tool
        playwright_tool = PlaywrightToolWithMemory(context=None, cdp_url=None)
        if not skip_browser:
            await playwright_tool._ensure_browser()
            await send_progress(40, "Playwright browser launched")
        else:
            await send_progress(40, "Skipping browser launch (SKIP_BROWSER set)")

        # Register playwright tool
        mcp.add_tool(playwright_tool.mcp)
        await send_progress(55, "Playwright tool registered")

        # Initialize executor and computer tools (HUD Computer only, no cloud providers)
        browser_executor = BrowserExecutor(playwright_tool)
        await send_progress(65, "Browser executor initialized")

        # Mount hubs with environment
        setup_hub.env = playwright_tool
        evaluate_hub.env = playwright_tool
        mcp.mount(setup_hub)
        mcp.mount(evaluate_hub)
        await send_progress(80, "Setup and evaluate hubs registered")

        # Navigate to initial URL
        if not skip_browser:
            initial_url = os.getenv("BROWSER_URL") or os.getenv("INITIAL_URL") or (
                "https://en.wikipedia.org/wiki/Main_Page"
            )
            await playwright_tool.navigate(initial_url)
            await send_progress(100, f"Navigated to {initial_url}")
        else:
            await send_progress(100, "Initialization complete (browser launch skipped)")
    except Exception as e:
        if progress_token:
            await ctx.session.send_progress_notification(
                progress_token=progress_token,
                progress=0,
                total=100,
                message=f"Initialization failed: {str(e)}",
            )
        raise


@mcp.shutdown
async def shutdown_environment():
    global playwright_tool, browser_executor
    logger.info("Shutting down deep_research environment")
    try:
        if playwright_tool and playwright_tool._browser:
            await playwright_tool._browser.close()
    except Exception as e:
        logger.error(f"Error closing browser: {e}")
    finally:
        playwright_tool = None
        browser_executor = None


if __name__ == "__main__":
    mcp.run()

