"""MCP server for remote browser environment."""

import asyncio
import sys
import logging
import os
import json
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Optional, TypedDict

# Configure logging
logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="[%(levelname)s] %(asctime)s | %(name)s | %(message)s",
    force=True,
)
logger = logging.getLogger(__name__)

from fastmcp import FastMCP
from hud.tools.helper import mcp_intialize_wrapper

# Import tools
from .tools import PlaywrightToolWithMemory, BrowserExecutor, create_computer_tools

# Import setup and evaluate hubs
from .setup import setup as setup_hub
from .evaluate import evaluate as evaluate_hub

# Import providers
from .providers import get_provider, BrowserProvider

# Global state
browser_provider: Optional[BrowserProvider] = None
playwright_tool: Optional[PlaywrightToolWithMemory] = None
browser_executor: Optional[BrowserExecutor] = None

# Track if we're already cleaning up to prevent double cleanup
_cleanup_in_progress = False


@asynccontextmanager
async def lifespan(app: FastMCP):
    """FastMCP lifespan manager - handles startup and cleanup."""
    global browser_provider, playwright_tool, browser_executor, _cleanup_in_progress

    try:
        yield {}  # Nothing to initialize at startup
    finally:
        # This runs when the server shuts down
        if _cleanup_in_progress:
            logger.info("Cleanup already in progress, skipping")
            return

        _cleanup_in_progress = True
        logger.info("🔧 Lifespan cleanup started")

        try:
            # Only close the browser through the provider - this is what actually terminates the remote session
            # The playwright tool just has a connection to the remote browser, it doesn't need separate cleanup
            if browser_provider:
                try:
                    logger.info("Closing browser provider session...")
                    await asyncio.wait_for(browser_provider.close(), timeout=10.0)
                    logger.info("Browser provider closed successfully")
                except asyncio.TimeoutError:
                    logger.error("Timeout closing browser provider (10s exceeded)")
                except Exception as e:
                    logger.error(f"Error closing browser provider: {e}")
            else:
                logger.info("No browser provider to close")

            logger.info("✅ Lifespan cleanup finished")
        except Exception as e:
            logger.error(f"Error during lifespan cleanup: {e}")
        finally:
            _cleanup_in_progress = False
            browser_provider = None
            playwright_tool = None
            browser_executor = None


# Create FastMCP instance
mcp = FastMCP(
    name="HUD Remote Browser Environment",
    instructions="""
    This is a remote browser automation environment that connects to cloud browser providers.
    The browser provider is configured via the BROWSER_PROVIDER environment variable.
    
    Available tools:
    - setup: Initialize browser environment with various setup functions
    - evaluate: Evaluate browser state with various evaluator functions
    - playwright tools: Browser automation (navigate, click, type, etc.)
    - computer tools: Control browser as if it were a desktop application
    """,
    lifespan=lifespan,
)


class Telemetry(TypedDict):
    """Standard evaluation result format."""

    provider: str
    status: str
    live_url: str | None
    timestamp: str
    cdp_url: str | None
    instance_id: str | None


@mcp.resource("telemetry://live")
async def get_telemetry_resource() -> Telemetry:
    """MCP resource containing telemetry data including provider's live view URL."""
    global browser_provider

    telemetry_data: Telemetry = {
        "provider": os.getenv("BROWSER_PROVIDER", "unknown"),
        "status": "unknown",
        "live_url": None,
        "timestamp": datetime.now().isoformat(),
        "cdp_url": None,
        "instance_id": None,
    }

    if browser_provider:
        try:
            # Get provider status
            status = await browser_provider.get_status()
            telemetry_data.update(
                {
                    "status": "running" if browser_provider.is_running else "stopped",
                    "cdp_url": browser_provider.cdp_url,
                    "instance_id": status.get("instance_id"),
                    "live_url": browser_provider.get_live_view_url(),
                }
            )
        except Exception as e:
            logger.error(f"Error getting telemetry data: {e}")
    else:
        telemetry_data["status"] = "not_initialized"

    return telemetry_data


@mcp_intialize_wrapper()
async def initialize_environment(session=None, progress_token=None):
    """Initialize the remote browser environment with progress reporting."""
    global browser_provider, playwright_tool, browser_executor

    async def send_progress(progress: int, message: str):
        if progress_token and session:
            await session.send_progress_notification(
                progress_token=progress_token,
                progress=progress,
                total=100,
                message=message,
            )
        logger.info(f"[{progress}%] {message}")

    try:
        await send_progress(10, "Starting remote browser environment initialization...")

        # Get provider configuration from environment
        provider_name = os.getenv("BROWSER_PROVIDER")
        if not provider_name:
            error_msg = (
                "BROWSER_PROVIDER environment variable is required. "
                "Supported providers: anchorbrowser, steel, browserbase, hyperbrowser, kernel"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        provider_name = provider_name.lower()
        await send_progress(20, f"Using browser provider: {provider_name}")

        # Initialize the browser provider
        provider_class = get_provider(provider_name)
        provider_config = {}

        # Add provider-specific configuration
        if provider_name == "anchorbrowser":
            provider_config["api_key"] = os.getenv("ANCHOR_API_KEY")
            provider_config["base_url"] = os.getenv(
                "ANCHOR_BASE_URL", "https://api.anchorbrowser.io"
            )
        elif provider_name == "steel":
            provider_config["api_key"] = os.getenv("STEEL_API_KEY")
            provider_config["base_url"] = os.getenv("STEEL_BASE_URL", "https://api.steel.dev")
        elif provider_name == "browserbase":
            provider_config["api_key"] = os.getenv("BROWSERBASE_API_KEY")
            provider_config["project_id"] = os.getenv("BROWSERBASE_PROJECT_ID")
        elif provider_name == "hyperbrowser":
            provider_config["api_key"] = os.getenv("HYPERBROWSER_API_KEY")
        elif provider_name == "kernel":
            provider_config["api_key"] = os.getenv("KERNEL_API_KEY")

        browser_provider = provider_class(provider_config)
        await send_progress(30, "Browser provider initialized")

        # Launch the browser and get CDP URL
        await send_progress(40, "Launching remote browser...")

        # Build launch options
        launch_options = {}

        # Add proxy configuration if environment variables are set
        proxy_type = os.getenv("BROWSER_PROXY_TYPE")
        if proxy_type:
            if proxy_type == "custom":
                proxy_config = {
                    "type": "custom",
                    "server": os.getenv("BROWSER_PROXY_SERVER"),
                    "username": os.getenv("BROWSER_PROXY_USERNAME"),
                    "password": os.getenv("BROWSER_PROXY_PASSWORD"),
                    "active": True,
                }
            elif proxy_type == "anchor_residential":
                proxy_config = {
                    "type": "anchor_residential",
                    "country_code": os.getenv("BROWSER_PROXY_COUNTRY", "us"),
                    "active": True,
                }
            else:
                proxy_config = None

            if proxy_config:
                launch_options["proxy"] = proxy_config
                await send_progress(45, f"Using {proxy_type} proxy")

        # Add other launch options from environment
        max_duration = os.getenv("BROWSER_MAX_DURATION")
        if max_duration:
            launch_options["max_duration"] = int(max_duration)
        idle_timeout = os.getenv("BROWSER_IDLE_TIMEOUT")
        if idle_timeout:
            launch_options["idle_timeout"] = int(idle_timeout)

        # Create browser session
        cdp_url = await browser_provider.launch(**launch_options)
        await send_progress(60, f"Browser launched, CDP URL: {cdp_url}")

        # Initialize PlaywrightToolWithMemory with context as itself
        # The tool itself is the context - it has the page, history, etc.
        playwright_tool = PlaywrightToolWithMemory(context=None, cdp_url=cdp_url)

        # Ensure browser is connected before registering tools
        await playwright_tool._ensure_browser()
        await send_progress(65, "Browser connection established")

        # Add playwright tool to MCP
        mcp.add_tool(playwright_tool.mcp)
        await send_progress(70, "Playwright tool registered")

        # Initialize browser executor
        browser_executor = BrowserExecutor(playwright_tool)
        await send_progress(75, "Browser executor initialized")

        # Create and register computer tools
        computer_tools = create_computer_tools(browser_executor)
        for _, tool_instance in computer_tools.items():
            mcp.add_tool(tool_instance.mcp)
        await send_progress(80, f"Registered {len(computer_tools)} computer tools")

        # Set the playwright_tool as environment for setup and evaluate hubs
        # This allows all setup/evaluate functions to access the browser
        setup_hub.env = playwright_tool
        evaluate_hub.env = playwright_tool

        # Mount the hubs
        mcp.mount(setup_hub)
        mcp.mount(evaluate_hub)
        await send_progress(90, "Setup and evaluate tools registered")

        # Navigate to initial URL if specified
        initial_url = os.getenv("BROWSER_URL")
        if initial_url:
            await send_progress(95, f"Navigating to {initial_url}")
            await playwright_tool.navigate(initial_url)

        await send_progress(100, "Remote browser environment ready!")

    except Exception as e:
        if progress_token and session:
            await session.send_progress_notification(
                progress_token=progress_token,
                progress=0,
                total=100,
                message=f"Initialization failed: {str(e)}",
            )
        raise


if __name__ == "__main__":
    mcp.run()
