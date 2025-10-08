"""MCP server for remote browser environment."""

import sys
import logging
import os
import asyncio
from datetime import datetime
from typing import Optional, TypedDict, Any

# Configure stderr logging
logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="[%(levelname)s] %(asctime)s | %(name)s | %(message)s",
    force=True,
)
logger = logging.getLogger(__name__)

from hud.server import MCPServer
from hud.server.context import attach_context

# Import tools
from .tools import PlaywrightToolWithMemory, BrowserExecutor, AnthropicComputerToolWithRecord, OpenAIComputerToolWithRecord
from hud.tools.computer import (
    AnthropicComputerTool,
    OpenAIComputerTool,
    HudComputerTool,
)
from hud.tools import PlaywrightTool

# Import setup and evaluate hubs
from .setup import setup as setup_hub
from .evaluate import evaluate as evaluate_hub

# Import providers
from .providers import get_provider, BrowserProvider

# Global persistent context (initialized during startup)
persistent_ctx = None
playwright_tool: Optional[PlaywrightToolWithMemory] = None
browser_executor: Optional[BrowserExecutor] = None

# Create Hud FastMCP instance
mcp = MCPServer(
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
    global persistent_ctx

    if persistent_ctx:
        try:
            telemetry = persistent_ctx.get_telemetry()  # Now synchronous
            return Telemetry(
                provider=telemetry["provider"],
                status=telemetry["status"],
                live_url=telemetry["live_url"],
                timestamp=datetime.now().isoformat(),
                cdp_url=None,
                instance_id=telemetry["instance_id"],
            )
        except Exception as e:
            logger.error(f"Error getting telemetry data: {e}")
            # Return default telemetry on error instead of None
            return Telemetry(
                provider=os.getenv("BROWSER_PROVIDER", "unknown"),
                status="error",
                live_url=None,
                timestamp=datetime.now().isoformat(),
                cdp_url=None,
                instance_id=None,
            )

    return Telemetry(
        provider=os.getenv("BROWSER_PROVIDER", "unknown"),
        status="not_initialized",
        live_url=None,
        timestamp=datetime.now().isoformat(),
        cdp_url=None,
        instance_id=None,
    )


@mcp.initialize
async def initialize_environment(ctx):
    """Initialize the remote browser environment with progress reporting."""
    global persistent_ctx, playwright_tool, browser_executor

    # Extract progress token from context if available
    progress_token = None
    if ctx.meta and hasattr(ctx.meta, "progressToken"):
        progress_token = ctx.meta.progressToken

    async def send_progress(progress: int, message: str):
        if progress_token and hasattr(ctx, "session"):
            try:
                await ctx.session.send_progress_notification(
                    progress_token=progress_token,
                    progress=progress,
                    total=100,
                    message=message,
                )
            except Exception as e:
                logger.warning(f"Failed to send progress notification: {e}")
        logger.info(f"[{progress}%] {message}")

    try:
        await send_progress(5, "Connecting to persistent context...")

        # Connect to persistent context server
        max_retries = 10
        retry_delay = 1.0

        for attempt in range(max_retries):
            try:
                persistent_ctx = attach_context("/tmp/hud_remote_browser_ctx.sock")
                if persistent_ctx is None:
                    raise ConnectionError("Failed to attach to context server")
                logger.info("Connected to persistent remote browser context")

                # Log current state
                state = persistent_ctx.get_state_summary()
                logger.info(f"Context state: {state}")

                if persistent_ctx.get_is_initialized():
                    logger.info("Resuming with existing browser session")
                else:
                    logger.info("Starting fresh browser session")
                break

            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(
                        f"Context server not ready yet (attempt {attempt + 1}/{max_retries}): {e}"
                    )
                    await asyncio.sleep(retry_delay)
                else:
                    logger.error(
                        f"Failed to connect to context server after {max_retries} attempts: {e}"
                    )
                    logger.error(
                        "The context server should be started automatically. Check container logs."
                    )
                    raise

        await send_progress(10, "Connected to persistent context")

        # At this point, persistent_ctx is guaranteed to be set
        assert persistent_ctx is not None

        # Check if we need to initialize a new browser session
        if not persistent_ctx.get_is_initialized():
            await send_progress(15, "Initializing new browser session...")

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

            # Store provider config in context
            persistent_ctx.set_provider_config(provider_config)

            browser_provider = provider_class(provider_config)
            persistent_ctx.set_browser_provider(browser_provider)
            await send_progress(30, "Browser provider initialized")

            # Launch the browser and get CDP URL
            await send_progress(40, "Launching remote browser...")

            # Build launch options
            launch_options = {}

            # Add other launch options from environment
            max_duration = os.getenv("BROWSER_MAX_DURATION")
            if max_duration:
                launch_options["max_duration"] = int(max_duration)
            idle_timeout = os.getenv("BROWSER_IDLE_TIMEOUT")
            if idle_timeout:
                launch_options["idle_timeout"] = int(idle_timeout)

            # Store launch options in context
            persistent_ctx.set_launch_options(launch_options)

            # Create browser session
            cdp_url = await browser_provider.launch(**launch_options)

            # Build and store telemetry data
            telemetry_data = {
                "provider": provider_name,
                "status": "running",
                "live_url": browser_provider.get_live_view_url()
                if hasattr(browser_provider, "get_live_view_url")
                else None,
                "cdp_url": cdp_url,
                "instance_id": browser_provider._instance_id
                if hasattr(browser_provider, "_instance_id")
                else None,
                "timestamp": datetime.now().isoformat(),
            }
            persistent_ctx.set_telemetry(telemetry_data)

            await send_progress(60, f"Browser launched")
        else:
            # Reuse existing browser session
            await send_progress(20, "Reusing existing browser session...")

            # Get existing CDP URL from context
            cdp_url = persistent_ctx.get_cdp_url()
            if not cdp_url:
                raise ValueError("No CDP URL in persistent context")

            await send_progress(60, f"Using existing CDP URL")

        # Initialize PlaywrightToolWithMemory with CDP URL from context
        # This reconnects to the existing browser session on reloads
        # playwright_tool = PlaywrightTool(cdp_url=cdp_url)
        playwright_tool = PlaywrightToolWithMemory(context=None, cdp_url=cdp_url)

        # Ensure browser is connected before registering tools
        await playwright_tool._ensure_browser()
        await send_progress(65, "Browser connection established")

        # Add playwright tool to MCP
        mcp.add_tool(playwright_tool)
        await send_progress(70, "Playwright tool registered")

        # Initialize browser executor
        browser_executor = BrowserExecutor(playwright_tool)
        await send_progress(75, "Browser executor initialized")

        # Create and register computer tools with default dimensions
        mcp.add_tool(HudComputerTool(executor=browser_executor))
        # mcp.add_tool(AnthropicComputerTool(executor=browser_executor))
        # mcp.add_tool(OpenAIComputerTool(executor=browser_executor))
        mcp.add_tool(AnthropicComputerToolWithRecord(executor=browser_executor))
        mcp.add_tool(OpenAIComputerToolWithRecord(executor=browser_executor))

        await send_progress(80, "Registered hud computer tools")

        # Set the persistent context as environment for setup and evaluate hubs
        setup_hub.env = persistent_ctx
        evaluate_hub.env = persistent_ctx

        # Also store the current playwright tool on the persistent context
        # Note: This is NOT pickled/persisted, it's just for current session access
        persistent_ctx.playwright_tool = playwright_tool

        # Mount the hubs
        mcp.mount(setup_hub)
        mcp.mount(evaluate_hub)
        await send_progress(90, "Setup and evaluate tools registered")

        # Navigate to initial URL if specified (only for new sessions)
        if not persistent_ctx.get_is_initialized():
            initial_url = os.getenv("BROWSER_URL")
            if initial_url:
                await send_progress(95, f"Navigating to {initial_url}")
                await playwright_tool.navigate(initial_url)

            # Mark as initialized
            persistent_ctx.set_initialized(True)

        await send_progress(100, "Remote browser environment ready!")

    except Exception as e:
        logger.error(f"Initialization failed: {e}")
        import traceback

        logger.error(f"Traceback: {traceback.format_exc()}")
        raise


@mcp.shutdown
async def shutdown_environment():
    """Shutdown the remote browser environment (only called on SIGTERM)."""
    global persistent_ctx, playwright_tool, browser_executor

    logger.info("🔧 SIGTERM received - shutting down browser provider")
    try:
        # Close the browser provider
        if persistent_ctx:
            logger.info("Closing browser provider...")
            try:
                provider = persistent_ctx.get_browser_provider()
                if provider and hasattr(provider, "close"):
                    provider.close()
                    logger.info("Browser provider closed")
            except Exception as e:
                logger.error(f"Error closing provider: {e}")

        logger.info("✅ Browser shutdown completed")
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")
    finally:
        # Clear local references
        playwright_tool = None
        browser_executor = None


if __name__ == "__main__":
    mcp.run()
