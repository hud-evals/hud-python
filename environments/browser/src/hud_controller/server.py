# Suppress warnings
import warnings

warnings.filterwarnings("ignore", category=SyntaxWarning)
warnings.filterwarnings("ignore", message="Xlib.xauth: warning")
warnings.filterwarnings("ignore", module="Xlib")

import sys
import logging
import os
import json
import asyncio
from datetime import datetime
import contextlib

# Configure logging before imports to go to stderr
logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="[%(levelname)s] %(asctime)s | %(name)s | %(message)s",
    force=True,
)
logger = logging.getLogger(__name__)

from fastmcp import Context  # for type annotations
from hud.server import MCPServer
from hud.server.context import attach_context

from .evaluate import evaluate as evaluate_hub
from .setup import setup as setup_hub
from .problems import ProblemRegistry

# Global persistent context (initialized during startup)
persistent_ctx = None
service_manager = None
playwright_tool = None  # Store playwright tool globally for launch_app

# Create main server first so decorators can reference it
# Note: MCPServer (from HUD SDK) automatically redirects stdout to stderr
# during initialization to prevent library prints from corrupting MCP protocol
mcp = MCPServer(
    name="HUD Browser Environment",
    instructions="""
    This is a browser automation environment with full GUI access.
    Use the computer tool to interact with the browser and applications.
    You can also launch additional apps dynamically with launch_app.
    """,
)


# The decorator intercepts the MCP initialization to provide RequestContext
@mcp.initialize
async def initialize_environment(ctx):
    """Initialize the browser environment with clean startup sequence."""
    global persistent_ctx, service_manager, playwright_tool

    logger.info("Initializing browser environment...")

    # Connect to persistent context server (must be running)
    max_retries = 10
    retry_delay = 1.0  # seconds

    for attempt in range(max_retries):
        try:
            persistent_ctx = attach_context("/tmp/hud_browser_ctx.sock")
            logger.info("Connected to persistent browser context server")

            # Get service manager from persistent context
            service_manager = persistent_ctx.get_service_manager()

            # Log current state
            state = persistent_ctx.get_state_summary()
            logger.info(f"Context state: {state}")

            if persistent_ctx.get_is_initialized():
                logger.info("Resuming with existing browser environment")
            else:
                logger.info("Starting fresh browser environment")
            break  # Success, exit the retry loop

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

    # At this point, persistent_ctx and service_manager are guaranteed to be set
    assert persistent_ctx is not None
    assert service_manager is not None

    try:
        # Only start services if not already initialized
        if not persistent_ctx.get_is_initialized():
            logger.info("Starting core services...")

            # Start ONLY core services (X11, VNC) - NO app launching
            await service_manager.start_services()
            logger.info("X11 server started")

            # Wait for X11 to be ready
            await service_manager.wait_for_x11()
            logger.info("X11 ready")

            # Start VNC and wait for it
            await service_manager.wait_for_vnc()
            logger.info("VNC server ready at http://localhost:8080/vnc.html")
        else:
            # Services already running from previous session
            logger.info("Using existing X11 server")
            logger.info("VNC server already running at http://localhost:8080/vnc.html")

        # Initialize browser tools
        logger.info("Initializing browser tools...")
        from hud.tools import (
            HudComputerTool,
            PlaywrightTool,
            AnthropicComputerTool,
            OpenAIComputerTool,
        )

        # Always create new tools and context (they can't be pickled/shared)
        # But the underlying services (X11, VNC, apps) persist
        playwright_tool = PlaywrightTool()
        await playwright_tool._ensure_browser()
        logger.info("Created Playwright browser instance")

        # Set context on hubs (they'll access service_manager through it)
        setup_hub.env = persistent_ctx
        evaluate_hub.env = persistent_ctx

        # Store playwright tool on context for evaluate functions that need it
        # Note: This is NOT pickled/persisted, it's just for current session access
        persistent_ctx.playwright_tool = playwright_tool

        logger.info("Configured hubs with browser context")

        # Mount hubs
        mcp.mount(setup_hub)
        mcp.mount(evaluate_hub)
        logger.info("Mounted setup and evaluate hubs")

        # Register interaction tools
        with contextlib.redirect_stdout(sys.stderr):
            # Check for agent-specific environment variables
            tool_kwargs = {}
            
            # Check for agent display dimensions via environment variables
            agent_width = os.getenv("AGENT_DISPLAY_WIDTH")
            agent_height = os.getenv("AGENT_DISPLAY_HEIGHT")
            
            if agent_width:
                tool_kwargs['width'] = int(agent_width)
            if agent_height:
                tool_kwargs['height'] = int(agent_height)
            
            # Enable image rescaling when agent provides dimensions
            if 'width' in tool_kwargs or 'height' in tool_kwargs:
                tool_kwargs['rescale_images'] = True
                logger.info(f"Agent display dimensions detected: {tool_kwargs}")
            else:
                logger.info("No agent display dimensions provided, using defaults")
            
            # Create computer tools with agent configuration
            mcp.add_tool(HudComputerTool(**tool_kwargs))
            mcp.add_tool(AnthropicComputerTool(**tool_kwargs))
            mcp.add_tool(OpenAIComputerTool(**tool_kwargs))
            mcp.add_tool(playwright_tool)

        # Mark as initialized
        if not persistent_ctx.get_is_initialized():
            persistent_ctx.set_initialized(True)

        logger.info("Browser environment ready!")

    except Exception as e:
        logger.error(f"Initialization failed: {e}")
        import traceback

        logger.error(f"Traceback: {traceback.format_exc()}")
        raise


# === MCP RESOURCES ===


@mcp.resource("problems://registry")
async def get_problems_registry_resource() -> str:
    """MCP resource containing all available problems."""
    return ProblemRegistry.to_json()


@mcp.resource("problems://{env}")
async def get_env_problems_resource(env: str) -> str:
    """MCP resource containing environment-specific problems."""
    env_problems = ProblemRegistry.get_problems_by_app(env)
    return json.dumps({"env": env, "problems": env_problems, "count": len(env_problems)}, indent=2)


@mcp.resource("schema://problem/{problem_name}")
async def get_problem_schema_resource(problem_name: str) -> str:
    """MCP resource containing detailed schema for a specific problem."""
    schema = ProblemRegistry.get_problem_schema(problem_name)
    return json.dumps(schema, indent=2)


@mcp.resource("telemetry://live")
async def get_telemetry_resource() -> str:
    """MCP resource containing telemetry data including VNC live_url."""
    telemetry_data = {
        "live_url": "http://localhost:8080/vnc.html",
        "display": os.getenv("DISPLAY", ":1"),
        "vnc_port": 8080,
        "websockify_port": 8080,
        "status": "ready",
        "timestamp": datetime.now().isoformat(),
        "services": {"x11": "running", "vnc": "running", "websockify": "running"},
    }
    return json.dumps(telemetry_data, indent=2)


# === APPLICATION TOOLS ===


@mcp.tool()
async def launch_app(ctx: Context, app_name: str) -> str:
    """Launch a specific application dynamically and navigate to it.

    Args:
        app_name: Name of the app to launch (e.g., 'todo', '2048')

    Returns:
        Success message with app URL
    """
    await ctx.info(f"Launching app: {app_name}")

    assert persistent_ctx is not None, "Persistent context not initialized"

    # Get the service manager from persistent context to ensure state consistency
    ctx_service_manager = persistent_ctx.get_service_manager()
    assert ctx_service_manager is not None, "Service manager not available from context"

    app_info = await ctx_service_manager.launch_app(app_name)
    app_url = app_info["url"]

    # Get the port information from service manager while we have access
    try:
        backend_port = ctx_service_manager.get_app_port(app_name)
        frontend_port = ctx_service_manager.get_app_frontend_port(app_name)

        # Store ports in persistent context
        persistent_ctx.set_app_ports(app_name, frontend_port, backend_port)
    except Exception as e:
        logger.error(f"Failed to get ports for {app_name}: {e}")

    # Track in persistent context
    persistent_ctx.add_running_app(app_name)

    # Automatically navigate to the app after launching
    # Use the global playwright tool to navigate
    if playwright_tool:
        try:
            await playwright_tool.navigate(app_url)
            # Give the page a moment to fully load
            await asyncio.sleep(1)
            return f"Launched {app_name} at {app_url} and navigated to it"
        except Exception as e:
            logger.warning(f"Could not auto-navigate to app: {e}")

    return f"App initialized."


# API request tool (doesn't need X11)
@mcp.tool()
async def api_request(
    ctx: Context, url: str, method: str = "GET", data: dict | None = None
) -> dict:
    """Make HTTP API requests.

    Args:
        url: The URL to request
        method: HTTP method (GET, POST, etc.)
        data: Optional JSON data for POST/PUT requests

    Returns:
        Response data as dict
    """
    import httpx

    await ctx.debug(f"Making {method} request to {url}")

    async with httpx.AsyncClient() as client:
        response = await client.request(method, url, json=data)
        return {
            "status": response.status_code,
            "data": response.json()
            if response.headers.get("content-type", "").startswith("application/json")
            else response.text,
        }


@mcp.tool()
async def query_database(ctx: Context, query: str) -> list[dict]:
    """Execute a database query (mock implementation).

    Args:
        query: SQL query to execute

    Returns:
        Query results as list of dicts
    """
    await ctx.warning("This is a mock database query tool")

    # Mock implementation
    if "users" in query.lower():
        return [
            {"id": 1, "name": "Alice", "email": "alice@example.com"},
            {"id": 2, "name": "Bob", "email": "bob@example.com"},
        ]
    return []


if __name__ == "__main__":
    import typer

    app = typer.Typer()

    @app.command()
    def run():
        """Run the MCP server."""
        mcp.run(transport="stdio")

    app()
