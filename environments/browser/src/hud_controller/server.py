import asyncio
import sys
import logging
import os
import json
from typing import Optional, Any
from pathlib import Path
from datetime import datetime

# Configure logging before imports to go to stderr
logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="[%(levelname)s] %(asctime)s | %(name)s | %(message)s",
    force=True,
)
logger = logging.getLogger(__name__)

from mcp.server.fastmcp import FastMCP, Context
from mcp.types import InitializeRequest, InitializeResult, Implementation
from mcp.shared.context import RequestContext
from mcp.server.models import InitializationOptions

# Import the helper for initialization progress and tool registration
from hud.tools.helper import mcp_intialize_wrapper, register_instance_tool

from .services import ServiceManager
from .evaluators import evaluate_tool
from .setup import setup_tool
from .problems import ProblemRegistry
from .context import initialize_context, get_global_context

service_manager = ServiceManager()

# --- IMPORTANT ---
# Two options for initializing the environment:
# 1. Use the decorator to enable progress notifications during MCP initialization
# 2. Just launch the server and initialize the mcp server after


# --- OPTION 1 ---
# The decorator intercepts the MCP initialization to provide session and progress_token
@mcp_intialize_wrapper()
async def initialize_environment(session=None, progress_token=None):
    """
    Initialize the environment with progress reporting.

    This function works with or without session/progress_token parameters.
    - With them: Sends progress notifications during MCP initialization
    - Without them: Can be called directly before creating the MCP server
    - The decorator intercepts the MCP initialization to provide session and progress_token
    """
    import logging

    logger = logging.getLogger(__name__)
    logger.info(
        f"initialize_environment called! session={session}, progress_token={progress_token}"
    )

    async def send_progress(progress: float, message: str):
        """Send progress notification through the session."""
        if progress_token:
            logger.info(f"Sending progress: {progress}% - {message}")
            await session.send_progress_notification(
                progress_token=progress_token, progress=progress, total=100, message=message
            )
        else:
            logger.info(f"No progress token, skipping: {progress}% - {message}")

    try:
        await send_progress(0, "Starting environment services...")

        # Start core services
        await service_manager.start_services()
        await send_progress(20, "X11 server started")

        # Wait for X11 to be ready
        await service_manager.wait_for_x11()
        await send_progress(40, "X11 ready")

        # Start VNC and wait for it
        await service_manager.wait_for_vnc()
        vnc_message = {"message": "VNC server ready", "live_url": "http://localhost:8080/vnc.html"}
        await send_progress(60, json.dumps(vnc_message))

        # Initialize tools now that X11 is ready
        await send_progress(70, "Initializing tools...")

        # Create and register computer tool
        from hud.tools import (
            HudComputerTool,
            PlaywrightTool,
            AnthropicComputerTool,
            OpenAIComputerTool,
        )

        register_instance_tool(mcp, HudComputerTool())
        register_instance_tool(mcp, AnthropicComputerTool())
        register_instance_tool(mcp, OpenAIComputerTool())

        # Store playwright tool instance for browser launch
        playwright_tool = PlaywrightTool()
        register_instance_tool(mcp, playwright_tool)

        # Initialize global context with service manager and playwright tool
        global_context = initialize_context(service_manager, playwright_tool)

        # Set context on setup and evaluate tools
        setup_tool.context = global_context
        evaluate_tool.context = global_context

        # Register setup and evaluate tools with MCP
        # They will handle their own __call__ methods
        register_instance_tool(mcp, setup_tool)
        register_instance_tool(mcp, evaluate_tool)

        await send_progress(80, "All tools ready")

        # Launch apps and browser in parallel (with error handling)
        launch_apps = os.getenv("LAUNCH_APPS", "")
        browser_url = os.getenv("BROWSER_URL", "")

        if launch_apps:
            await send_progress(85, f"Launching apps: {launch_apps}")

            # Launch apps first
            app_tasks = []
            for app in launch_apps.split(","):
                app = app.strip()
                if app:
                    app_tasks.append(service_manager.launch_app(app))

            # Wait for apps with error handling
            if app_tasks:
                try:
                    app_results = await asyncio.gather(*app_tasks, return_exceptions=True)
                    await send_progress(90, "Apps launched (some may have failed)")

                    # If no browser URL specified, use the first successfully launched app
                    if not browser_url:
                        for result in app_results:
                            if isinstance(result, dict) and "url" in result:
                                browser_url = result["url"]
                                logger.info(f"Auto-navigating to first app: {browser_url}")
                                break
                except Exception as e:
                    logger.error(f"App launch failed: {e}")
                    await send_progress(90, f"Apps failed to launch: {e}")

        # Now launch browser using PlaywrightTool after apps are ready
        try:
            await send_progress(93, "Launching browser...")
            # Just ensure browser is ready (will launch on first use)
            await playwright_tool._ensure_browser()
            await send_progress(95, "Browser launched")

            # Navigate if URL specified
            if browser_url:
                await send_progress(97, f"Navigating to {browser_url}")
                nav_result = await playwright_tool.navigate(browser_url)
                if nav_result.get("success"):
                    await send_progress(
                        98, f"Navigation successful: {nav_result.get('title', browser_url)}"
                    )
                else:
                    await send_progress(
                        98, f"Navigation failed: {nav_result.get('error', 'Unknown error')}"
                    )
            else:
                await send_progress(98, "Browser ready (no navigation URL specified)")

        except Exception as e:
            logger.error(f"Browser launch failed: {e}")
            await send_progress(95, f"Browser failed: {e}")

        await send_progress(100, "Environment ready!")

    except Exception as e:
        if progress_token:
            await session.send_progress_notification(
                progress_token=progress_token,
                progress=0,
                total=100,
                message=f"Initialization failed: {str(e)}",
            )
        raise


# --- OPTION 2 ---
# Just launch the server and initialize the mcp server after
# Or run the services in any other way (e.g. in a separate process)
# await initialize_environment()
# mcp = FastMCP("HUD Browser Environment")
# mcp.run()


# --- MCP SERVER & BASIC TOOLS ---
# Create FastMCP instance
# Note: The mcp_intialize_wrapper above handles the response to the initialize request with progress
mcp = FastMCP(
    name="HUD Browser Environment",
    instructions="""
    This is a browser automation environment with full GUI access.
    Use the computer tool to interact with the browser and applications.
    You can also launch additional apps dynamically with launch_app.
    """,
)


# === PARAMETERIZED MCP RESOURCES ===


@mcp.resource("evaluators://registry")
async def get_evaluators_resource() -> str:
    """MCP resource containing all available evaluators."""
    return evaluate_tool.get_registry_json()


@mcp.resource("evaluators://{env}")
async def get_env_evaluators_resource(env: str) -> str:
    """MCP resource containing environment-specific evaluators."""
    # Get all evaluators and filter by app prefix
    all_evaluators = json.loads(evaluate_tool.get_registry_json())
    env_evaluators = [
        e for e in all_evaluators.get("functions", []) if e.get("name", "").startswith(f"{env}_")
    ]
    return json.dumps(
        {"env": env, "evaluators": env_evaluators, "count": len(env_evaluators)}, indent=2
    )


@mcp.resource("setup://registry")
async def get_setup_registry_resource() -> str:
    """MCP resource containing all available setup tools."""
    return setup_tool.get_registry_json()


@mcp.resource("setup://{env}")
async def get_env_setup_resource(env: str) -> str:
    """MCP resource containing environment-specific setup tools."""
    # Get all setup tools and filter by app prefix
    all_setup = json.loads(setup_tool.get_registry_json())
    env_setup = [
        s for s in all_setup.get("functions", []) if s.get("name", "").startswith(f"{env}_")
    ]
    return json.dumps({"env": env, "setup_tools": env_setup, "count": len(env_setup)}, indent=2)


@mcp.resource("problems://registry")
async def get_problems_registry_resource() -> str:
    """MCP resource containing all available problems."""
    return ProblemRegistry.to_json()


@mcp.resource("problems://{env}")
async def get_env_problems_resource(env: str) -> str:
    """MCP resource containing environment-specific problems."""
    env_problems = ProblemRegistry.get_problems_by_app(env)
    return json.dumps({"env": env, "problems": env_problems, "count": len(env_problems)}, indent=2)


@mcp.resource("schema://evaluator/{evaluator_name}")
async def get_evaluator_schema_resource(evaluator_name: str) -> str:
    """MCP resource containing detailed schema for a specific evaluator."""
    # Get evaluator from registry
    all_evaluators = json.loads(evaluate_tool.get_registry_json())
    for e in all_evaluators.get("functions", []):
        if e.get("name") == evaluator_name:
            return json.dumps(e, indent=2)
    return json.dumps({"error": f"Evaluator '{evaluator_name}' not found"}, indent=2)


@mcp.resource("schema://setup/{setup_name}")
async def get_setup_schema_resource(setup_name: str) -> str:
    """MCP resource containing detailed schema for a specific setup tool."""
    # Get setup tool from registry
    all_setup = json.loads(setup_tool.get_registry_json())
    for s in all_setup.get("functions", []):
        if s.get("name") == setup_name:
            return json.dumps(s, indent=2)
    return json.dumps({"error": f"Setup tool '{setup_name}' not found"}, indent=2)


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
async def launch_app(app_name: str, ctx: Context) -> str:
    """Launch a specific application dynamically.

    Args:
        app_name: Name of the app to launch (e.g., 'todo', 'chat')

    Returns:
        Success message with app URL
    """
    await ctx.info(f"Launching app: {app_name}")
    await ctx.report_progress(0, 100, f"Starting {app_name} app...")

    app_info = await service_manager.launch_app(app_name)

    await ctx.report_progress(100, 100, f"{app_name} app ready!")
    return f"Launched {app_name} at {app_info['url']}"


# API request tool (doesn't need X11)
@mcp.tool()
async def api_request(
    url: str, method: str = "GET", data: dict | None = None, ctx: Context | None = None
) -> dict:
    """Make HTTP API requests.

    Args:
        url: The URL to request
        method: HTTP method (GET, POST, etc.)
        data: Optional JSON data for POST/PUT requests
        ctx: Optional context for logging

    Returns:
        Response data as dict
    """
    import httpx

    if ctx:
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
async def query_database(query: str, ctx: Context) -> list[dict]:
    """Execute a database query (mock implementation).

    Args:
        query: SQL query to execute
        ctx: Context for logging

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
    def run(transport: str = "stdio"):
        """Run the MCP server."""
        mcp.run(transport=transport)

    app()
