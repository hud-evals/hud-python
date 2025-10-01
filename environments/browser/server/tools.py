"""Browser environment MCP tools."""

import asyncio
import logging

import httpx
from server.shared import http_client, ENV_SERVER_URL, playwright
from hud.server import MCPRouter

logger = logging.getLogger(__name__)

# Create router for this module
router = MCPRouter()


@router.tool
async def launch_app(app_name: str) -> str:
    """Launch a specific application dynamically and navigate to it.

    Args:
        app_name: Name of the app to launch (e.g., 'todo', '2048')

    Returns:
        Success message with app URL
    """
    # http_client is imported from controller module

    try:
        # Call environment server to launch app with timeout
        response = await http_client.post(
            "/apps/launch",
            json={"app_name": app_name},
            timeout=60.0,  # 60 second timeout
        )

        if response.status_code == 404:
            return f"App '{app_name}' not found"
        elif response.status_code != 200:
            return f"Failed to launch app: {response.text}"
    except httpx.ReadTimeout:
        return f"Timeout launching app '{app_name}'. The environment server may still be starting up. Try again in a few seconds."
    except httpx.ConnectError:
        return (
            f"Could not connect to environment server. Make sure it's running at {ENV_SERVER_URL}"
        )
    except Exception as e:
        return f"Error launching app '{app_name}': {str(e)}"

    app_info = response.json()
    app_url = app_info["url"]

    # Automatically navigate to the app after launching
    try:
        await playwright(action="navigate", url=app_url)
        # Give the page a moment to fully load
        await asyncio.sleep(1)
        return f"Launched {app_name} at {app_url} and navigated to it"
    except Exception as e:
        logger.warning(f"Could not auto-navigate to app: {e}")
        return f"Launched {app_name} at {app_url} (navigation failed: {e})"


@router.tool
async def api_request(url: str, method: str = "GET", data: dict | None = None) -> dict:
    """Make HTTP API requests.

    Args:
        url: The URL to request
        method: HTTP method (GET, POST, etc.)
        data: Optional JSON data for POST/PUT requests

    Returns:
        Response data as dict
    """
    logger.debug(f"Making {method} request to {url}")

    # Create a separate client for external requests
    # (to not interfere with the persistent environment client)
    async with httpx.AsyncClient() as client:
        response = await client.request(method, url, json=data)
        return {
            "status": response.status_code,
            "data": response.json()
            if response.headers.get("content-type", "").startswith("application/json")
            else response.text,
        }


__all__ = ["playwright"]
