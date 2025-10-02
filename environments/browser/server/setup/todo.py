"""Todo app setup tools."""

import logging
from typing import Dict, Any, List

from server.main import http_client
from server.tools import playwright
from hud.server import MCPRouter

logger = logging.getLogger(__name__)

# Create router for this module
router = MCPRouter()


@router.tool()
async def todo_seed(num_items: int = 5):
    """Seed database with default test todos.

    Args:
        num_items: Number of test items to create (default: 5)

    Returns:
        Setup result with seeded items info
    """
    try:
        # Launch the todo app first
        response = await http_client.post("/apps/launch", json={"app_name": "todo"})

        if response.status_code != 200:
            return {"error": f"Failed to launch todo: {response.text}"}

        app_info = response.json()
        backend_port = app_info.get("backend_port", 5000)

        # Call the app's seed API
        url = f"http://localhost:{backend_port}/api/eval/seed"
        seed_response = await http_client.post(url)
        seed_response.raise_for_status()
        result = seed_response.json()

        # Navigate to the app and reload to show seeded items
        try:
            await playwright(action="navigate", url=app_info["url"])
            # Small delay to ensure navigation completes
            import asyncio

            await asyncio.sleep(0.5)
        except Exception:
            pass

        return {
            "status": "success",
            "message": f"Seeded database with {result.get('items_created', num_items)} test items",
            "app_url": app_info["url"],
        }
    except Exception as e:
        logger.error(f"todo_seed failed: {e}")
        return {"error": f"Failed to seed database: {str(e)}"}


@router.tool()
async def todo_reset():
    """Reset database to empty state.

    Returns:
        Setup result with reset confirmation
    """
    try:
        # Get app info
        app_response = await http_client.get("/apps/todo")
        if app_response.status_code != 200:
            return {"error": "Todo app not running"}

        app_data = app_response.json()
        backend_port = app_data.get("backend_port", 5000)

        # Call the app's reset API
        url = f"http://localhost:{backend_port}/api/eval/reset"
        reset_response = await http_client.delete(url)
        reset_response.raise_for_status()

        return {"status": "success", "message": "Database reset to empty state"}
    except Exception as e:
        logger.error(f"todo_reset failed: {e}")
        return {"error": f"Failed to reset database: {str(e)}"}


@router.tool()
async def todo_custom_seed(items: List[Dict[str, Any]]):
    """Seed database with custom todo items.

    Args:
        items: List of todo items to create, each with 'title' and optional 'completed'

    Returns:
        Setup result with seeded items info
    """
    try:
        # Ensure all items have required fields
        formatted_items = []
        for item in items:
            formatted_item = {
                "title": item.get("title", ""),
                "description": item.get("description", ""),
                "completed": item.get("completed", False),
            }
            formatted_items.append(formatted_item)

        # Launch app if needed
        response = await http_client.post("/apps/launch", json={"app_name": "todo"})

        if response.status_code != 200:
            return {"error": f"Failed to launch todo: {response.text}"}

        app_info = response.json()
        backend_port = app_info.get("backend_port", 5000)

        # Call the app's custom seed API
        url = f"http://localhost:{backend_port}/api/eval/seed_custom"
        seed_response = await http_client.post(url, json=formatted_items)
        seed_response.raise_for_status()

        return {
            "status": "success",
            "message": f"Seeded database with {len(items)} custom items",
            "app_url": app_info["url"],
        }
    except Exception as e:
        logger.error(f"todo_custom_seed failed: {e}")
        return {"error": f"Failed to seed custom items: {str(e)}"}
