"""Todo app evaluation tools."""

import logging

from server.main import http_client
from hud.server import MCPRouter

logger = logging.getLogger(__name__)

# Create router for this module
router = MCPRouter()


@router.tool
async def todo_completed(expected_count: int):
    """Check if expected number of todos are completed.

    Args:
        expected_count: The expected number of completed todos

    Returns:
        Evaluation result
    """
    try:
        # Get app info
        app_response = await http_client.get("/apps/todo")
        if app_response.status_code != 200:
            return {"error": "Todo app not running", "reward": 0.0, "done": True}

        app_data = app_response.json()
        backend_port = app_data.get("backend_port", 5000)

        # Get stats
        url = f"http://localhost:{backend_port}/api/eval/stats"
        response = await http_client.get(url)
        response.raise_for_status()
        stats = response.json()

        completed_count = stats.get("completed_items", 0)
        total_count = stats.get("total_items", 0)

        success = completed_count >= expected_count
        reward = (
            1.0 if success else (completed_count / expected_count if expected_count > 0 else 0.0)
        )

        return {
            "reward": reward,
            "done": success,
            "info": {
                "success": success,
                "completed_count": completed_count,
                "total_count": total_count,
                "expected_count": expected_count,
                "message": f"Found {completed_count} completed todos (expected {expected_count})",
            },
        }
    except Exception as e:
        logger.error(f"todo_completed failed: {e}")
        return {
            "reward": 0.0,
            "done": True,
            "info": {
                "success": False,
                "error": str(e),
                "message": "Failed to evaluate completed todos",
            },
        }


@router.tool
async def todo_exists(title: str):
    """Check if a todo with specific title exists.

    Args:
        title: The title of the todo to check

    Returns:
        Evaluation result
    """
    try:
        # Get app info
        app_response = await http_client.get("/apps/todo")
        if app_response.status_code != 200:
            return {"error": "Todo app not running", "reward": 0.0, "done": True}

        app_data = app_response.json()
        backend_port = app_data.get("backend_port", 5000)

        # Get todos
        url = f"http://localhost:{backend_port}/api/eval/todos"
        response = await http_client.get(url)
        response.raise_for_status()
        todos = response.json()

        # Check if any todo has the expected title
        exists = any(todo.get("title") == title for todo in todos)

        return {
            "reward": 1.0 if exists else 0.0,
            "done": exists,
            "info": {
                "success": exists,
                "title": title,
                "total_todos": len(todos),
                "message": f"Todo '{title}' {'exists' if exists else 'does not exist'}",
            },
        }
    except Exception as e:
        logger.error(f"todo_exists failed: {e}")
        return {
            "reward": 0.0,
            "done": True,
            "info": {
                "success": False,
                "error": str(e),
                "message": f"Failed to check if todo '{title}' exists",
            },
        }


@router.tool
async def todo_completion_rate(target_rate: float = 0.5):
    """Check if completion rate meets target.

    Args:
        target_rate: The target completion rate (default 0.5)

    Returns:
        Evaluation result
    """
    try:
        # Get app info
        app_response = await http_client.get("/apps/todo")
        if app_response.status_code != 200:
            return {"error": "Todo app not running", "reward": 0.0, "done": True}

        app_data = app_response.json()
        backend_port = app_data.get("backend_port", 5000)

        # Get stats
        url = f"http://localhost:{backend_port}/api/eval/stats"
        response = await http_client.get(url)
        response.raise_for_status()
        stats = response.json()

        total_count = stats.get("total_items", 0)
        completed_count = stats.get("completed_items", 0)

        if total_count == 0:
            # No todos, consider this as 0% completion
            actual_rate = 0.0
        else:
            actual_rate = completed_count / total_count

        success = actual_rate >= target_rate
        reward = min(1.0, actual_rate / target_rate) if target_rate > 0 else 1.0

        return {
            "reward": reward,
            "done": success,
            "info": {
                "success": success,
                "actual_rate": actual_rate,
                "target_rate": target_rate,
                "completed_count": completed_count,
                "total_count": total_count,
                "message": f"Completion rate: {actual_rate:.1%} (target: {target_rate:.1%})",
            },
        }
    except Exception as e:
        logger.error(f"todo_completion_rate failed: {e}")
        return {
            "reward": 0.0,
            "done": True,
            "info": {
                "success": False,
                "error": str(e),
                "message": "Failed to evaluate completion rate",
            },
        }


@router.tool
async def todo_total_count(min_count: int = 1):
    """Check if total todo count meets minimum.

    Args:
        min_count: The minimum number of todos expected (default 1)

    Returns:
        Evaluation result
    """
    try:
        # Get app info
        app_response = await http_client.get("/apps/todo")
        if app_response.status_code != 200:
            return {"error": "Todo app not running", "reward": 0.0, "done": True}

        app_data = app_response.json()
        backend_port = app_data.get("backend_port", 5000)

        # Get stats
        url = f"http://localhost:{backend_port}/api/eval/stats"
        response = await http_client.get(url)
        response.raise_for_status()
        stats = response.json()
        total_count = stats.get("total_items", 0)

        success = total_count >= min_count
        reward = 1.0 if success else (total_count / min_count if min_count > 0 else 0.0)

        return {
            "reward": reward,
            "done": success,
            "info": {
                "success": success,
                "total_count": total_count,
                "min_count": min_count,
                "message": f"Found {total_count} todos (minimum: {min_count})",
            },
        }
    except Exception as e:
        logger.error(f"todo_total_count failed: {e}")
        return {
            "reward": 0.0,
            "done": True,
            "info": {
                "success": False,
                "error": str(e),
                "message": "Failed to evaluate total count",
            },
        }
