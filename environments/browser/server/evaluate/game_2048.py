"""2048 game evaluation tools."""

import logging
import math

from server.main import http_client
from hud.server import MCPRouter

logger = logging.getLogger(__name__)

# Create router for this module
router = MCPRouter()


@router.tool
async def game_2048_max_number(target: int):
    """Check if player reached target tile value.

    Uses logarithmic reward scaling to match text-2048 implementation.

    Args:
        target: The target tile value to reach

    Returns:
        Evaluation result with logarithmic reward
    """
    try:
        if not http_client:
            return {"error": "HTTP client not initialized", "reward": 0.0, "done": False}

        # Get app info
        app_response = await http_client.get("/apps/2048")
        if app_response.status_code != 200:
            return {"error": "2048 app not running", "reward": 0.0, "done": False}

        app_data = app_response.json()
        backend_port = app_data.get("backend_port", 5001)

        # Get game state
        url = f"http://localhost:{backend_port}/api/game/state"
        response = await http_client.get(url)
        response.raise_for_status()
        game_state = response.json()

        highest_tile = game_state.get("highest_tile", 0)
        score = game_state.get("score", 0)

        # Only give reward if progress has been made
        if score == 0:
            reward = 0.0
        elif target > 1 and highest_tile > 1:
            # Logarithmic reward scale
            reward = min(1.0, math.log(highest_tile) / math.log(target))
        else:
            reward = 0.0

        done = highest_tile >= target

        return {
            "reward": reward,
            "done": done,
            "content": f"Target: {target}, Highest: {highest_tile}",
            "info": {"target": target, "highest_tile": highest_tile, "progress": reward},
        }
    except Exception as e:
        logger.error(f"game_2048_max_number failed: {e}")
        return {
            "reward": 0.0,
            "done": False,
            "content": f"Failed to evaluate max number: {str(e)}",
            "isError": True,
        }


@router.tool
async def game_2048_efficiency(min_ratio: float):
    """Evaluate game efficiency based on score/moves ratio.

    Args:
        min_ratio: The minimum efficiency ratio to achieve

    Returns:
        Evaluation result with linear reward based on efficiency
    """
    try:
        if not http_client:
            return {"error": "HTTP client not initialized", "reward": 0.0, "done": False}

        # Get app info
        app_response = await http_client.get("/apps/2048")
        if app_response.status_code != 200:
            return {"error": "2048 app not running", "reward": 0.0, "done": False}

        app_data = app_response.json()
        backend_port = app_data.get("backend_port", 5001)

        # Get game state
        url = f"http://localhost:{backend_port}/api/game/state"
        response = await http_client.get(url)
        response.raise_for_status()
        game_state = response.json()

        score = game_state.get("score", 0)
        moves = game_state.get("moves", 0)

        # Calculate the efficiency ratio directly
        ratio = score / moves if moves > 0 else 0.0

        # Linear reward: proportional to ratio / min_ratio, capped at 1.0
        reward = min(1.0, ratio / min_ratio) if min_ratio > 0 else 0.0
        done = ratio >= min_ratio

        return {
            "reward": reward,
            "done": done,
            "content": f"Efficiency: {ratio:.2f} (target: {min_ratio})",
            "info": {"score": score, "moves": moves, "ratio": ratio, "target_ratio": min_ratio},
        }
    except Exception as e:
        logger.error(f"game_2048_efficiency failed: {e}")
        return {
            "reward": 0.0,
            "done": False,
            "content": f"Failed to evaluate efficiency: {str(e)}",
            "isError": True,
        }


@router.tool
async def game_2048_score_reached(target_score: int):
    """Check if player reached target score.

    Args:
        target_score: The target score to reach

    Returns:
        Evaluation result
    """
    try:
        if not http_client:
            return {"error": "HTTP client not initialized", "reward": 0.0, "done": False}

        # Get app info
        app_response = await http_client.get("/apps/2048")
        if app_response.status_code != 200:
            return {"error": "2048 app not running", "reward": 0.0, "done": False}

        app_data = app_response.json()
        backend_port = app_data.get("backend_port", 5001)

        # Get game state
        url = f"http://localhost:{backend_port}/api/game/state"
        response = await http_client.get(url)
        response.raise_for_status()
        game_state = response.json()
        score = game_state.get("score", 0)

        # Linear reward based on score progress
        reward = min(1.0, score / target_score) if target_score > 0 else 0.0
        done = score >= target_score

        return {
            "reward": reward,
            "done": done,
            "content": f"Score: {score} (target: {target_score})",
            "info": {"score": score, "target_score": target_score, "progress": reward},
        }
    except Exception as e:
        logger.error(f"game_2048_score_reached failed: {e}")
        return {
            "reward": 0.0,
            "done": False,
            "content": f"Failed to evaluate score: {str(e)}",
            "isError": True,
        }


@router.tool
async def game_2048_game_won():
    """Check if game is won (reached target tile).

    Returns:
        Evaluation result
    """
    try:
        if not http_client:
            return {"error": "HTTP client not initialized", "reward": 0.0, "done": False}

        # Get app info
        app_response = await http_client.get("/apps/2048")
        if app_response.status_code != 200:
            return {"error": "2048 app not running", "reward": 0.0, "done": False}

        app_data = app_response.json()
        backend_port = app_data.get("backend_port", 5001)

        # Get game state
        url = f"http://localhost:{backend_port}/api/game/state"
        response = await http_client.get(url)
        response.raise_for_status()
        game_state = response.json()

        won = game_state.get("won", False)
        highest_tile = game_state.get("highest_tile", 0)
        target_tile = game_state.get("target_tile", 2048)

        return {
            "reward": 1.0 if won else 0.0,
            "done": won,
            "content": f"Game {'won' if won else 'in progress'} (highest: {highest_tile}, target: {target_tile})",
            "info": {"won": won, "highest_tile": highest_tile, "target_tile": target_tile},
        }
    except Exception as e:
        logger.error(f"game_2048_game_won failed: {e}")
        return {
            "reward": 0.0,
            "done": False,
            "content": f"Failed to check win status: {str(e)}",
            "isError": True,
        }
