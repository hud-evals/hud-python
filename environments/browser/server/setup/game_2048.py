"""2048 game setup tools."""

import logging
from typing import List

from server.main import http_client
from hud.server import MCPRouter

logger = logging.getLogger(__name__)

# Create router for this module
router = MCPRouter()


@router.tool
async def game_2048_board(board_size: int = 4, target_tile: int = 2048):
    """Initialize new game with specified board size and target tile.

    Args:
        board_size: Size of the game board (3-6)
        target_tile: Target tile value (64, 128, 256, 512, 1024, 2048, etc.)

    Returns:
        Game initialization status
    """
    try:
        # Launch the 2048 app first
        response = await http_client.post("/apps/launch", json={"app_name": "2048"})

        if response.status_code != 200:
            return {"error": f"Failed to launch 2048: {response.text}"}

        app_info = response.json()
        backend_port = app_info.get("backend_port", 5001)

        # Initialize new game
        url = f"http://localhost:{backend_port}/api/game/new"
        game_response = await http_client.post(
            url, json={"board_size": board_size, "target_tile": target_tile}
        )
        game_response.raise_for_status()

        return {
            "status": "success",
            "message": f"{board_size}x{board_size} game initialized with target {target_tile}",
            "app_url": app_info["url"],
        }
    except Exception as e:
        logger.error(f"game_2048_board failed: {e}")
        return {"error": f"Failed to initialize game: {str(e)}"}


@router.tool
async def game_2048_set_board(board: List[List[int]], score: int = 0, moves: int = 0):
    """Set specific board configuration for testing.

    Args:
        board: 2D list representing the board state
        score: Initial score
        moves: Initial move count

    Returns:
        Board configuration status
    """
    try:
        # Get app info
        app_response = await http_client.get("/apps/2048")
        if app_response.status_code != 200:
            return {"error": "2048 app not running"}

        app_data = app_response.json()
        backend_port = app_data.get("backend_port", 5001)

        # Set the board configuration
        url = f"http://localhost:{backend_port}/api/eval/set_board"
        response = await http_client.post(
            url, json={"board": board, "score": score, "moves": moves}
        )
        response.raise_for_status()
        result = response.json()

        highest_tile = result.get("highest_tile", 0)

        return {
            "status": "success",
            "message": f"Board set successfully (highest tile: {highest_tile})",
            "highest_tile": highest_tile,
        }
    except Exception as e:
        logger.error(f"game_2048_set_board failed: {e}")
        return {"error": f"Failed to set board: {str(e)}"}


@router.tool
async def game_2048_near_win(target_tile: int = 2048):
    """Set board close to winning (with tiles near target).

    Args:
        target_tile: The target tile for the game

    Returns:
        Near-win board status
    """
    try:
        # Create a board that's one move away from target
        if target_tile == 2048:
            board = [[1024, 1024, 256, 128], [512, 256, 64, 32], [128, 64, 16, 8], [32, 16, 4, 2]]
        elif target_tile == 1024:
            board = [[512, 512, 128, 64], [256, 128, 32, 16], [64, 32, 8, 4], [16, 8, 2, 0]]
        elif target_tile == 512:
            board = [[256, 256, 64, 32], [128, 64, 16, 8], [32, 16, 4, 2], [8, 4, 2, 0]]
        else:
            # Generic near-win board
            half_target = target_tile // 2
            quarter_target = target_tile // 4
            board = [
                [half_target, half_target, quarter_target, quarter_target // 2],
                [quarter_target, quarter_target // 2, 16, 8],
                [16, 8, 4, 2],
                [4, 2, 0, 0],
            ]

        # Set a high score and move count
        score = sum(sum(row) for row in board) * 2
        moves = 150

        result = await game_2048_set_board(board=board, score=score, moves=moves)

        if "error" not in result:
            result["message"] = f"Board set near winning state for target {target_tile}"

        return result
    except Exception as e:
        logger.error(f"game_2048_near_win failed: {e}")
        return {"error": f"Failed to set near-win board: {str(e)}"}


@router.tool
async def game_2048_reset():
    """Reset game to initial state.

    Returns:
        Reset status
    """
    try:
        # Just initialize a new game with defaults
        return await game_2048_board()
    except Exception as e:
        logger.error(f"game_2048_reset failed: {e}")
        return {"error": f"Failed to reset game: {str(e)}"}
