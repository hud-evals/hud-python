"""
MCP server for text-based 2048 game environment using BaseHub pattern.
"""

import sys
import logging
from fastmcp import FastMCP

from hud.tools.helper import mcp_intialize_wrapper

from .game import Game2048
from .tools import MoveTool

# Configure logging to stderr
logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Global game instance (initialized during startup)
game = None

# Import setup/evaluate layers
from .setup import setup as setup_hub
from .evaluate import evaluate as evaluate_hub

# Create main server first
mcp = FastMCP(name="text-2048")


@mcp_intialize_wrapper()
async def initialize_environment(session=None, progress_token=None):
    """Initialize the 2048 environment with progress notifications."""
    global game

    logger.info("Initializing 2048 environment...")

    if session and progress_token:
        await session.send_progress_notification(
            progress_token=progress_token,
            progress=0,
            total=100,
            message="Starting 2048 game environment...",
        )

    # Create and initialize the game
    if session and progress_token:
        await session.send_progress_notification(
            progress_token=progress_token,
            progress=30,
            total=100,
            message="Creating game instance...",
        )

    game = Game2048()

    if session and progress_token:
        await session.send_progress_notification(
            progress_token=progress_token,
            progress=50,
            total=100,
            message="Setting up game board...",
        )

    # Initialize game state
    game.reset()

    # Set up the game instance on hubs and tools
    if session and progress_token:
        await session.send_progress_notification(
            progress_token=progress_token,
            progress=70,
            total=100,
            message="Configuring tools and hubs...",
        )

    setup_hub.env = game
    evaluate_hub.env = game

    # Create and register move tool
    move_tool = MoveTool(env=game)
    mcp.add_tool(move_tool.mcp)

    # Mount hubs
    logger.info(f"Mounting setup hub: {setup_hub}")
    logger.info(f"Setup hub tools before mount: {hasattr(setup_hub, '_tool_manager')}")
    
    # Check what tools are in the hub before mounting
    if hasattr(setup_hub, '_tool_manager'):
        logger.info(f"Setup hub tools: {list(setup_hub._tool_manager._tools.keys())}")
    
    mcp.mount(setup_hub)
    
    logger.info(f"Mounting evaluate hub: {evaluate_hub}")
    if hasattr(evaluate_hub, '_tool_manager'):
        logger.info(f"Evaluate hub tools: {list(evaluate_hub._tool_manager._tools.keys())}")
    
    mcp.mount(evaluate_hub)
    
    # Check what tools are available after mounting
    logger.info(f"Main server tools after mounting: {list(mcp._tool_manager._tools.keys())}")

    if session and progress_token:
        await session.send_progress_notification(
            progress_token=progress_token,
            progress=100,
            total=100,
            message="2048 environment ready!",
        )

    logger.info("2048 environment ready")


if __name__ == "__main__":
    mcp.run()
