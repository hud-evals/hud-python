"""
MCP server for Minetest GUI environment using HudComputerTool and hubs.
"""

import sys
import logging

from hud.server import MCPServer
from hud.server.context import attach_context

from .setup import setup as setup_hub
from .evaluate import evaluate as evaluate_hub


# Configure logging to stderr
logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="[%(levelname)s] %(asctime)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


# Create main server first
mcp = MCPServer(name="HUD Minetest Environment")


@mcp.initialize
async def initialize_environment(ctx):
    """Initialize Minetest environment: connect to context and register tools."""
    logger.info("Initializing Minetest environment...")

    # Connect to persistent context server
    game_ctx = attach_context("/tmp/hud_minetest_ctx.sock")

    # Attach hubs to context
    setup_hub.env = game_ctx
    evaluate_hub.env = game_ctx

    # Mount hubs
    mcp.mount(setup_hub)
    mcp.mount(evaluate_hub)

    # Register computer tool for real GUI control (auto-selects XDO/PyAutoGUI)
    from hud.tools import HudComputerTool, AnthropicComputerTool, OpenAIComputerTool

    mcp.add_tool(HudComputerTool())
    mcp.add_tool(AnthropicComputerTool())
    mcp.add_tool(OpenAIComputerTool())

    logger.info("Minetest environment ready (computer tool registered)")


if __name__ == "__main__":
    mcp.run()

