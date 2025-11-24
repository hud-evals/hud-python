import sys
import logging
from pathlib import Path
from hud.server import MCPServer

from .tools import JupyterToolWithRecord as JupyterTool
from .setup import setup as setup_hub
from .evaluate import evaluate as evaluate_hub

logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="[%(levelname)s] %(asctime)s | %(name)s | %(message)s",
)

logger = logging.getLogger(__name__)

mcp = MCPServer(name="Jupyter")

# Global tool instance
jupyter_tool = None


@mcp.initialize
async def initialize_environment():
    """Initialize the Jupyter environment."""
    global jupyter_tool
    logger.info("Initializing jupyter environment")

    # Create tool (kernel will be created on first use)
    jupyter_tool = JupyterTool(url_suffix="localhost:8888", kernel_name="python3")
    mcp.add_tool(jupyter_tool)

    # Ensure kernel is started and register it for reuse
    await jupyter_tool._ensure_kernel()
    JupyterTool.register_shared_kernel("SpreadSheetBench", jupyter_tool._kernel_id)

    # Set environment on hubs so they can access the jupyter tool
    setup_hub.env = jupyter_tool
    evaluate_hub.env = jupyter_tool

    # Mount hubs (this creates dispatcher tools, hiding individual tools from agents)
    mcp.mount(setup_hub)
    mcp.mount(evaluate_hub)

    logger.info("Jupyter environment initialized successfully")


@mcp.shutdown
async def shutdown_environment():
    """Clean shutdown of the Jupyter environment."""
    global jupyter_tool

    if jupyter_tool:
        await jupyter_tool.shutdown()
        logger.info("Jupyter kernel shut down")


if __name__ == "__main__":
    mcp.run()
