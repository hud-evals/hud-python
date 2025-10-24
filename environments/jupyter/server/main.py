import sys
import logging
from pathlib import Path
from hud.server import MCPServer

from .tools import JupyterToolWithRecord as JupyterTool
from .setup import router as setup_router
from .evaluate import router as evalute_router

logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="[%(levelname)s] %(asctime)s | %(name)s | %(message)s",
)

logger = logging.getLogger(__name__)

mcp = MCPServer(name="Jupyter")
mcp.include_router(setup_router)
mcp.include_router(evalute_router)

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
