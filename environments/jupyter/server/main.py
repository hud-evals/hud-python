import sys
import os
import logging
import asyncio
import threading
from pathlib import Path
from hud.server import MCPServer
# from .api import create_spreadsheetbench_app
from .jupyter import JupyterKernel, JupyterKernelWrapper
from .tools import JupyterTool
from .setup import setup
from .evaluate import evaluate
from .config import VOLUMES_PATH
import tornado.httpserver
import tornado.ioloop

# Always log to stderr – stdout is reserved for JSON-RPC
logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="[%(levelname)s] %(asctime)s | %(name)s | %(message)s",
)

logger = logging.getLogger(__name__)

# Create the server early so decorators can reference it
mcp = MCPServer(name="Jupyter")

# Global kernel instance (single kernel per container, like original SpreadsheetBench)
jupyter_kernel = None

@mcp.initialize
async def initialize_environment(ctx=None):
    """Initialize the Jupyter environment with single kernel (like original SpreadsheetBench)."""
    global jupyter_kernel

    # Extract progress token from context
    progress_token = getattr(ctx.meta, "progressToken", None) if ctx and ctx.meta else None

    # Send progress updates if available
    async def send_progress(progress: int, message: str):
        if progress_token and ctx and ctx.session:
            await ctx.session.send_progress_notification(
                progress_token=progress_token,
                progress=progress,
                total=100,
                message=message,
            )

    await send_progress(10, "Starting jupyter environment...")
    logger.info("Initializing jupyter environment")

    # Initialize single Jupyter kernel (original SpreadsheetBench approach)
    wrapper = JupyterKernelWrapper(name="mcp-kernel")
    url_suffix = wrapper.__enter__()

    jupyter_kernel = JupyterKernel(url_suffix, "mcp-main")
    await jupyter_kernel.initialize()

    await send_progress(40, "Jupyter kernel initialized...")

    # Setup workspace directories
    workspace = Path("/app/notebooks")
    workspace.mkdir(exist_ok=True)

    sample_files = Path(VOLUMES_PATH)
    sample_files.mkdir(exist_ok=True)

    shared_data = Path("/app/shared_data")
    shared_data.mkdir(exist_ok=True)

    await send_progress(60, "Setting up directories...")
    await send_progress(80, "Registering tools...")

    # Register Jupyter tool (BaseTool pattern)
    jupyter_tool = JupyterTool(kernel=jupyter_kernel)
    mcp.add_tool(jupyter_tool.mcp)

    # Register setup and evaluate hubs
    mcp.mount(setup)
    mcp.mount(evaluate)

    await send_progress(100, "Jupyter environment ready!")
    logger.info("Jupyter environment initialized successfully")


@mcp.shutdown
async def shutdown_environment():
    """Clean shutdown of the Jupyter environment."""
    global jupyter_kernel

    if jupyter_kernel:
        await jupyter_kernel.shutdown_async()
        logger.info("Jupyter kernel shut down")


if __name__ == "__main__":
    mcp.run()
