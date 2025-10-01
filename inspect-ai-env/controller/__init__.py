"""Controller package - registers hooks and tools."""

import sys
import os
import httpx
import logging
import warnings
import atexit
from contextlib import asynccontextmanager

from hud.server import MCPServer

logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="[%(levelname)s] %(asctime)s | %(name)s | %(message)s",
    force=True,  # Force all loggers to use stderr
)

# Suppress httpx INFO logs to avoid cluttering MCP protocol
httpx_logger = logging.getLogger("httpx")
httpx_logger.setLevel(logging.WARNING)  # Only show warnings and errors
httpcore_logger = logging.getLogger("httpcore")
httpcore_logger.setLevel(logging.WARNING)  # Only show warnings and errors

logger = logging.getLogger(__name__)

# Create a lifespan context manager to handle cleanup
@asynccontextmanager
async def lifespan(app):
    """Ensure HTTP client is closed on server shutdown."""
    # Startup
    yield
    # Shutdown - this runs regardless of how the server stops
    logger.info("Lifespan shutdown: closing HTTP client")
    if http_client:
        await http_client.aclose()
        logger.info("HTTP client closed")

mcp = MCPServer(name="inspect_ai_env", lifespan=lifespan)

http_client = httpx.AsyncClient(
    base_url="http://localhost:8000", timeout=10.0
)

# Import tools and hooks to register them with the server
from . import tools, hooks

__all__ = ["mcp", "http_client"]
