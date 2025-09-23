"""Controller package - registers hooks and tools."""

import sys
import os
import httpx
import logging
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

mcp = MCPServer()

ENV_SERVER_PORT = os.getenv("ENV_SERVER_PORT", 8005)
http_client = httpx.AsyncClient(base_url=f"http://localhost:{ENV_SERVER_PORT}", timeout=10.0)

# Import tools and hooks to register them with the server
from . import tools, hooks

__all__ = ["mcp", "http_client"]
