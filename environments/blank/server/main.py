import sys
import logging
from hud.server import MCPServer
from server.shared import http_client

# Configure logging to stderr
logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="[%(levelname)s] %(asctime)s | %(name)s | %(message)s",
    force=True,
)
for logger_name in ["httpx", "httpcore"]:
    logging.getLogger(logger_name).setLevel(logging.WARNING)

# Create main MCP server
mcp = MCPServer(name="blank-environment")

# Include routers
from server.tools import router as tools_router

mcp.include_router(tools_router)


# Lifecycle hooks
@mcp.initialize
async def init():
    """Check if the environment is healthy"""
    if http_client:
        await http_client.get("/health")
    else:
        raise ValueError("http_client is not set")


@mcp.shutdown
async def cleanup():
    """Close the HTTP client"""
    if http_client:
        await http_client.aclose()


if __name__ == "__main__":
    mcp.run(transport="stdio")
