"""Browser server - MCP tools for browser automation."""

import contextlib
import logging
import sys
from hud.server import MCPServer
from hud.tools import HudComputerTool, AnthropicComputerTool, OpenAIComputerTool

# Configure logging
logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="[%(levelname)s] %(asctime)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# MCP server instance
mcp = MCPServer(name="HUD Browser Environment")

from server.shared import ENV_SERVER_URL, http_client, playwright

# Register tools
mcp.tool(playwright)
mcp.tool(HudComputerTool(display_num=1))
mcp.tool(AnthropicComputerTool(display_num=1))
mcp.tool(OpenAIComputerTool(display_num=1))

# Import and register routers
from server.tools import router as tools_router
from server.resources import router as resources_router
from server.setup import router as setup_router
from server.evaluate import router as evaluate_router

# Include regular routers
mcp.include_router(tools_router)
mcp.include_router(resources_router)

# Include hidden routers (tools dispatched through single tool)
mcp.include_router(setup_router, hidden=True)
mcp.include_router(evaluate_router, hidden=True)


async def shutdown_env() -> str:
    """Ask the environment backend to shutdown."""
    with contextlib.redirect_stdout(sys.stderr):
        try:
            resp = await http_client.post("/shutdown")
            if resp.status_code in (200, 204):
                logger.info("Requested environment shutdown")
                return "Environment shutdown requested"
            logger.warning("Environment /shutdown returned %s", resp.status_code)
            return f"Environment shutdown returned {resp.status_code}"
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to call /shutdown: %s", exc)
            return f"Failed to request environment shutdown: {exc}"


@mcp.shutdown
async def on_shutdown() -> None:
    """Graceful controller shutdown: remote then local cleanup."""
    with contextlib.redirect_stdout(sys.stderr):
        try:
            await shutdown_env()
        except Exception as exc:  # noqa: BLE001
            logger.warning("shutdown_env failed: %s", exc)

        # Close Playwright browser/session if present
        try:
            await playwright.close()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Playwright close failed: %s", exc)

        # Close shared HTTP client
        try:
            await http_client.aclose()
        except Exception as exc:  # noqa: BLE001
            logger.warning("HTTP client close failed: %s", exc)


__all__ = ["mcp", "http_client", "ENV_SERVER_URL"]


if __name__ == "__main__":
    mcp.run()
