import sys
import os
import logging
from hud.server import MCPServer
from tau2.registry import registry

logger = logging.getLogger(__name__)

mcp = MCPServer(name="tau2-bench")
from .setup import setup
from .evaluate import evaluate
from .tools._wrapper import wrap_all_tools

# Global conversation tool reference (needed for setup to bootstrap conversations)
_conversation_tool = None


def get_conversation_tool():
    """Get the globally registered conversation tool."""
    return _conversation_tool


@mcp.initialize
async def init():
    """Initialize tau2-bench environment and dynamically load domain tools."""
    global _conversation_tool

    logger.info("Initializing tau2-bench environment")

    # Mount hubs
    mcp.mount(setup)
    mcp.mount(evaluate)

    # Register conversation tool for multi-turn mode
    from .tools.conversation import create_conversation_tool

    _conversation_tool = create_conversation_tool()
    mcp.add_tool(_conversation_tool)

    # Load domain-specific tools
    domain = os.getenv("DOMAIN", "airline")
    logger.info(f"Loading domain: {domain}")

    env_constructor = registry.get_env_constructor(domain)
    environment = env_constructor(solo_mode=False)
    wrapped_tools = wrap_all_tools(environment.tools)

    # Register all domain tools
    for tool_name, tool_instance in wrapped_tools.items():
        mcp.add_tool(tool_instance.mcp)

    logger.info(f"Initialized with {len(wrapped_tools)} domain tools + send_message")


if __name__ == "__main__":
    mcp.run(transport="stdio")
