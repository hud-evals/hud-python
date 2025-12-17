"""
HUD Environment - A unified abstraction for MCP environments.

The Environment class is a server that you can also use as a client.
It subclasses MCPServer to get server capabilities (@env.tool, serve())
and composes FastMCP Client instances for remote connections.

Usage:
    from hud.environment import Environment

    # Create and connect
    env = Environment("my-env").connect_hub("browser", prefix="web")

    async with env:
        # Get tools in any format
        openai_tools = env.as_openai_chat_tools()
        claude_tools = env.as_claude_tools()

        # Call tools with any format - auto-parses and returns matching format
        result = await env.call_tool("web_navigate", url="https://google.com")

        # Framework integrations (requires external deps)
        agent_tools = env.as_openai_agent_tools()   # needs openai-agents
        lc_tools = env.as_langchain_tools()         # needs langchain-core
"""

from hud.environment.connection import ConnectionConfig, ConnectionType, Connector
from hud.environment.environment import Environment
from hud.environment.mock import MockMixin, generate_mock_value
from hud.environment.router import ConflictResolution, ToolRouter
from hud.environment.scenarios import ScenarioMixin
from hud.environment.types import EnvConfig
from hud.environment.utils import ToolFormat, format_result, parse_tool_call, parse_tool_calls

__all__ = [
    "ConflictResolution",
    "ConnectionConfig",
    "ConnectionType",
    "Connector",
    "EnvConfig",
    "Environment",
    "MockMixin",
    "ScenarioMixin",
    "ToolFormat",
    "ToolRouter",
    "format_result",
    "generate_mock_value",
    "parse_tool_call",
    "parse_tool_calls",
]
