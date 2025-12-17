"""Connection connectors - methods for connecting to various sources."""

from hud.environment.connectors.local import LocalConnectorMixin
from hud.environment.connectors.openai import OpenAIConnectorMixin
from hud.environment.connectors.remote import RemoteConnectorMixin

__all__ = ["ConnectorsMixin"]


class ConnectorsMixin(
    RemoteConnectorMixin,
    LocalConnectorMixin,
    OpenAIConnectorMixin,
):
    """Combined connector mixin providing all connection methods.

    Remote connections:
        connect_hub(slug) - HUD Hub environment
        connect_url(url) - MCP server via URL
        connect_openapi(spec) - Mount OpenAPI spec as MCP server

    Local connections (in-process):
        connect_image(image) - Docker image via stdio
        connect_fastapi(app) - Mount FastAPI app as MCP server
        connect_server(server) - Mount MCPServer/FastMCP directly

    MCP config:
        connect_mcp(config) - Single mcp_config server (auto-detects local/remote)
        connect_mcp_config(mcp_config) - Multiple mcp_config servers

    Framework imports:
        connect_function_tools(tools) - Import OpenAI Agents SDK FunctionTools
    """
