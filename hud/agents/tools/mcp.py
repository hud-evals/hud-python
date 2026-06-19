"""MCPTool: capability base for tools that pipe one upstream MCP tool through an ``MCPClient``.

``ToolAgent`` enumerates ``client.list_tools()`` after the MCP handshake and
constructs one instance of every ``MCPTool`` subclass in its catalog per
discovered upstream tool. ``provider_name`` is the upstream name; ``execute``
forwards straight through ``client.call_tool``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from hud.agents.tools.base import AgentTool, AgentToolSpec
from hud.capabilities import MCPClient

if TYPE_CHECKING:
    import mcp.types as mcp_types

    from hud.types import MCPToolResult


class MCPTool(AgentTool[MCPClient]):
    """Capability base: tool that proxies one upstream MCP tool over ``MCPClient``."""

    client_type = MCPClient

    def __init__(
        self,
        *,
        spec: AgentToolSpec,
        client: MCPClient,
        mcp_tool: mcp_types.Tool,
    ) -> None:
        super().__init__(spec=spec, client=client)
        self.mcp_tool = mcp_tool

    @property
    def provider_name(self) -> str:
        return self.mcp_tool.name

    async def execute(self, arguments: dict[str, Any]) -> MCPToolResult:
        return await self.client.call_tool(self.mcp_tool.name, arguments)


__all__ = ["MCPTool"]
