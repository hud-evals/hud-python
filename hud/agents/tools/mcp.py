"""MCPTool: capability base for tools that pipe one upstream MCP tool through an ``MCPClient``.

``MCPTool`` expands one provider catalog entry into one provider-facing tool per
upstream MCP tool. ``provider_name`` is the upstream name; ``execute`` forwards
straight through ``client.call_tool``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from hud.agents.tools.base import AgentTool, AgentToolSpec
from hud.capabilities import MCPClient

if TYPE_CHECKING:
    import mcp.types as mcp_types

    from hud.capabilities import CapabilityClient
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

    @classmethod
    async def build(
        cls,
        *,
        model: str,
        connections: dict[str, CapabilityClient],
    ) -> tuple[dict[str, AgentTool[Any]], list[Any]]:
        client = connections.get(cls.client_type.protocol)
        if client is None:
            return {}, []

        mcp_client = cast("MCPClient", client)
        tools: dict[str, AgentTool[Any]] = {}
        params: list[Any] = []
        spec = cls.default_spec(model)
        if spec is None:
            return {}, []
        for mcp_tool in await mcp_client.list_tools():
            tool = cls(spec=spec, client=mcp_client, mcp_tool=mcp_tool)
            tools[tool.provider_name] = tool
            params.append(tool.to_params())
        return tools, params


__all__ = ["MCPTool"]
