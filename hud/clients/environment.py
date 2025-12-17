"""Environment-based client adapter for agents."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import mcp.types as types

from hud.types import MCPToolCall, MCPToolResult

if TYPE_CHECKING:
    from hud.environment import Environment
    from hud.eval.context import EvalContext

__all__ = ["EnvironmentClient"]


class EnvironmentClient:
    """Adapter wrapping Environment/EvalContext as AgentMCPClient."""

    def __init__(self, env: Environment | EvalContext) -> None:
        self._env = env
        self._initialized = False

    @property
    def mcp_config(self) -> dict[str, dict[str, Any]]:
        return {}

    @property
    def is_connected(self) -> bool:
        return self._initialized

    async def initialize(self, mcp_config: dict[str, dict[str, Any]] | None = None) -> None:
        if not self._initialized:
            await self._env.list_tools()
            self._initialized = True

    async def list_tools(self) -> list[types.Tool]:
        return await self._env.list_tools()

    async def call_tool(self, tool_call: MCPToolCall) -> MCPToolResult:
        result = await self._env.call_tool(tool_call.name, **(tool_call.arguments or {}))
        if isinstance(result, MCPToolResult):
            return result
        return MCPToolResult(
            content=[types.TextContent(type="text", text=str(result))],
            isError=False,
        )

    async def shutdown(self) -> None:
        self._initialized = False
