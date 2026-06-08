"""MCPClient — fastmcp.Client wrapper that fits the CapabilityClient contract.

Establishes an MCP session (initialize handshake) on ``connect``. Exposes
``list_tools`` for post-handshake discovery and ``call_tool`` for invocation,
both speaking raw MCP types so they slot into ``MCPTool``.
"""

from __future__ import annotations

from contextlib import AsyncExitStack
from typing import TYPE_CHECKING, Any, ClassVar, Self

import fastmcp
from fastmcp.client.auth import BearerAuth

from .base import Capability, CapabilityClient

if TYPE_CHECKING:
    import mcp.types as mcp_types

    from hud.types import MCPToolResult


class MCPClient(CapabilityClient):
    """Live MCP session opened over the URL in a ``mcp/2025-11-25`` capability."""

    protocol: ClassVar[str] = "mcp/2025-11-25"

    def __init__(
        self,
        capability: Capability,
        client: fastmcp.Client[Any],
        exit_stack: AsyncExitStack,
    ) -> None:
        self.capability = capability
        self._client = client
        self._exit_stack = exit_stack

    @classmethod
    async def connect(cls, cap: Capability) -> Self:
        from hud.patches import apply_all_patches

        apply_all_patches()

        token = cap.params.get("auth_token")
        client: fastmcp.Client[Any] = fastmcp.Client(
            cap.url,
            auth=BearerAuth(token) if token else None,
        )
        stack = AsyncExitStack()
        await stack.enter_async_context(client)
        return cls(cap, client, stack)

    async def list_tools(self) -> list[mcp_types.Tool]:
        """Tools advertised by the MCP server (initialize already complete)."""
        return await self._client.list_tools()

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> MCPToolResult:
        """Invoke a tool, returning the raw MCP ``CallToolResult``.

        FastMCP and mcp-python use slightly different result shapes; normalize the
        alternate field names (``is_error`` / ``structured_content``) and a missing
        ``content`` so callers always get a canonical ``CallToolResult``.
        """
        from hud.types import MCPToolResult as _Result

        raw = await self._client.call_tool_mcp(name=name, arguments=arguments)
        data = raw.model_dump()
        if "isError" not in data and "is_error" in data:
            data["isError"] = data.pop("is_error")
        if "structuredContent" not in data and "structured_content" in data:
            data["structuredContent"] = data.pop("structured_content")
        data.setdefault("content", [])
        return _Result.model_validate(data)

    async def close(self) -> None:
        await self._exit_stack.aclose()


__all__ = ["MCPClient"]
