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
from fastmcp.client.transports import SSETransport, StreamableHttpTransport

from hud.telemetry.context import get_current_trace_id

from .base import Capability, CapabilityClient

if TYPE_CHECKING:
    import mcp.types as mcp_types

    from hud.types import MCPToolResult

_TRACE_ID_META_KEY = "hud/trace-id"


def get_mcp_trace_id() -> str | None:
    """Get the trace ID propagated with the active inbound MCP request."""
    from fastmcp.server.dependencies import get_context, get_http_headers

    try:
        context = get_context()
    except RuntimeError:
        return None
    request_context = context.request_context
    meta = request_context.meta if request_context else None
    extra = (meta.model_extra or {}) if meta else {}
    trace_id = extra.get(_TRACE_ID_META_KEY)
    if trace_id is None:
        trace_id = get_http_headers().get("trace-id")
    if trace_id is not None and not isinstance(trace_id, str):
        raise TypeError(f"{_TRACE_ID_META_KEY} must be a string")
    return trace_id


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
        token = cap.params.get("auth_token")
        auth = BearerAuth(token) if token else None
        transport = cap.params.get("transport")
        if transport == "sse":
            client: fastmcp.Client[Any] = fastmcp.Client(SSETransport(cap.url, auth=auth))
        elif transport == "streamable-http":
            client = fastmcp.Client(StreamableHttpTransport(cap.url, auth=auth))
        else:
            client = fastmcp.Client(cap.url, auth=auth)
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

        trace_id = get_current_trace_id()
        meta = {_TRACE_ID_META_KEY: trace_id} if trace_id is not None else None
        raw = await self._client.call_tool_mcp(name=name, arguments=arguments, meta=meta)
        data = raw.model_dump()
        if "isError" not in data and "is_error" in data:
            data["isError"] = data.pop("is_error")
        if "structuredContent" not in data and "structured_content" in data:
            data["structuredContent"] = data.pop("structured_content")
        data.setdefault("content", [])
        return _Result.model_validate(data)

    async def close(self) -> None:
        await self._exit_stack.aclose()


__all__ = ["MCPClient", "get_mcp_trace_id"]
