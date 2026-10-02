"""MCPClient — fastmcp.Client wrapper that fits the CapabilityClient contract.

Establishes an MCP session (initialize handshake) on ``connect``. ``list_tools``
and ``call_tool`` translate MCP's models into HUD's own :mod:`hud.types`
shapes, so MCP's types stay at this boundary.
"""

from __future__ import annotations

from contextlib import AsyncExitStack
from typing import Any, ClassVar, Self

import fastmcp
import mcp.types as mcp_types
from fastmcp.client.auth import BearerAuth
from fastmcp.client.transports import SSETransport, StreamableHttpTransport

from hud.telemetry.context import get_current_trace_id
from hud.types import Tool, ToolResult

from .base import Capability, CapabilityClient

_TRACE_ID_META_KEY = "hud/trace-id"


def get_mcp_trace_id() -> str | None:
    """Get the trace ID propagated with the active inbound MCP request."""
    from fastmcp.server.dependencies import get_context, get_http_headers

    try:
        context = get_context()
    except RuntimeError:
        return None
    request_context = context.request_context
    meta = (request_context.meta if request_context else None) or {}
    trace_id = meta.get(_TRACE_ID_META_KEY)
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

    async def list_tools(self) -> list[Tool]:
        """Tools advertised by the MCP server (initialize already complete)."""
        tools = await self._client.list_tools()
        return [Tool.model_validate(tool.model_dump(mode="json", by_alias=True)) for tool in tools]

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        """Invoke a tool and return its result as a HUD ``ToolResult``.

        The result is not checked against the tool's declared output schema: a
        server whose structured content drifts from its schema still hands the
        agent the content it returned.
        """
        trace_id = get_current_trace_id()
        meta = {_TRACE_ID_META_KEY: trace_id} if trace_id is not None else None
        raw = await self._client.session.send_request(
            mcp_types.CallToolRequest(
                params=mcp_types.CallToolRequestParams(name=name, arguments=arguments, _meta=meta),
            ),
            mcp_types.CallToolResult,
        )
        return ToolResult.model_validate(raw.model_dump(mode="json", by_alias=True))

    async def close(self) -> None:
        await self._exit_stack.aclose()


__all__ = ["MCPClient", "get_mcp_trace_id"]
