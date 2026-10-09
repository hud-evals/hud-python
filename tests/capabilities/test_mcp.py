"""``MCPClient`` against a FastMCP server over each HTTP transport."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import pytest
from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from mcp.shared.exceptions import McpError

from hud.capabilities import Capability, MCPClient
from tests.harness import serve_asgi

if TYPE_CHECKING:
    from starlette.types import ASGIApp, Receive, Scope, Send


def tools() -> FastMCP:
    server = FastMCP("tools")

    @server.tool
    def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    @server.tool
    def refuse() -> str:
        """Always fail."""
        raise ToolError("not today")

    return server


@pytest.mark.parametrize(("transport", "path"), [("streamable-http", "/mcp"), ("sse", "/sse")])
async def test_tool_calls_return_canonical_results_over_each_transport(
    transport: Literal["streamable-http", "sse"], path: str
) -> None:
    app = tools().http_app(transport=transport, path=path)
    async with serve_asgi(app) as port:
        client = await MCPClient.connect(
            Capability.mcp(url=f"http://127.0.0.1:{port}{path}", transport=transport)
        )
        try:
            listed = sorted(tool.name for tool in await client.list_tools())
            added = await client.call_tool("add", {"a": 2, "b": 3})
            refused = await client.call_tool("refuse", {})
        finally:
            await client.close()

    assert listed == ["add", "refuse"]
    assert (added.isError, added.structuredContent) == (False, {"result": 5})
    assert [block.model_dump(include={"type", "text"}) for block in added.content] == [
        {"type": "text", "text": "5"}
    ]
    assert refused.isError is True
    assert [block.model_dump(include={"type", "text"}) for block in refused.content] == [
        {"type": "text", "text": "not today"}
    ]


def bearer_only(app: ASGIApp, token: str, seen: list[str]) -> ASGIApp:
    """``app`` behind a gate that answers 401 to any HTTP request without ``token``."""

    async def gated(scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http":
            authorization = dict(scope["headers"]).get(b"authorization", b"").decode()
            seen.append(authorization)
            if authorization != f"Bearer {token}":
                await send({"type": "http.response.start", "status": 401, "headers": []})
                await send({"type": "http.response.body", "body": b""})
                return
        await app(scope, receive, send)

    return gated


async def test_an_auth_token_rides_every_request_and_is_needed_where_the_server_asks() -> None:
    seen: list[str] = []
    app = bearer_only(tools().http_app(path="/mcp"), "s3cret", seen)

    async with serve_asgi(app) as port:
        url = f"http://127.0.0.1:{port}/mcp"
        client = await MCPClient.connect(Capability.mcp(url=url, auth_token="s3cret"))
        try:
            added = await client.call_tool("add", {"a": 2, "b": 3})
        finally:
            await client.close()
        authorized = set(seen)
        with pytest.raises(McpError, match="HTTPStatusError"):
            await MCPClient.connect(Capability.mcp(url=url))

    assert added.structuredContent == {"result": 5}
    assert authorized == {"Bearer s3cret"}
