"""``MCPClient`` against a FastMCP server over each HTTP transport."""

from __future__ import annotations

from typing import Literal

import pytest
from fastmcp import FastMCP
from fastmcp.exceptions import ToolError

from hud.capabilities import Capability, MCPClient
from tests.harness import serve_asgi


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
