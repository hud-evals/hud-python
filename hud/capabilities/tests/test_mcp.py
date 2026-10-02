"""MCPClient against a real FastMCP server over streamable HTTP."""

from __future__ import annotations

import asyncio
import contextlib
import socket
from typing import TYPE_CHECKING

import fastmcp
import pytest
import uvicorn

from hud.capabilities import Capability
from hud.capabilities.mcp import MCPClient, get_mcp_trace_id
from hud.telemetry.context import set_trace_context
from hud.types import TextContent

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


@contextlib.asynccontextmanager
async def _serve(server: fastmcp.FastMCP) -> AsyncIterator[int]:
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    config = uvicorn.Config(server.http_app(), log_level="warning", lifespan="on")
    http = uvicorn.Server(config)
    serving = asyncio.create_task(http.serve(sockets=[sock]))
    while not http.started:  # noqa: ASYNC110 - uvicorn exposes startup only as a flag
        await asyncio.sleep(0.01)
    try:
        yield port
    finally:
        http.should_exit = True
        await serving


@contextlib.asynccontextmanager
async def _cut_tool_calls(upstream: int) -> AsyncIterator[int]:
    """Forward HTTP to ``upstream``, dropping the connection once a tool call is sent."""

    async def relay(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        up_reader, up_writer = await asyncio.open_connection("127.0.0.1", upstream)

        async def pump(src: asyncio.StreamReader, dst: asyncio.StreamWriter, *, cut: bool) -> None:
            try:
                while data := await src.read(65536):
                    if cut and b'"tools/call"' in data:
                        return
                    dst.write(data)
                    await dst.drain()
            finally:
                writer.close()
                up_writer.close()

        await asyncio.gather(
            pump(reader, up_writer, cut=True),
            pump(up_reader, writer, cut=False),
            return_exceptions=True,
        )

    proxy = await asyncio.start_server(relay, "127.0.0.1", 0)
    try:
        yield proxy.sockets[0].getsockname()[1]
    finally:
        proxy.close()


def _server() -> fastmcp.FastMCP:
    server = fastmcp.FastMCP("probe")

    @server.tool()
    def whoami() -> str:
        return get_mcp_trace_id() or ""

    return server


async def test_tool_calls_return_hud_results_carrying_the_trace_id() -> None:
    async with _serve(_server()) as port:
        client = await MCPClient.connect(Capability.mcp(url=f"http://127.0.0.1:{port}/mcp"))
        try:
            with set_trace_context("child-trace"):
                result = await client.call_tool("whoami", {})
        finally:
            await client.close()

    assert result.isError is False
    assert result.content == [TextContent(type="text", text="child-trace")]


async def test_a_dropped_connection_fails_the_tool_call_instead_of_hanging() -> None:
    async with _serve(_server()) as port, _cut_tool_calls(port) as proxy:
        client = await MCPClient.connect(Capability.mcp(url=f"http://127.0.0.1:{proxy}/mcp"))
        try:
            with pytest.raises(Exception) as failure:
                await asyncio.wait_for(client.call_tool("whoami", {}), timeout=30)
        finally:
            with contextlib.suppress(Exception):
                await client.close()

    assert not isinstance(failure.value, TimeoutError)
