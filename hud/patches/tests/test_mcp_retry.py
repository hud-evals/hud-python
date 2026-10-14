"""Streamable-HTTP MCP calls ride out connections the peer drops before replying.

A tunnel or proxy that hangs up on a request without answering surfaces in httpx as
``RemoteProtocolError``. The patched transport retries it with backoff until
``settings.client_timeout`` is spent, then answers the pending call with a transport error
instead of leaving it hanging. These tests drive a real fastmcp server behind a TCP proxy
that can hang up on new connections.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import pytest
from fastmcp import FastMCP
from mcp import McpError
from mcp.types import TextContent

from hud.capabilities import Capability
from hud.capabilities.mcp import MCPClient
from hud.settings import settings

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


class HangupProxy:
    """TCP proxy to a fastmcp server; while ``hangups`` is positive it drops new POST connections.

    A dropped connection reads the request and closes without writing a response.
    """

    def __init__(self, backend_port: int) -> None:
        self.backend_port = backend_port
        self.hangups = 0
        self.server: asyncio.Server | None = None
        self._open: set[asyncio.StreamWriter] = set()

    async def start(self) -> int:
        self.server = await asyncio.start_server(self._handle, "127.0.0.1", 0)
        return self.server.sockets[0].getsockname()[1]

    def drop_open_connections(self) -> None:
        for writer in list(self._open):
            writer.close()

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        request = await reader.read(65536)
        if self.hangups > 0 and request.startswith(b"POST"):
            self.hangups -= 1
            writer.close()
            return
        backend_reader, backend_writer = await asyncio.open_connection(
            "127.0.0.1", self.backend_port
        )
        backend_writer.write(request)
        self._open.add(writer)

        async def pump(source: asyncio.StreamReader, sink: asyncio.StreamWriter) -> None:
            try:
                while data := await source.read(65536):
                    sink.write(data)
                    await sink.drain()
            except OSError:
                pass
            finally:
                sink.close()

        await asyncio.gather(pump(reader, backend_writer), pump(backend_reader, writer))
        self._open.discard(writer)


@pytest.fixture
async def proxied_tool_server() -> AsyncIterator[tuple[HangupProxy, str, list[int]]]:
    executions: list[int] = []
    server = FastMCP("retry-test")

    @server.tool
    def work() -> str:
        executions.append(1)
        return "done"

    backend = await asyncio.start_server(lambda r, w: None, "127.0.0.1", 0)
    backend_port = backend.sockets[0].getsockname()[1]
    backend.close()
    await backend.wait_closed()
    serving = asyncio.create_task(
        server.run_async(transport="http", host="127.0.0.1", port=backend_port, show_banner=False)
    )
    for _ in range(100):
        try:
            _, probe = await asyncio.open_connection("127.0.0.1", backend_port)
        except OSError:
            await asyncio.sleep(0.05)
        else:
            probe.close()
            break
    proxy = HangupProxy(backend_port)
    proxy_port = await proxy.start()
    try:
        yield proxy, f"http://127.0.0.1:{proxy_port}/mcp", executions
    finally:
        assert proxy.server is not None
        proxy.server.close()
        serving.cancel()
        await asyncio.gather(serving, return_exceptions=True)


async def test_call_survives_connections_hung_up_before_a_reply(
    proxied_tool_server: tuple[HangupProxy, str, list[int]],
) -> None:
    proxy, url, executions = proxied_tool_server
    client = await MCPClient.connect(Capability.mcp(name="tools", url=url))
    try:
        proxy.drop_open_connections()
        proxy.hangups = 2

        result = await client.call_tool("work", {})

        content = result.content[0]
        assert isinstance(content, TextContent)
        assert content.text == "done"
        assert executions == [1]
    finally:
        await client.close()


async def test_call_fails_with_a_transport_error_once_the_retry_budget_is_spent(
    proxied_tool_server: tuple[HangupProxy, str, list[int]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    proxy, url, executions = proxied_tool_server
    monkeypatch.setattr(settings, "client_timeout", 1)
    client = await MCPClient.connect(Capability.mcp(name="tools", url=url))
    try:
        proxy.drop_open_connections()
        proxy.hangups = 10_000

        with pytest.raises(McpError, match="Transport error: RemoteProtocolError"):
            await client.call_tool("work", {})

        assert executions == []
    finally:
        await client.close()
