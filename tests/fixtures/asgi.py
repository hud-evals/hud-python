"""Serve an ASGI app, or an MCP server, on a loopback port inside the test's event loop."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Literal

import uvicorn

from hud.capabilities import Capability
from tests.conftest import eventually

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from fastmcp import FastMCP
    from starlette.types import ASGIApp


@asynccontextmanager
async def serve_asgi(app: ASGIApp) -> AsyncIterator[int]:
    """Serve ``app`` on ``127.0.0.1`` and yield its port; stop it when the block ends."""
    server = uvicorn.Server(
        uvicorn.Config(
            app, host="127.0.0.1", port=0, log_level="error", lifespan="on", ws="websockets-sansio"
        )
    )
    serving = asyncio.create_task(server.serve())
    await eventually(lambda: server.started or serving.done())
    if serving.done():
        serving.result()
    try:
        yield server.servers[0].sockets[0].getsockname()[1]
    finally:
        server.should_exit = True
        await serving


@asynccontextmanager
async def mcp_server(
    server: FastMCP,
    *,
    name: str = "tools",
    transport: Literal["streamable-http", "sse"] = "streamable-http",
) -> AsyncIterator[Capability]:
    """Serve ``server`` over HTTP on loopback; yield the capability that reaches it."""
    path = "/mcp" if transport == "streamable-http" else "/sse"
    async with serve_asgi(server.http_app(path=path, transport=transport)) as port:
        yield Capability.mcp(name=name, url=f"http://127.0.0.1:{port}{path}", transport=transport)
