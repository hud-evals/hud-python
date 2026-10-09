"""Serve an ASGI app on a loopback port inside the test's event loop."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

import uvicorn

from .scenario import eventually

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

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
