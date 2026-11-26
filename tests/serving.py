"""Serve an environment's control channel from the test process, for tests of the wire."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from hud.environment.server import bind
from hud.runtime import Runtime

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from hud import Environment


@asynccontextmanager
async def serving(env: Environment) -> AsyncGenerator[Runtime]:
    """Serve ``env``'s control channel on loopback and yield its runtime."""
    await env.start()
    try:
        async with bind(env) as server:
            host, port = server.sockets[0].getsockname()[:2]
            yield Runtime(f"tcp://{host}:{port}")
    finally:
        await env.stop()
