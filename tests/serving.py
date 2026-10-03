"""Serve an environment's control channel from the test process, for tests of the wire."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from hud.clients import HudClient
from hud.environment.server import bind
from hud.runtime import Runtime

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Sequence

    from hud import Environment
    from hud.capabilities import Connection
    from hud.environment.egress import WorkspaceRoute


@asynccontextmanager
async def serving(env: Environment) -> AsyncIterator[Runtime]:
    """Serve ``env``'s control channel on loopback and yield its runtime."""
    await env.start()
    try:
        async with bind(env) as server:
            host, port = server.sockets[0].getsockname()[:2]
            yield Runtime(f"tcp://{host}:{port}")
    finally:
        await env.stop()


@asynccontextmanager
async def hud_client(
    runtime: Runtime,
    *,
    session_id: str | None = None,
    lease: float | None = None,
    workspace_routes: Sequence[WorkspaceRoute] = (),
    connections: Sequence[Connection] = (),
) -> AsyncIterator[HudClient]:
    """A control-channel session on ``runtime``, as the ``HudClient`` it is."""
    async with runtime.connect(
        session_id=session_id,
        lease=lease,
        workspace_routes=workspace_routes,
        connections=connections,
    ) as client:
        assert isinstance(client, HudClient)
        yield client


@asynccontextmanager
async def served(env: Environment) -> AsyncIterator[HudClient]:
    """Serve ``env`` on loopback and yield a client connected to it."""
    async with serving(env) as runtime, hud_client(runtime) as client:
        yield client
