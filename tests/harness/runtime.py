"""The HUD runtime service's tunnel, faked: each WebSocket is spliced to a local control channel."""

from __future__ import annotations

import asyncio
import contextlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from starlette.websockets import WebSocket

    from .services import FakeServices

SESSIONS = "/runtime/sessions"
SESSION = "/runtime/sessions/{id}"
TUNNEL = "/runtime/tunnels/{id}"


def relay(port: int) -> Callable[[WebSocket, dict[str, str]], Awaitable[None]]:
    """A tunnel endpoint splicing each WebSocket to the control channel on ``port``."""

    async def handler(websocket: WebSocket, params: dict[str, str]) -> None:
        del params
        await websocket.accept()
        reader, writer = await asyncio.open_connection("127.0.0.1", port)

        async def upstream() -> None:
            while (message := await websocket.receive())["type"] != "websocket.disconnect":
                writer.write(message.get("bytes") or message.get("text", "").encode())
                await writer.drain()

        async def downstream() -> None:
            while data := await reader.read(65536):
                await websocket.send_bytes(data)

        tasks = [asyncio.create_task(upstream()), asyncio.create_task(downstream())]
        await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        writer.close()
        with contextlib.suppress(Exception):
            await websocket.close()

    return handler


def host_on_runtime(services: FakeServices, port: int, session: str = "sess-1") -> None:
    """Let the fake runtime service lease ``session`` and tunnel it to the env on ``port``."""
    services.route("runtime", "POST", SESSIONS, json={"id": session})
    services.route("runtime", "DELETE", SESSION, json={})
    services.websocket("runtime", TUNNEL, relay(port))
