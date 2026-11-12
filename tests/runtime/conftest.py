"""The HUD runtime service, hosting an environment this process serves, and a check that a
process has exited."""

from __future__ import annotations

import asyncio
import contextlib
import os
import subprocess
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from starlette.websockets import WebSocket

    from tests.fixtures.services import FakeServices

SESSIONS = "/runtime/sessions"
SESSION = "/runtime/sessions/{id}"
TUNNEL = "/runtime/tunnels/{id}"


def tunnel(port: int) -> Callable[[WebSocket, dict[str, str]], Awaitable[None]]:
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


def host_on_runtime(services: FakeServices, url: str) -> None:
    """Let the fake runtime service lease ``sess-1`` and tunnel it to the env served at
    ``url``."""
    services.route("runtime", "POST", SESSIONS, json={"id": "sess-1"})
    services.route("runtime", "DELETE", SESSION, json={})
    services.websocket("runtime", TUNNEL, tunnel(int(url.rsplit(":", 1)[1])))


def gone(pid: int) -> bool:
    """Whether process ``pid`` has exited (an unreaped zombie counts as exited)."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    status = subprocess.run(
        ["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True, check=False
    )
    return status.stdout.strip().startswith("Z")
