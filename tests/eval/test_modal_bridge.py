"""The sandbox-side bridge of ``ModalRuntime``.

WebSocket connections on its listen port reach the environment's TCP port byte for byte,
the environment starts when the runtime says so, and the bridge exits as the environment does.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import socket
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import websockets

from hud.runtime import modal_bridge

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

BRIDGE_SOURCE = Path(modal_bridge.__file__).read_text()


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


@contextlib.asynccontextmanager
async def echo_server() -> AsyncGenerator[int]:
    async def echo(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        while data := await reader.read(65536):
            writer.write(data)
            await writer.drain()
        writer.close()

    server = await asyncio.start_server(echo, "127.0.0.1", 0)
    async with server:
        yield server.sockets[0].getsockname()[1]


async def test_the_bridge_splices_websockets_to_the_environment_and_exits_with_it(
    tmp_path: Path,
) -> None:
    listen = free_port()
    start = tmp_path / "start"
    done = tmp_path / "done"
    async with echo_server() as upstream:
        bridge = await asyncio.create_subprocess_exec(
            sys.executable,
            "-c",
            BRIDGE_SOURCE,
            "--listen",
            str(listen),
            "--upstream",
            str(upstream),
            "--start-file",
            str(start),
            "--",
            sys.executable,
            "-c",
            f"import pathlib, time\nwhile not pathlib.Path({str(done)!r}).exists():\n"
            "    time.sleep(0.1)\nraise SystemExit(7)",
            env={**os.environ, "HOME": str(tmp_path)},
        )
        try:
            for _ in range(50):
                with socket.socket() as probe:
                    if probe.connect_ex(("127.0.0.1", listen)) == 0:
                        break
                await asyncio.sleep(0.1)
            large = os.urandom(3 * 1024 * 1024)
            async with websockets.connect(f"ws://127.0.0.1:{listen}", max_size=None) as first:
                await first.send(b"before start")
                assert await first.recv() == b"before start"
                async with websockets.connect(f"ws://127.0.0.1:{listen}", max_size=None) as second:
                    await second.send(large)
                    received = b""
                    while len(received) < len(large):
                        chunk = await second.recv()
                        received += chunk if isinstance(chunk, bytes) else chunk.encode()
                    assert received == large
                assert await first.ping()
            assert bridge.returncode is None
            start.touch()
            await asyncio.sleep(0.5)
            assert bridge.returncode is None
            done.touch()
            assert await asyncio.wait_for(bridge.wait(), timeout=10) == 7
        finally:
            if bridge.returncode is None:
                bridge.kill()
                await bridge.wait()
