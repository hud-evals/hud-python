"""``connect()`` readiness: the handshake retries until the env actually serves.

A provisioned substrate can sit behind a proxied port (``docker -p``, a
port-forward) that *accepts* TCP before the env behind it is up — those
connections die with EOF at the handshake. Readiness is therefore
protocol-level: ``connect`` keeps retrying through both refused connects and
handshake EOFs until ``hello`` answers or the deadline passes.
"""

from __future__ import annotations

import asyncio

import pytest

from hud.clients import connect
from hud.environment.utils import read_frame, send_frame
from hud.eval.runtime import Runtime

HELLO_RESULT = {"session_id": "s-1", "env": {"name": "stub", "version": "1.0"}, "bindings": []}


async def test_connect_retries_through_accept_then_eof_until_the_env_serves() -> None:
    attempts = 0

    async def handler(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        nonlocal attempts
        attempts += 1
        if attempts <= 2:
            # The docker-proxy shape: accept, then hang up without serving.
            writer.close()
            return
        try:
            msg = await read_frame(reader)
            assert msg is not None
            await send_frame(writer, {"jsonrpc": "2.0", "id": msg["id"], "result": HELLO_RESULT})
            await read_frame(reader)  # hold the connection until the client closes
        finally:
            # 3.12's Server.wait_closed() waits on every connection; a handler
            # that returns without closing its writer deadlocks teardown.
            writer.close()

    server = await asyncio.start_server(handler, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    try:
        async with connect(Runtime(f"tcp://127.0.0.1:{port}"), ready_timeout=10) as client:
            assert client.manifest is not None
            assert client.manifest.server_info.name == "stub"
    finally:
        server.close()
        await server.wait_closed()

    assert attempts == 3


async def test_connect_gives_up_at_the_deadline_when_the_env_never_serves() -> None:
    async def handler(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        # Read the hello frame, then hang up without answering: guarantees the
        # client sees EOF on the reply (not a racing write reset).
        try:
            await read_frame(reader)
        finally:
            writer.close()

    server = await asyncio.start_server(handler, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    try:
        with pytest.raises(EOFError, match="closed connection during 'hello'"):
            async with connect(Runtime(f"tcp://127.0.0.1:{port}"), ready_timeout=1.2):
                pass
    finally:
        server.close()
        await server.wait_closed()
