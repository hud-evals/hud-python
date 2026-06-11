"""Capability tunneling: substrate-local daemons reached through the control port.

A capability whose resolved address is loopback lives in the substrate's
network namespace — a container publishes only the control port, so the
client can't dial it directly. Instead the manifest binding the client sees
points at a local forwarder; each connection to it becomes one fresh
connection to the control port, opened with a ``tunnel.open`` preface frame
and spliced raw to the daemon. The preface is transport-level routing — a
connection is a stream or a control session from its first frame, never
upgraded mid-session. These tests drive that path end to end against a
served env fronting a real TCP echo server.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING
from urllib.parse import urlsplit

import pytest

from hud.capabilities import Capability
from hud.environment import Environment
from hud.environment.utils import read_frame, send_frame

from .conftest import served

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


@pytest.fixture
async def echo_port() -> AsyncIterator[int]:
    """A substrate-side TCP daemon: echoes every byte back."""

    async def echo(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            while data := await reader.read(1024):
                writer.write(data)
                await writer.drain()
        finally:
            writer.close()

    server = await asyncio.start_server(echo, "127.0.0.1", 0)
    yield server.sockets[0].getsockname()[1]
    server.close()
    await server.wait_closed()


def _echo_env(port: int) -> Environment:
    cap = Capability(name="echo", protocol="rfb/3.8", url=f"rfb://127.0.0.1:{port}", params={})
    return Environment("echo-env", capabilities=[cap])


async def test_bytes_round_trip_through_the_forwarded_binding(echo_port: int) -> None:
    async with served(_echo_env(echo_port)) as client:
        parts = urlsplit(client.binding("echo").url)
        assert parts.port != echo_port  # the binding points at the forwarder

        reader, writer = await asyncio.open_connection(parts.hostname, parts.port)
        writer.write(b"ping through the tunnel")
        await writer.drain()
        assert await reader.readexactly(23) == b"ping through the tunnel"
        writer.close()
        await writer.wait_closed()


async def test_concurrent_tunnel_streams_do_not_interleave(echo_port: int) -> None:
    async with served(_echo_env(echo_port)) as client:
        parts = urlsplit(client.binding("echo").url)

        async def stream(payload: bytes) -> bytes:
            reader, writer = await asyncio.open_connection(parts.hostname, parts.port)
            writer.write(payload)
            await writer.drain()
            data = await reader.readexactly(len(payload))
            writer.close()
            await writer.wait_closed()
            return data

        payloads = [f"stream-{i}".encode() * 100 for i in range(8)]
        assert await asyncio.gather(*(stream(p) for p in payloads)) == payloads


async def test_tunnel_open_for_an_unknown_capability_returns_an_error_frame(
    echo_port: int,
) -> None:
    async with served(_echo_env(echo_port)) as client:
        assert client._endpoint is not None
        reader, writer = await asyncio.open_connection(*client._endpoint)
        await send_frame(
            writer,
            {"jsonrpc": "2.0", "id": 1, "method": "tunnel.open", "params": {"capability": "nope"}},
        )
        opened = await read_frame(reader)
        assert opened is not None and "error" in opened
        assert "nope" in opened["error"]["message"]
        writer.close()
        await writer.wait_closed()


async def test_tunnel_open_mid_session_is_not_a_method(echo_port: int) -> None:
    """A control session never mutates into a stream: tunnel.open is a preface only."""
    async with served(_echo_env(echo_port)) as client:
        assert client._endpoint is not None
        reader, writer = await asyncio.open_connection(*client._endpoint)
        await send_frame(writer, {"jsonrpc": "2.0", "id": 1, "method": "tasks.list"})
        assert await read_frame(reader) is not None  # a control session is established
        await send_frame(
            writer,
            {"jsonrpc": "2.0", "id": 2, "method": "tunnel.open", "params": {"capability": "echo"}},
        )
        opened = await read_frame(reader)
        assert opened is not None and "error" in opened
        assert opened["error"]["code"] == -32601  # method not found
        writer.close()
        await writer.wait_closed()


async def test_closing_the_client_tears_down_its_forwarders(echo_port: int) -> None:
    async with served(_echo_env(echo_port)) as client:
        parts = urlsplit(client.binding("echo").url)

    with pytest.raises(OSError):
        _, writer = await asyncio.open_connection(parts.hostname, parts.port)
        writer.close()
