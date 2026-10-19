"""Capability tunneling: environment daemons reached through the control port.

A capability address belongs to the environment's network. The manifest keeps
that address for processes running there, while a client binding points at a
local forwarder. Each connection to the forwarder becomes one connection to the
control port, opened with a ``tunnel.open`` preface frame and then spliced raw to
the daemon. These scenarios front a TCP echo daemon with a served env.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit

import pytest

from hud.capabilities import Capability
from hud.environment import Environment
from hud.eval import LocalRuntime, Task
from tests.fixtures.environments import served
from tests.support import FRAME_LIMIT, encode, wire

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


@pytest.fixture
async def echo_port() -> AsyncIterator[int]:
    """A daemon on the environment's network that echoes every byte back."""

    async def echo(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            while data := await reader.read(65536):
                writer.write(data)
                await writer.drain()
        finally:
            writer.close()

    server = await asyncio.start_server(echo, "127.0.0.1", 0)
    yield server.sockets[0].getsockname()[1]
    server.close()
    await server.wait_closed()


def echo_env(port: int) -> Environment:
    return Environment(
        "echo-env",
        capabilities=[Capability(name="echo", protocol="rfb/3.8", url=f"rfb://127.0.0.1:{port}")],
    )


async def round_trip(host: str | None, port: int | None, payload: bytes) -> bytes:
    reader, writer = await asyncio.open_connection(host, port)
    writer.write(payload)
    await writer.drain()
    data = await reader.readexactly(len(payload))
    writer.close()
    await writer.wait_closed()
    return data


async def test_concurrent_tunnel_streams_do_not_interleave(echo_port: int) -> None:
    payloads = [f"stream-{i}".encode() * 100 for i in range(8)]

    async with served(echo_env(echo_port)) as client:
        binding = urlsplit(client.binding("echo").url)
        echoed = await asyncio.gather(
            *(round_trip(binding.hostname, binding.port, payload) for payload in payloads)
        )

    assert echoed == payloads


async def test_closing_the_client_closes_its_forwarders(echo_port: int) -> None:
    async with served(echo_env(echo_port)) as client:
        binding = urlsplit(client.binding("echo").url)

    with pytest.raises(OSError):
        await asyncio.open_connection(binding.hostname, binding.port)


@pytest.mark.parametrize("preface_size", [128, FRAME_LIMIT])
async def test_raw_bytes_sent_with_the_preface_reach_the_daemon(
    echo_port: int, preface_size: int
) -> None:
    preface: dict[str, Any] = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tunnel.open",
        "params": {"capability": "echo"},
        "padding": "",
    }
    preface["padding"] = "x" * (preface_size - (len(encode(preface)) - 1))
    env = echo_env(echo_port)

    async with (
        LocalRuntime(env)(Task(env=env.name, id="tunnel")) as runtime,
        wire(runtime.url) as stream,
    ):
        await stream.write(encode(preface) + b"raw bytes")
        opened = await stream.read()
        echoed = await asyncio.wait_for(stream.reader.readexactly(9), timeout=30)

    assert opened == {"jsonrpc": "2.0", "id": 1, "result": {"capability": "echo"}}
    assert echoed == b"raw bytes"


@pytest.mark.parametrize(
    ("params", "code"),
    [
        pytest.param(
            {"capability": "nope"},
            -32000,
            id="unknown-capability",
        ),
    ],
)
async def test_a_refused_tunnel_gets_one_error_frame_and_is_closed(
    echo_port: int, params: dict[str, Any], code: int
) -> None:
    env = echo_env(echo_port)

    async with (
        LocalRuntime(env)(Task(env=env.name, id="tunnel")) as runtime,
        wire(runtime.url) as stream,
    ):
        refused = await stream.call("tunnel.open", params)
        after = await stream.read()

    assert (refused["id"], refused["error"]["code"]) == (1, code)
    assert after is None


async def test_manifest_keeps_environment_addresses_and_client_bindings_are_forwarded() -> None:
    local = Capability.cdp(name="browser", url="ws://127.0.0.1:9222")
    remote = Capability.ssh(name="box", url="ssh://box.example.com:22", host_pubkey="ssh-ed25519 x")
    env = Environment("mixed-env", capabilities=[local, remote])

    async with served(env) as client:
        assert client.manifest is not None
        assert client.manifest.bindings == [local, remote]
        for declared in (local, remote):
            forwarded = client.binding(declared.name)
            assert forwarded.url != declared.url
            assert urlsplit(forwarded.url).hostname == "127.0.0.1"
            assert (forwarded.name, forwarded.protocol) == (declared.name, declared.protocol)
