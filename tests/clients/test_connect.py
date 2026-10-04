"""The control channel's client side: ``Runtime.connect`` and ``HudClient`` against a
scripted peer that answers each connection and records every frame it receives.
"""

from __future__ import annotations

import asyncio
import json
from typing import ClassVar, Self
from urllib.parse import urlsplit

import pytest

from hud.capabilities import (
    Capability,
    CapabilityClient,
    find_capability,
    open_client,
    register_client,
)
from hud.runtime import Runtime
from tests.clients.conftest import Action, control_peer

HELLO = {"session_id": "s-1", "env": {"name": "stub", "version": "1.0"}, "bindings": []}
BINDINGS = [
    {"name": "shell", "protocol": "ssh/2", "url": "ssh://agent@environment:22"},
    {"name": "screen-0", "protocol": "rfb/3.8", "url": "rfb://environment:5900"},
    {"name": "screen-1", "protocol": "rfb/3.8", "url": "rfb://environment:5901"},
    {"name": "custom", "protocol": "custom/1", "url": "tcp://environment:7000"},
]


@register_client
class EchoClient(CapabilityClient):
    """A client for a protocol the SDK does not know, registered the way a harness package would."""

    protocol: ClassVar[str] = "echo/1"

    def __init__(self, capability: Capability) -> None:
        self.capability = capability
        self.closed = False

    @classmethod
    async def connect(cls, cap: Capability) -> Self:
        return cls(cap)

    async def close(self) -> None:
        self.closed = True


@pytest.mark.parametrize("immediately", [True, False], ids=["on-accept", "during-hello"])
async def test_connect_retries_an_env_that_hangs_up_until_it_answers_hello(
    immediately: bool,
) -> None:
    not_ready = [Action("hang_up", immediately=immediately)]

    async with (
        control_peer(not_ready, not_ready, [Action("answer", result=HELLO)]) as peer,
        Runtime(peer.url, ready_timeout=10).connect() as client,
    ):
        assert client.session_id == "s-1"

    assert peer.accepted == 3


async def test_connect_retries_a_port_that_refuses_until_the_env_starts_listening() -> None:
    probe = await asyncio.start_server(lambda reader, writer: None, "127.0.0.1", 0)
    port = probe.sockets[0].getsockname()[1]
    probe.close()
    await probe.wait_closed()

    async def hello(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        request = json.loads(await reader.readline())
        writer.write(
            json.dumps({"jsonrpc": "2.0", "id": request["id"], "result": HELLO}).encode() + b"\n"
        )
        await writer.drain()
        await reader.read()

    async def listen_later() -> asyncio.Server:
        await asyncio.sleep(1)
        return await asyncio.start_server(hello, "127.0.0.1", port)

    later = asyncio.create_task(listen_later())
    async with Runtime(f"tcp://127.0.0.1:{port}", ready_timeout=10).connect() as client:
        session_id = client.session_id
    server = await later
    server.close()

    assert session_id == "s-1"


async def test_an_env_that_keeps_hanging_up_is_given_up_on_at_the_runtimes_ready_timeout() -> None:
    loop = asyncio.get_running_loop()

    async with control_peer([Action("hang_up")]) as peer:
        started = loop.time()
        with pytest.raises(EOFError, match="env closed connection during 'hello'"):
            async with Runtime(peer.url, ready_timeout=0.3).connect():
                pass

    assert loop.time() - started < 5


@pytest.mark.parametrize(
    ("ref", "scheme", "reached"),
    [
        ("ssh", "ssh", b"shell"),
        ("ssh/2", "ssh", b"shell"),
        ("screen-1", "rfb", b"screen-1"),
    ],
    ids=["by-protocol", "by-versioned-protocol", "by-name-among-same-protocol"],
)
async def test_a_binding_resolves_to_a_local_tunnel_by_name_or_protocol(
    ref: str, scheme: str, reached: bytes
) -> None:
    async def announcer(name: str) -> asyncio.Server:
        async def announce(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
            writer.write(name.encode())
            await writer.drain()
            writer.close()

        return await asyncio.start_server(announce, "127.0.0.1", 0)

    upstreams = {binding["name"]: await announcer(binding["name"]) for binding in BINDINGS}
    try:
        async with (
            control_peer(
                [Action("answer", result={**HELLO, "bindings": BINDINGS})],
                upstreams={
                    name: ("127.0.0.1", server.sockets[0].getsockname()[1])
                    for name, server in upstreams.items()
                },
            ) as peer,
            Runtime(peer.url).connect() as client,
        ):
            url = urlsplit(find_capability(client.capabilities, ref).url)
            reader, writer = await asyncio.open_connection(url.hostname, url.port)
            received = await asyncio.wait_for(reader.read(), 10)
            writer.close()
    finally:
        for server in upstreams.values():
            server.close()

    assert (url.scheme, url.hostname, received) == (scheme, "127.0.0.1", reached)


async def test_a_stream_the_env_refuses_ends_the_local_connection() -> None:
    async with (
        control_peer([Action("answer", result={**HELLO, "bindings": BINDINGS[:1]})]) as peer,
        Runtime(peer.url).connect() as client,
    ):
        local = urlsplit(find_capability(client.capabilities, "shell").url)
        reader, writer = await asyncio.open_connection(local.hostname, local.port)
        received = await asyncio.wait_for(reader.read(), 10)
        writer.close()

    assert received == b""


async def test_a_registered_client_class_opens_its_protocols_capabilities() -> None:
    echo = Capability(name="echo", protocol="echo/1", url="tcp://environment:7100")
    async with open_client(echo) as opened:
        assert isinstance(opened, EchoClient)
        assert not opened.closed

    assert (opened.capability.name, opened.closed) == ("echo", True)


def test_a_second_client_class_for_a_registered_protocol_is_refused() -> None:
    class OtherEchoClient(EchoClient):
        pass

    with pytest.raises(ValueError, match=r"'echo/1' is already registered to .*\.EchoClient"):
        register_client(OtherEchoClient)
    assert register_client(EchoClient) is EchoClient
