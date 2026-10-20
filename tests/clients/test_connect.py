"""The control channel's client side: ``hud.clients.connect`` and ``HudClient`` against a
scripted peer that answers each connection and records every frame it receives.

Not covered: the 120 s control heartbeat. Its interval is a module constant with no
public knob, and waiting it out has no place in the default lane.
"""

from __future__ import annotations

import asyncio
import json
from urllib.parse import urlsplit

import pytest

from hud.clients import connect
from hud.eval.runtime import Runtime
from tests.clients.conftest import Action, control_peer

HELLO = {"session_id": "s-1", "env": {"name": "stub", "version": "1.0"}, "bindings": []}
BINDINGS = [
    {"name": "shell", "protocol": "ssh/2", "url": "ssh://agent@environment:22"},
    {"name": "screen-0", "protocol": "rfb/3.8", "url": "rfb://environment:5900"},
    {"name": "screen-1", "protocol": "rfb/3.8", "url": "rfb://environment:5901"},
    {"name": "custom", "protocol": "custom/1", "url": "tcp://environment:7000"},
]


@pytest.mark.parametrize("immediately", [True, False], ids=["on-accept", "during-hello"])
async def test_connect_retries_an_env_that_hangs_up_until_it_answers_hello(
    immediately: bool,
) -> None:
    not_ready = [Action("hang_up", immediately=immediately)]

    async with (
        control_peer(not_ready, not_ready, [Action("answer", result=HELLO)]) as peer,
        connect(Runtime(peer.url), ready_timeout=10) as client,
    ):
        assert client.manifest is not None

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
    async with connect(Runtime(f"tcp://127.0.0.1:{port}"), ready_timeout=10) as client:
        manifest = client.manifest
    server = await later
    server.close()

    assert manifest is not None and manifest.session_id == "s-1"


async def test_an_env_that_keeps_hanging_up_is_given_up_on_at_the_runtimes_ready_timeout() -> None:
    loop = asyncio.get_running_loop()

    async with control_peer([Action("hang_up")]) as peer:
        started = loop.time()
        with pytest.raises(EOFError, match="env closed connection during 'hello'"):
            async with connect(Runtime(peer.url, params={"ready_timeout": 0.3}), ready_timeout=60):
                pass

    assert loop.time() - started < 5


@pytest.mark.parametrize(
    ("ref", "scheme", "reached"),
    [
        ("shell", "ssh", b"shell"),
        ("ssh", "ssh", b"shell"),
        ("ssh/2", "ssh", b"shell"),
        ("screen-1", "rfb", b"screen-1"),
    ],
    ids=["by-name", "by-protocol", "by-versioned-protocol", "by-name-among-same-protocol"],
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
            connect(Runtime(peer.url)) as client,
        ):
            url = urlsplit(client.binding(ref).url)
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
        connect(Runtime(peer.url)) as client,
    ):
        local = urlsplit(client.binding("shell").url)
        reader, writer = await asyncio.open_connection(local.hostname, local.port)
        received = await asyncio.wait_for(reader.read(), 10)
        writer.close()

    assert received == b""
