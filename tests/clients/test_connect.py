"""The control channel's client side: ``hud.clients.connect`` and ``HudClient`` against a
scripted peer that answers each connection and records every frame it receives.

Not covered: the 120 s control heartbeat. Its interval is a module constant with no
public knob, and waiting it out has no place in the default lane.
"""

from __future__ import annotations

import asyncio
import json
import math
from typing import Any
from urllib.parse import urlsplit

import pytest

from hud.clients import HudClient, HudProtocolError, connect
from hud.environment import WorkspaceRoute
from hud.eval.runtime import Runtime
from tests.clients.conftest import Action, control_peer

HELLO = {"session_id": "s-1", "env": {"name": "stub", "version": "1.0"}, "bindings": []}
BINDINGS = [
    {"name": "shell", "protocol": "ssh/2", "url": "ssh://agent@environment:22"},
    {"name": "screen-0", "protocol": "rfb/3.8", "url": "rfb://environment:5900"},
    {"name": "screen-1", "protocol": "rfb/3.8", "url": "rfb://environment:5901"},
    {"name": "custom", "protocol": "custom/1", "url": "tcp://environment:7000"},
]


async def test_hello_names_the_session_and_the_env_it_reached() -> None:
    async with (
        control_peer([Action("answer", result=HELLO)]) as peer,
        connect(Runtime(peer.url)) as client,
    ):
        manifest = client.manifest

    assert manifest is not None
    assert (manifest.session_id, manifest.server_info.name, manifest.server_info.version) == (
        "s-1",
        "stub",
        "1.0",
    )


async def test_hello_carries_the_host_and_port_of_each_workspace_route() -> None:
    routes = (
        WorkspaceRoute.from_url("ssh", "https://inference.hud.so/v1"),
        WorkspaceRoute.from_url("shell", "http://gateway.test:8080"),
    )

    async with (
        control_peer([Action("answer", result=HELLO)]) as peer,
        connect(Runtime(peer.url), workspace_routes=routes),
    ):
        (hello,) = peer.requests("hello")

    assert hello["params"]["workspace_routes"] == [
        {"capability": "ssh", "host": "inference.hud.so", "port": 443},
        {"capability": "shell", "host": "gateway.test", "port": 8080},
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


@pytest.mark.parametrize(
    ("reply", "message"),
    [
        (
            Action(
                "answer",
                error={"code": -32000, "message": "frame exceeds size limit"},
                reply_id=None,
            ),
            "-32000",
        ),
        (Action("answer", error={"code": -32601, "message": "unknown method"}), "-32601"),
        (Action("answer", result=HELLO, reply_id=99), "reply id did not match"),
        (Action("answer", result=["not", "an", "object"]), "not an object"),
    ],
    ids=["an-error-frame", "an-error-reply", "a-mismatched-id", "a-result-that-is-not-an-object"],
)
async def test_a_handshake_the_env_breaks_fails_without_retrying(reply: Any, message: str) -> None:
    async with control_peer([reply]) as peer:
        with pytest.raises(HudProtocolError, match=message):
            async with connect(Runtime(peer.url), ready_timeout=10):
                pass

    assert peer.accepted == 1


async def test_an_env_that_keeps_hanging_up_is_given_up_on_at_the_runtimes_ready_timeout() -> None:
    loop = asyncio.get_running_loop()

    async with control_peer([Action("hang_up")]) as peer:
        started = loop.time()
        with pytest.raises(EOFError, match="env closed connection during 'hello'"):
            async with connect(Runtime(peer.url, params={"ready_timeout": 0.3}), ready_timeout=60):
                pass

    assert loop.time() - started < 5


async def test_a_refused_port_is_given_up_on_at_the_runtimes_ready_timeout() -> None:
    server = await asyncio.start_server(lambda reader, writer: None, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    server.close()
    await server.wait_closed()
    loop = asyncio.get_running_loop()

    started = loop.time()
    with pytest.raises(OSError):
        runtime = Runtime(f"tcp://127.0.0.1:{port}", params={"ready_timeout": 0.3})
        async with connect(runtime, ready_timeout=60):
            pass

    assert loop.time() - started < 5


@pytest.mark.parametrize("ready_timeout", [-1.0, math.nan, math.inf])
async def test_a_ready_timeout_that_is_not_a_positive_finite_number_is_refused(
    ready_timeout: float,
) -> None:
    runtime = Runtime("tcp://127.0.0.1:9", params={"ready_timeout": ready_timeout})

    with pytest.raises(ValueError, match="must be a positive finite number"):
        async with connect(runtime):
            pass


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


@pytest.mark.parametrize(
    ("ref", "candidates"),
    [
        ("rfb", ["screen-0", "screen-1"]),
        ("browser", ["shell", "screen-0", "screen-1", "custom"]),
    ],
    ids=["an-ambiguous-protocol", "a-missing-capability"],
)
async def test_a_binding_that_does_not_name_one_capability_says_what_there_is(
    ref: str, candidates: list[str]
) -> None:
    async with (
        control_peer([Action("answer", result={**HELLO, "bindings": BINDINGS})]) as peer,
        connect(Runtime(peer.url)) as client,
    ):
        with pytest.raises(KeyError) as raised:
            client.binding(ref)

    assert all(name in str(raised.value) for name in candidates)


async def test_a_client_on_raw_streams_hands_out_bindings_as_the_env_wrote_them() -> None:
    async with control_peer([Action("answer", result={**HELLO, "bindings": BINDINGS})]) as peer:
        reader, writer = await asyncio.open_connection("127.0.0.1", peer.port)
        client = HudClient(reader, writer)
        try:
            await client.hello()
            url = client.binding("shell").url
        finally:
            await client.close()

    assert url == "ssh://agent@environment:22"


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


async def test_a_stream_to_an_env_that_stopped_accepting_ends_the_local_connection() -> None:
    async with (
        control_peer([Action("answer", result={**HELLO, "bindings": BINDINGS[:1]})]) as peer,
        connect(Runtime(peer.url)) as client,
    ):
        local = urlsplit(client.binding("shell").url)
        peer.stop_accepting()
        reader, writer = await asyncio.open_connection(local.hostname, local.port)
        received = await asyncio.wait_for(reader.read(), 10)
        writer.close()

    assert received == b""


async def test_a_task_list_that_breaks_the_protocol_raises_and_the_session_goes_on() -> None:
    async with (
        control_peer(
            [
                Action("answer", result=HELLO),
                Action("answer", result={"tasks": "fix"}),
                Action("answer", result={"cancelled": True}),
            ]
        ) as peer,
        connect(Runtime(peer.url)) as client,
    ):
        with pytest.raises(HudProtocolError, match=r"tasks\.list: 'tasks' must be a list"):
            await client.list_tasks()
        await client.cancel()

    assert [request["method"] for request in peer.requests()] == [
        "hello",
        "tasks.list",
        "tasks.cancel",
    ]


async def test_a_reply_to_another_request_raises_and_ends_the_connection() -> None:
    async with (
        control_peer(
            [Action("answer", result=HELLO), Action("answer", result={"score": 1.0}, reply_id=7)]
        ) as peer,
        connect(Runtime(peer.url)) as client,
    ):
        with pytest.raises(HudProtocolError, match=r"'tasks\.grade': reply id did not match"):
            await client.grade({"answer": "done"})
        with pytest.raises((EOFError, OSError)):
            await client.cancel()

    assert len(peer.requests()) == 2
