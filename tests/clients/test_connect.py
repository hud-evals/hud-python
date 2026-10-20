"""The control channel's client side: ``hud.clients.connect`` and ``HudClient`` against a
scripted peer that answers each connection and records every frame it receives.

Not covered: the 120 s control heartbeat. Its interval is a module constant with no
public knob, and waiting it out has no place in the default lane.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
from typing import Any
from urllib.parse import urlsplit

import pytest

from hud.capabilities import Connection
from hud.clients import HudClient, HudProtocolError, connect
from hud.environment import WorkspaceRoute
from hud.eval.runtime import Runtime
from tests.clients.conftest import answer, control_peer, hang_up

HELLO = {"session_id": "s-1", "env": {"name": "stub", "version": "1.0"}, "bindings": []}
BINDINGS = [
    {"name": "shell", "protocol": "ssh/2", "url": "ssh://agent@environment:22"},
    {"name": "screen-0", "protocol": "rfb/3.8", "url": "rfb://environment:5900"},
    {"name": "screen-1", "protocol": "rfb/3.8", "url": "rfb://environment:5901"},
    {"name": "custom", "protocol": "custom/1", "url": "tcp://environment:7000"},
]


async def test_hello_names_the_session_and_the_env_it_reached() -> None:
    async with control_peer([answer(HELLO)]) as peer, connect(Runtime(peer.url)) as client:
        manifest = client.manifest

    assert peer.requests() == [{"jsonrpc": "2.0", "id": 1, "method": "hello", "params": {}}]
    assert manifest is not None
    assert (manifest.session_id, manifest.server_info.name, manifest.server_info.version) == (
        "s-1",
        "stub",
        "1.0",
    )


async def test_hello_carries_the_workspace_routes_and_connections_it_was_given() -> None:
    routes = (
        WorkspaceRoute.from_url("ssh", "https://inference.hud.so/v1"),
        WorkspaceRoute.from_url("shell", "http://gateway.test:8080"),
    )
    connection = Connection(
        name="inference",
        capability="ssh",
        url="https://inference.hud.so",
        headers={"Authorization": "Bearer secret"},
    )

    async with (
        control_peer([answer(HELLO)]) as peer,
        connect(Runtime(peer.url), workspace_routes=routes, connections=[connection]),
    ):
        (hello,) = peer.requests("hello")

    assert hello["params"]["workspace_routes"] == [
        {"capability": "ssh", "host": "inference.hud.so", "port": 443},
        {"capability": "shell", "host": "gateway.test", "port": 8080},
    ]
    assert hello["params"]["connections"] == [
        {
            "name": "inference",
            "capability": "ssh",
            "url": "https://inference.hud.so",
            "headers": {"Authorization": "Bearer secret"},
        }
    ]


@pytest.mark.parametrize("immediately", [True, False], ids=["on-accept", "during-hello"])
async def test_connect_retries_an_env_that_hangs_up_until_it_answers_hello(
    immediately: bool,
) -> None:
    not_ready = [hang_up(immediately=immediately)]

    async with (
        control_peer(not_ready, not_ready, [answer(HELLO)]) as peer,
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
            answer(error={"code": -32000, "message": "frame exceeds size limit"}, id=None),
            "hud rpc error -32000: frame exceeds size limit",
        ),
        (
            answer(error={"code": -32601, "message": "unknown method"}),
            "hud rpc error -32601: unknown method",
        ),
        (answer(HELLO, id=99), "'hello': reply id did not match request"),
        (answer(["not", "an", "object"]), "'hello': result was not an object"),
    ],
    ids=["an-error-frame", "an-error-reply", "a-mismatched-id", "a-result-that-is-not-an-object"],
)
async def test_a_handshake_the_env_breaks_fails_without_retrying(reply: Any, message: str) -> None:
    async with control_peer([reply]) as peer:
        with pytest.raises(HudProtocolError, match=message):
            async with connect(Runtime(peer.url), ready_timeout=10):
                pass

    assert peer.accepted == 1


async def test_an_env_that_never_answers_is_given_up_on_at_the_runtimes_ready_timeout() -> None:
    loop = asyncio.get_running_loop()

    async with control_peer([hang_up()]) as peer:
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
    ("ref", "scheme"),
    [("shell", "ssh"), ("ssh", "ssh"), ("ssh/2", "ssh"), ("screen-1", "rfb")],
    ids=["by-name", "by-protocol", "by-versioned-protocol", "by-name-among-same-protocol"],
)
async def test_a_binding_resolves_to_a_local_tunnel_by_name_or_protocol(
    ref: str, scheme: str
) -> None:
    async with (
        control_peer([answer({**HELLO, "bindings": BINDINGS})]) as peer,
        connect(Runtime(peer.url)) as client,
    ):
        url = urlsplit(client.binding(ref).url)

    assert (url.scheme, url.hostname) == (scheme, "127.0.0.1")


@pytest.mark.parametrize(
    ("ref", "message"),
    [
        ("rfb", "ambiguous capability 'rfb'; matches: screen-0 (rfb/3.8), screen-1 (rfb/3.8)"),
        (
            "browser",
            "no capability 'browser' (available: shell (ssh/2), screen-0 (rfb/3.8), "
            "screen-1 (rfb/3.8), custom (custom/1))",
        ),
    ],
    ids=["an-ambiguous-protocol", "a-missing-capability"],
)
async def test_a_binding_that_does_not_name_one_capability_says_what_there_is(
    ref: str, message: str
) -> None:
    async with (
        control_peer([answer({**HELLO, "bindings": BINDINGS})]) as peer,
        connect(Runtime(peer.url)) as client,
    ):
        with pytest.raises(KeyError) as raised:
            client.binding(ref)

    assert raised.value.args == (message,)


async def test_a_client_on_raw_streams_hands_out_bindings_as_the_env_wrote_them() -> None:
    async with control_peer([answer({**HELLO, "bindings": BINDINGS})]) as peer:
        reader, writer = await asyncio.open_connection("127.0.0.1", peer.port)
        client = HudClient(reader, writer)
        try:
            await client.hello()
            url = client.binding("shell").url
        finally:
            await client.close()

    assert url == "ssh://agent@environment:22"


async def test_a_stream_the_env_refuses_ends_the_local_connection(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.WARNING, logger="hud.clients")

    async with (
        control_peer([answer({**HELLO, "bindings": BINDINGS[:1]})]) as peer,
        connect(Runtime(peer.url)) as client,
    ):
        local = urlsplit(client.binding("shell").url)
        reader, writer = await asyncio.open_connection(local.hostname, local.port)
        received = await asyncio.wait_for(reader.read(), 10)
        writer.close()

    assert received == b""
    assert any(
        "tunnel.open 'shell' refused:" in m and "unknown capability" in m for m in caplog.messages
    )


async def test_a_stream_to_an_env_that_stopped_accepting_ends_the_local_connection(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.WARNING, logger="hud.clients")

    async with (
        control_peer([answer({**HELLO, "bindings": BINDINGS[:1]})]) as peer,
        connect(Runtime(peer.url)) as client,
    ):
        local = urlsplit(client.binding("shell").url)
        peer.stop_accepting()
        reader, writer = await asyncio.open_connection(local.hostname, local.port)
        received = await asyncio.wait_for(reader.read(), 10)
        writer.close()

    assert received == b""
    assert any(
        m.startswith("tunnel peer 127.0.0.1:") and "connection failed" in m for m in caplog.messages
    )


async def test_a_task_list_that_breaks_the_protocol_raises_and_the_session_goes_on() -> None:
    async with (
        control_peer(
            [answer(HELLO), answer({"tasks": "fix"}), answer({"cancelled": True})]
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
        control_peer([answer(HELLO), answer({"score": 1.0}, id=7)]) as peer,
        connect(Runtime(peer.url)) as client,
    ):
        with pytest.raises(HudProtocolError, match=r"'tasks\.grade': reply id did not match"):
            await client.grade({"answer": "done"})
        with pytest.raises((EOFError, OSError)):
            await client.cancel()

    assert len(peer.requests()) == 2
