"""``hud.clients.connect`` and ``HudClient`` against a scripted control-channel peer.

The peer (:func:`tests.harness.control_peer`) is the env side of the wire:
it answers each connection from a script and records every frame, so the
frames the client sends are pinned literally. Capability streams it tunnels
reach real servers: a served workspace's SSH daemon through a flaky relay.

Not covered: the 120 s control heartbeat. Its interval is a module constant
with no public knob, and waiting it out has no place in the default lane.
"""

from __future__ import annotations

import asyncio
import logging
import math
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit

import pytest
from dirty_equals import IsStr

from hud import Environment
from hud.capabilities import Connection, SSHClient
from hud.clients import HudClient, HudProtocolError, connect
from hud.environment import WorkspaceRoute
from hud.eval.runtime import Runtime
from tests.harness import answer, control_peer, eventually, hang_up, relay, served

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable
    from pathlib import Path


def hello(*bindings: dict[str, Any]) -> dict[str, Any]:
    return {
        "session_id": "s-1",
        "env": {"name": "stub", "version": "1.0"},
        "bindings": list(bindings),
    }


BINDINGS = (
    {"name": "shell", "protocol": "ssh/2", "url": "ssh://agent@environment:22"},
    {"name": "screen-0", "protocol": "rfb/3.8", "url": "rfb://environment:5900"},
    {"name": "screen-1", "protocol": "rfb/3.8", "url": "rfb://environment:5901"},
    {"name": "custom", "protocol": "custom/1", "url": "tcp://environment:7000"},
)


@pytest.mark.parametrize(
    ("routes", "connections", "params"),
    [
        ((), (), {}),
        (
            (
                WorkspaceRoute.from_url("ssh", "https://inference.hud.so/v1"),
                WorkspaceRoute.from_url("shell", "http://gateway.test:8080"),
            ),
            (),
            {
                "workspace_routes": [
                    {"capability": "ssh", "host": "inference.hud.so", "port": 443},
                    {"capability": "shell", "host": "gateway.test", "port": 8080},
                ]
            },
        ),
        (
            (),
            (
                Connection(
                    name="inference",
                    capability="ssh",
                    url="https://inference.hud.so",
                    headers={"Authorization": "Bearer secret"},
                ),
            ),
            {
                "connections": [
                    {
                        "name": "inference",
                        "capability": "ssh",
                        "url": "https://inference.hud.so",
                        "headers": {"Authorization": "Bearer secret"},
                    }
                ]
            },
        ),
    ],
    ids=["bare", "workspace routes", "controller connections"],
)
async def test_hello_carries_the_workspace_routes_and_connections(
    routes: tuple[WorkspaceRoute, ...], connections: tuple[Connection, ...], params: dict[str, Any]
) -> None:
    async with (
        control_peer([answer(hello())]) as peer,
        connect(Runtime(peer.url), workspace_routes=routes, connections=connections) as client,
    ):
        manifest = client.manifest

    assert peer.requests() == [{"jsonrpc": "2.0", "id": 1, "method": "hello", "params": params}]
    assert manifest is not None
    assert (manifest.session_id, manifest.server_info.name, manifest.server_info.version) == (
        "s-1",
        "stub",
        "1.0",
    )


@pytest.mark.parametrize(
    "not_ready",
    [hang_up(immediately=True), hang_up()],
    ids=["accept then close", "close during hello"],
)
async def test_connect_retries_until_the_env_answers_hello(not_ready: Any) -> None:
    async with (
        control_peer([not_ready], [not_ready], [answer(hello())]) as peer,
        connect(Runtime(peer.url), ready_timeout=10) as client,
    ):
        assert client.manifest is not None

    assert peer.accepted == 3


@pytest.mark.parametrize(
    ("script", "message"),
    [
        (
            [answer(error={"code": -32000, "message": "frame exceeds size limit"}, id=None)],
            "hud rpc error -32000: frame exceeds size limit",
        ),
        (
            [answer(error={"code": -32601, "message": "unknown method"})],
            "hud rpc error -32601: unknown method",
        ),
        ([answer(hello(), id=99)], "'hello': reply id did not match request"),
        ([answer(["not", "an", "object"])], "'hello': result was not an object"),
    ],
    ids=["an error frame", "an error reply", "a mismatched id", "a non-object"],
)
async def test_connect_fails_without_retrying_on_a_handshake_the_env_breaks(
    script: list[Any], message: str
) -> None:
    async with control_peer(script) as peer:
        with pytest.raises(HudProtocolError, match=message):
            async with connect(Runtime(peer.url), ready_timeout=10):
                pass

    assert peer.accepted == 1


async def test_connect_gives_up_at_the_runtime_ready_timeout_on_an_env_that_never_answers() -> None:
    loop = asyncio.get_running_loop()
    async with control_peer([hang_up()]) as peer:
        started = loop.time()
        with pytest.raises(EOFError, match="env closed connection during 'hello'"):
            async with connect(Runtime(peer.url, params={"ready_timeout": 0.3}), ready_timeout=60):
                pass

    assert loop.time() - started < 5


async def test_connect_gives_up_on_a_refused_port_after_the_ready_timeout() -> None:
    server = await asyncio.start_server(lambda reader, writer: None, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    server.close()
    await server.wait_closed()

    with pytest.raises(OSError):
        async with connect(Runtime(f"tcp://127.0.0.1:{port}"), ready_timeout=0.3):
            pass


@pytest.mark.parametrize("value", [True, 0, -1.0, math.nan, math.inf, "5"])
async def test_an_invalid_runtime_ready_timeout_is_refused(value: object) -> None:
    with pytest.raises(ValueError, match="must be a positive finite number"):
        async with connect(Runtime("tcp://127.0.0.1:9", params={"ready_timeout": value})):
            pass


async def test_only_tcp_control_transports_are_supported() -> None:
    with pytest.raises(NotImplementedError, match="control transport 'unix' not supported"):
        async with connect(Runtime("unix:///run/hud.sock")):
            pass


@pytest.mark.parametrize(
    ("ref", "outcome"),
    [
        ("shell", IsStr(regex=r"ssh://agent@127\.0\.0\.1:\d+")),
        ("ssh", IsStr(regex=r"ssh://agent@127\.0\.0\.1:\d+")),
        ("ssh/2", IsStr(regex=r"ssh://agent@127\.0\.0\.1:\d+")),
        ("screen-1", IsStr(regex=r"rfb://127\.0\.0\.1:\d+")),
        (
            "rfb",
            KeyError("ambiguous capability 'rfb'; matches: screen-0 (rfb/3.8), screen-1 (rfb/3.8)"),
        ),
        (
            "browser",
            KeyError(
                "no capability 'browser' (available: shell (ssh/2), screen-0 (rfb/3.8), "
                "screen-1 (rfb/3.8), custom (custom/1))"
            ),
        ),
    ],
)
async def test_a_binding_resolves_by_name_or_protocol_to_a_local_tunnel(
    ref: str, outcome: Any
) -> None:
    async with (
        control_peer([answer(hello(*BINDINGS))]) as peer,
        connect(Runtime(peer.url)) as client,
    ):
        if isinstance(outcome, KeyError):
            with pytest.raises(KeyError) as raised:
                client.binding(ref)
            assert raised.value.args == outcome.args
        else:
            assert client.binding(ref).url == outcome


async def test_a_raw_stream_client_needs_hello_and_passes_bindings_through() -> None:
    async with control_peer([answer(hello(*BINDINGS))]) as peer:
        reader, writer = await asyncio.open_connection("127.0.0.1", peer.port)
        client = HudClient(reader, writer)
        try:
            with pytest.raises(RuntimeError, match=r"call hello\(\) before accessing bindings"):
                client.binding("shell")
            await client.hello()
            url = client.binding("shell").url
        finally:
            await client.close()

    assert url == "ssh://agent@environment:22"


async def test_opening_a_protocol_without_a_client_points_at_binding() -> None:
    async with (
        control_peer([answer(hello(*BINDINGS))]) as peer,
        connect(Runtime(peer.url)) as client,
    ):
        with pytest.raises(ValueError, match=r"use binding\('custom'\) for raw access"):
            await client.open("custom")


async def test_an_opened_ssh_client_tunnels_to_its_upstream_and_closes_with_the_connection(
    tmp_path: Path,
) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    env = Environment("tunneled")
    env.workspace(root, track_files=False)
    async with served(env) as workspace:
        assert workspace.manifest is not None
        (shell,) = (cap for cap in workspace.manifest.bindings if cap.name == "shell")
        address = urlsplit(shell.url)
        assert address.hostname is not None and address.port is not None
        async with relay(address.hostname, address.port) as flaky:
            binding = {**shell.to_manifest(), "url": "ssh://agent@environment:22"}
            async with control_peer(
                [answer(hello(binding))], upstreams={"shell": ("127.0.0.1", flaky.port)}
            ) as peer:
                async with connect(Runtime(peer.url)) as client:
                    ssh = await client.open("shell")
                    assert isinstance(ssh, SSHClient)
                    result = await ssh.run("echo through the tunnel", timeout=10)
                    again = await client.open("ssh")

                assert again is ssh
                assert ssh.conn.is_closed()
                await eventually(lambda: flaky.open == 0)

    assert result.stdout == "through the tunnel\n"
    assert (flaky.accepted, peer.tunnels) == (1, 1)


@pytest.mark.parametrize(
    ("break_tunnel", "warning"),
    [
        (
            lambda peer: peer.stop_accepting(),
            IsStr(regex=r"tunnel peer 127\.0\.0\.1:\d+ connection failed: .*"),
        ),
        (lambda peer: None, IsStr(regex=r"tunnel\.open 'shell' refused: .*unknown capability.*")),
    ],
    ids=["the env stops accepting", "the env refuses the stream"],
)
async def test_a_tunnel_the_env_cannot_open_ends_the_local_stream(
    break_tunnel: Callable[[Any], None], warning: Any, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.WARNING, logger="hud.clients")
    async with (
        control_peer([answer(hello(BINDINGS[0]))]) as peer,
        connect(Runtime(peer.url)) as client,
    ):
        local = urlsplit(client.binding("shell").url)
        break_tunnel(peer)
        reader, writer = await asyncio.open_connection(local.hostname, local.port)
        try:
            received = await asyncio.wait_for(reader.read(), 10)
        finally:
            writer.close()

    assert received == b""
    assert warning in caplog.messages


@pytest.mark.parametrize(
    ("call", "result", "sent", "returned"),
    [
        (
            lambda client: client.start_task("fix", {"bug": 1}),
            {"prompt": "Fix it."},
            ("tasks.start", {"id": "fix", "args": {"bug": 1}}),
            {"prompt": "Fix it."},
        ),
        (
            lambda client: client.start_task("fix"),
            {"prompt": "Fix it."},
            ("tasks.start", {"id": "fix", "args": {}}),
            {"prompt": "Fix it."},
        ),
        (
            lambda client: client.grade({"answer": "done"}),
            {"score": 1.0},
            ("tasks.grade", {"answer": "done"}),
            {"score": 1.0},
        ),
        (lambda client: client.cancel(), {}, ("tasks.cancel", {}), None),
        (
            lambda client: client.list_tasks(),
            {"tasks": [{"id": "fix", "description": "Fix it."}]},
            ("tasks.list", {}),
            [{"id": "fix", "description": "Fix it."}],
        ),
        (
            lambda client: client.list_tasks(),
            {"tasks": "fix"},
            ("tasks.list", {}),
            HudProtocolError(-32603, "tasks.list: 'tasks' must be a list"),
        ),
        (
            lambda client: client.grade({"answer": "done"}),
            {"error": {"code": -32600, "message": "no task in progress"}},
            ("tasks.grade", {"answer": "done"}),
            HudProtocolError(-32600, "no task in progress"),
        ),
    ],
    ids=["start", "start without args", "grade", "cancel", "list", "list malformed", "grade error"],
)
async def test_task_calls_send_one_request_and_return_its_result(
    call: Callable[[HudClient], Awaitable[Any]],
    result: dict[str, Any],
    sent: tuple[str, dict[str, Any]],
    returned: Any,
) -> None:
    reply = answer(error=result["error"]) if "error" in result else answer(result)
    async with control_peer([answer(hello()), reply]) as peer, connect(Runtime(peer.url)) as client:
        if isinstance(returned, HudProtocolError):
            with pytest.raises(HudProtocolError) as raised:
                await call(client)
            assert (raised.value.code, raised.value.message) == (
                returned.code,
                returned.message,
            )
        else:
            assert await call(client) == returned

    method, params = sent
    assert peer.requests()[1] == {"jsonrpc": "2.0", "id": 2, "method": method, "params": params}


async def test_a_reply_to_another_request_aborts_the_connection() -> None:
    async with (
        control_peer([answer(hello()), answer({"score": 1.0}, id=7)]) as peer,
        connect(Runtime(peer.url)) as client,
    ):
        with pytest.raises(HudProtocolError, match=r"'tasks\.grade': reply id did not match"):
            await client.grade({"answer": "done"})
        with pytest.raises((EOFError, OSError)):
            await client.cancel()

    assert [request["method"] for request in peer.requests()] == ["hello", "tasks.grade"]
