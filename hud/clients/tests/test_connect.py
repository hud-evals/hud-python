"""``connect()`` readiness: the handshake retries until the env actually serves.

A provisioned substrate can sit behind a proxied port (``docker -p``, a
port-forward) that *accepts* TCP before the env behind it is up — those
connections die with EOF at the handshake. Readiness is therefore
protocol-level: ``connect`` keeps retrying through both refused connects and
handshake EOFs until ``hello`` answers or the deadline passes.
"""

from __future__ import annotations

import asyncio
import errno
import logging
import socket
from typing import ClassVar
from urllib.parse import urlsplit

import pytest

import hud.clients.client as client_module
from hud.capabilities import Capability, CapabilityClient, Connection
from hud.clients import ControlTimeoutError, HudProtocolError, connect
from hud.environment import WorkspaceRoute
from hud.environment.utils import read_frame, send_frame
from hud.eval.runtime import Runtime
from hud.settings import settings

HELLO_RESULT = {"session_id": "s-1", "env": {"name": "stub", "version": "1.0"}, "bindings": []}


def test_workspace_route_from_url_extracts_transport_address() -> None:
    assert WorkspaceRoute.from_url("ssh", "https://inference.hud.so/v1") == WorkspaceRoute(
        "ssh",
        "inference.hud.so",
        443,
    )
    assert WorkspaceRoute.from_url("shell", "http://gateway.test:8080") == WorkspaceRoute(
        "shell",
        "gateway.test",
        8080,
    )


async def test_connect_sends_workspace_routes_in_hello() -> None:
    requests: list[dict[str, object]] = []

    async def handler(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            hello = await read_frame(reader)
            assert hello is not None
            requests.append(hello)
            await send_frame(writer, {"jsonrpc": "2.0", "id": hello["id"], "result": HELLO_RESULT})
            await read_frame(reader)
        finally:
            writer.close()

    server = await asyncio.start_server(handler, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    runtime = Runtime(f"tcp://127.0.0.1:{port}")
    route = WorkspaceRoute("ssh", "inference.hud.so", 443)
    try:
        async with connect(runtime, workspace_routes=(route,)):
            pass
    finally:
        server.close()
        await server.wait_closed()

    assert [request["method"] for request in requests] == ["hello"]
    params = requests[0]["params"]
    assert isinstance(params, dict)
    assert params == {"workspace_routes": [route.to_wire()]}


async def test_connect_sends_controller_connections_in_hello() -> None:
    requests: list[dict[str, object]] = []

    async def handler(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            hello = await read_frame(reader)
            assert hello is not None
            requests.append(hello)
            await send_frame(writer, {"jsonrpc": "2.0", "id": hello["id"], "result": HELLO_RESULT})
            await read_frame(reader)
        finally:
            writer.close()

    server = await asyncio.start_server(handler, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    connection = Connection(
        name="inference",
        capability="ssh",
        url="https://inference.hud.so",
        headers={"Authorization": "Bearer secret"},
    )
    try:
        async with connect(
            Runtime(f"tcp://127.0.0.1:{port}"),
            connections=(connection,),
        ):
            pass
    finally:
        server.close()
        await server.wait_closed()

    params = requests[0]["params"]
    assert isinstance(params, dict)
    assert params == {"connections": [connection.to_wire()]}


async def test_open_retries_transient_capability_connection_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class RetryingClient(CapabilityClient):
        protocol: ClassVar[str] = "test/1"
        attempts = 0

        @classmethod
        async def connect(cls, cap: Capability) -> RetryingClient:
            del cap
            cls.attempts += 1
            if cls.attempts < 3:
                raise ConnectionError("connection reset")
            return cls()

        async def close(self) -> None:
            pass

    async def handler(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            hello = await read_frame(reader)
            assert hello is not None
            await send_frame(
                writer,
                {
                    "jsonrpc": "2.0",
                    "id": hello["id"],
                    "result": {
                        **HELLO_RESULT,
                        "bindings": [
                            {
                                "name": "test",
                                "protocol": RetryingClient.protocol,
                                "url": "tcp://environment:1234",
                            }
                        ],
                    },
                },
            )
            await read_frame(reader)
        finally:
            writer.close()

    monkeypatch.setitem(client_module._CLIENT_REGISTRY, RetryingClient.protocol, RetryingClient)
    monkeypatch.setattr(client_module, "_CAPABILITY_CONNECT_BASE_DELAY_SECONDS", 0)
    server = await asyncio.start_server(handler, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    try:
        async with connect(Runtime(f"tcp://127.0.0.1:{port}")) as client:
            opened = await client.open("test")
            assert await client.open("test") is opened
    finally:
        server.close()
        await server.wait_closed()

    assert RetryingClient.attempts == 3


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


async def _serve_tunnels(
    outcomes: list[str],
) -> tuple[asyncio.Server, int, list[bytes]]:
    """A stub env: answers ``hello``, then settles each ``tunnel.open`` as the next outcome.

    ``eof`` closes the connection, ``stall`` never answers, ``unavailable`` replies with a
    dial error, ``invalid`` replies with invalid params, and ``open`` confirms the stream and
    echoes its bytes. Returns the bytes each confirmed stream received.
    """
    received: list[bytes] = []
    pending = iter(outcomes)

    async def handler(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            first = await read_frame(reader)
            assert first is not None
            if first["method"] == "hello":
                bindings = [{"name": "shell", "protocol": "ssh/2", "url": "ssh://agent@env:22"}]
                result = {**HELLO_RESULT, "bindings": bindings}
                await send_frame(writer, {"jsonrpc": "2.0", "id": first["id"], "result": result})
                await read_frame(reader)
                return
            outcome = next(pending, "open")
            if outcome in ("unavailable", "invalid"):
                code = -32000 if outcome == "unavailable" else -32602
                error = {"code": code, "message": outcome}
                await send_frame(writer, {"jsonrpc": "2.0", "id": first["id"], "error": error})
            elif outcome == "stall":
                await reader.read()
            elif outcome == "open":
                await send_frame(writer, {"jsonrpc": "2.0", "id": first["id"], "result": {}})
                data = await reader.readexactly(4)
                received.append(data)
                writer.write(data)
                await writer.drain()
        finally:
            writer.close()

    server = await asyncio.start_server(handler, "127.0.0.1", 0)
    return server, server.sockets[0].getsockname()[1], received


async def test_tunnel_rides_out_connections_the_env_drops_before_confirming(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(client_module, "_CAPABILITY_CONNECT_BASE_DELAY_SECONDS", 0.01)
    server, port, received = await _serve_tunnels(["eof", "unavailable", "eof", "open"])
    try:
        async with connect(Runtime(f"tcp://127.0.0.1:{port}")) as client:
            route = urlsplit(client.binding("shell").url)
            reader, writer = await asyncio.open_connection(route.hostname, route.port)
            try:
                writer.write(b"ping")
                assert await asyncio.wait_for(reader.readexactly(4), 5.0) == b"ping"
            finally:
                writer.close()
                await writer.wait_closed()
    finally:
        server.close()
        await server.wait_closed()

    assert received == [b"ping"]


async def test_tunnel_open_stalled_past_the_heartbeat_timeout_is_redialed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(client_module, "_CAPABILITY_CONNECT_BASE_DELAY_SECONDS", 0.01)
    monkeypatch.setattr(settings, "control_heartbeat_timeout", 0.05)
    server, port, received = await _serve_tunnels(["stall", "open"])
    try:
        async with connect(Runtime(f"tcp://127.0.0.1:{port}")) as client:
            route = urlsplit(client.binding("shell").url)
            reader, writer = await asyncio.open_connection(route.hostname, route.port)
            try:
                writer.write(b"ping")
                assert await asyncio.wait_for(reader.readexactly(4), 5.0) == b"ping"
            finally:
                writer.close()
                await writer.wait_closed()
    finally:
        server.close()
        await server.wait_closed()

    assert received == [b"ping"]


async def test_tunnel_refused_with_invalid_params_is_not_retried(
    caplog: pytest.LogCaptureFixture,
) -> None:
    server, port, received = await _serve_tunnels(["invalid", "open"])
    try:
        async with connect(Runtime(f"tcp://127.0.0.1:{port}")) as client:
            route = urlsplit(client.binding("shell").url)
            caplog.set_level(logging.WARNING, logger="hud.clients")

            reader, writer = await asyncio.open_connection(route.hostname, route.port)
            try:
                assert await asyncio.wait_for(reader.read(), 5.0) == b""
            finally:
                writer.close()
                await writer.wait_closed()
    finally:
        server.close()
        await server.wait_closed()

    assert received == []
    assert any("hud rpc error -32602: invalid" in message for message in caplog.messages)


async def test_tunnel_gives_up_once_the_outage_budget_is_spent(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setattr(client_module, "_CAPABILITY_CONNECT_BASE_DELAY_SECONDS", 0.01)
    monkeypatch.setattr(settings, "control_outage_budget", 0.1)
    server, port, received = await _serve_tunnels(["eof"] * 1000)
    try:
        async with connect(Runtime(f"tcp://127.0.0.1:{port}")) as client:
            route = urlsplit(client.binding("shell").url)
            caplog.set_level(logging.WARNING, logger="hud.clients")

            reader, writer = await asyncio.open_connection(route.hostname, route.port)
            try:
                assert await asyncio.wait_for(reader.read(), 5.0) == b""
            finally:
                writer.close()
                await writer.wait_closed()
    finally:
        server.close()
        await server.wait_closed()

    assert received == []
    assert any("tunnel 'shell'" in m and "failed" in m for m in caplog.messages)


async def test_heartbeat_serializes_calls_without_aborting_on_protocol_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests: list[dict[str, object]] = []
    heartbeat_received = asyncio.Event()
    release_heartbeat = asyncio.Event()

    async def handler(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            hello = await read_frame(reader)
            assert hello is not None
            requests.append(hello)
            await send_frame(
                writer,
                {"jsonrpc": "2.0", "id": hello["id"], "result": HELLO_RESULT},
            )

            heartbeat = await read_frame(reader)
            assert heartbeat is not None
            requests.append(heartbeat)
            heartbeat_received.set()
            await release_heartbeat.wait()
            await send_frame(
                writer,
                {
                    "jsonrpc": "2.0",
                    "id": heartbeat["id"],
                    "error": {"code": -32601, "message": "heartbeat refused"},
                },
            )

            grade = await read_frame(reader)
            assert grade is not None
            requests.append(grade)
            await send_frame(
                writer,
                {"jsonrpc": "2.0", "id": grade["id"], "result": {"score": 1.0}},
            )
            await read_frame(reader)
        finally:
            writer.close()

    monkeypatch.setattr(client_module, "_CONTROL_HEARTBEAT_INTERVAL_SECONDS", 0.01)
    server = await asyncio.start_server(handler, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    try:
        async with connect(Runtime(f"tcp://127.0.0.1:{port}")) as client:
            await asyncio.wait_for(heartbeat_received.wait(), 1.0)
            grade = asyncio.create_task(client.grade({"answer": "done"}))
            await asyncio.sleep(0)
            assert [request["method"] for request in requests] == ["hello", "hello"]
            release_heartbeat.set()
            assert await asyncio.wait_for(grade, 1.0) == {"score": 1.0}
    finally:
        server.close()
        await server.wait_closed()

    assert [request["method"] for request in requests] == ["hello", "hello", "tasks.grade"]
    assert requests[1]["params"] == {"session_id": "s-1"}


async def test_heartbeat_survives_a_slow_reply_from_a_live_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    heartbeat_received = asyncio.Event()
    requests: list[dict[str, object]] = []

    async def handler(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            hello = await read_frame(reader)
            assert hello is not None
            await send_frame(
                writer,
                {"jsonrpc": "2.0", "id": hello["id"], "result": HELLO_RESULT},
            )
            heartbeat = await read_frame(reader)
            assert heartbeat is not None
            heartbeat_received.set()
            await asyncio.sleep(5.5)
            await send_frame(
                writer,
                {"jsonrpc": "2.0", "id": heartbeat["id"], "result": HELLO_RESULT},
            )
            while (request := await read_frame(reader)) is not None:
                requests.append(request)
                result = {"score": 1.0} if request["method"] == "tasks.grade" else HELLO_RESULT
                await send_frame(writer, {"jsonrpc": "2.0", "id": request["id"], "result": result})
        finally:
            writer.close()

    monkeypatch.setattr(client_module, "_CONTROL_HEARTBEAT_INTERVAL_SECONDS", 0.01)
    server = await asyncio.start_server(handler, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    try:
        async with connect(Runtime(f"tcp://127.0.0.1:{port}")) as client:
            await asyncio.wait_for(heartbeat_received.wait(), 1.0)
            grade = await asyncio.wait_for(client.grade({"answer": "done"}), 10.0)
    finally:
        server.close()
        await server.wait_closed()

    assert grade == {"score": 1.0}
    assert "tasks.grade" in [request["method"] for request in requests]


async def test_heartbeat_fails_a_hung_env_with_a_typed_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    heartbeat_received = asyncio.Event()
    body_interrupted = asyncio.Event()

    async def handler(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            hello = await read_frame(reader)
            assert hello is not None
            await send_frame(
                writer,
                {"jsonrpc": "2.0", "id": hello["id"], "result": HELLO_RESULT},
            )
            heartbeat = await read_frame(reader)
            assert heartbeat is not None
            heartbeat_received.set()
            await reader.read()
        finally:
            writer.close()

    monkeypatch.setattr(client_module, "_CONTROL_HEARTBEAT_INTERVAL_SECONDS", 0.01)
    monkeypatch.setattr(client_module, "_CONTROL_REDIAL_INTERVAL_SECONDS", 0.01)
    monkeypatch.setattr(settings, "control_heartbeat_timeout", 0.05)
    monkeypatch.setattr(settings, "control_outage_budget", 0.2)
    server = await asyncio.start_server(handler, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    try:
        with pytest.raises(ControlTimeoutError, match=r"did not reply to 'hello' within 0\.05s"):
            async with connect(Runtime(f"tcp://127.0.0.1:{port}")):
                await asyncio.wait_for(heartbeat_received.wait(), 1.0)
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    body_interrupted.set()
                    raise
    finally:
        server.close()
        await server.wait_closed()

    assert body_interrupted.is_set()


async def test_heartbeat_redials_and_resumes_the_session_after_the_control_link_drops(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connections: list[list[dict[str, object]]] = []
    resumed = asyncio.Event()

    async def handler(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        requests: list[dict[str, object]] = []
        connections.append(requests)
        try:
            while (request := await read_frame(reader)) is not None:
                requests.append(request)
                if len(connections) == 1 and len(requests) == 2:
                    return  # the link drops under the first heartbeat
                if len(connections) == 2:
                    resumed.set()
                result = {"score": 1.0} if request["method"] == "tasks.grade" else HELLO_RESULT
                await send_frame(writer, {"jsonrpc": "2.0", "id": request["id"], "result": result})
        finally:
            writer.close()

    monkeypatch.setattr(client_module, "_CONTROL_HEARTBEAT_INTERVAL_SECONDS", 0.01)
    monkeypatch.setattr(client_module, "_CONTROL_REDIAL_INTERVAL_SECONDS", 0.01)
    server = await asyncio.start_server(handler, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    try:
        async with connect(Runtime(f"tcp://127.0.0.1:{port}")) as client:
            await asyncio.wait_for(resumed.wait(), 5.0)
            assert await client.grade({"answer": "done"}) == {"score": 1.0}
    finally:
        server.close()
        await server.wait_closed()

    assert connections[1][0]["params"] == {"session_id": "s-1", "takeover": True}
    assert "tasks.grade" in [request["method"] for request in connections[1]]


async def test_heartbeat_timeout_on_a_blackholed_link_redials_instead_of_failing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connections: list[list[dict[str, object]]] = []
    resumed = asyncio.Event()

    async def handler(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        requests: list[dict[str, object]] = []
        connections.append(requests)
        try:
            while (request := await read_frame(reader)) is not None:
                requests.append(request)
                if len(connections) == 1 and len(requests) == 2:
                    await reader.read()  # the link stays open but never answers the heartbeat
                    return
                if len(connections) == 2:
                    resumed.set()
                result = {"score": 1.0} if request["method"] == "tasks.grade" else HELLO_RESULT
                await send_frame(writer, {"jsonrpc": "2.0", "id": request["id"], "result": result})
        finally:
            writer.close()

    monkeypatch.setattr(client_module, "_CONTROL_HEARTBEAT_INTERVAL_SECONDS", 0.01)
    monkeypatch.setattr(client_module, "_CONTROL_REDIAL_INTERVAL_SECONDS", 0.01)
    monkeypatch.setattr(settings, "control_heartbeat_timeout", 0.05)
    server = await asyncio.start_server(handler, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    try:
        async with connect(Runtime(f"tcp://127.0.0.1:{port}")) as client:
            await asyncio.wait_for(resumed.wait(), 5.0)
            assert await client.grade({"answer": "done"}) == {"score": 1.0}
    finally:
        server.close()
        await server.wait_closed()

    assert connections[1][0]["params"] == {"session_id": "s-1", "takeover": True}


async def test_heartbeat_fails_with_the_last_error_once_the_outage_budget_is_spent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def handler(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            hello = await read_frame(reader)
            assert hello is not None
            await send_frame(writer, {"jsonrpc": "2.0", "id": hello["id"], "result": HELLO_RESULT})
        finally:
            writer.close()

    monkeypatch.setattr(client_module, "_CONTROL_HEARTBEAT_INTERVAL_SECONDS", 0.01)
    monkeypatch.setattr(client_module, "_CONTROL_REDIAL_INTERVAL_SECONDS", 0.01)
    monkeypatch.setattr(settings, "control_outage_budget", 0.1)
    server = await asyncio.start_server(handler, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    try:
        with pytest.raises(EOFError, match="closed connection during 'hello'"):
            async with connect(Runtime(f"tcp://127.0.0.1:{port}")):
                await asyncio.Event().wait()
    finally:
        server.close()
        await server.wait_closed()


async def test_outage_that_blackholes_new_connections_fails_with_a_typed_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def handler(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            hello = await read_frame(reader)
            assert hello is not None
            await send_frame(writer, {"jsonrpc": "2.0", "id": hello["id"], "result": HELLO_RESULT})
        finally:
            writer.close()

    real_open_connection = asyncio.open_connection
    dials = 0

    async def open_connection_then_hang(
        host: str, port: int
    ) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
        nonlocal dials
        dials += 1
        if dials == 1:
            return await real_open_connection(host, port)
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    monkeypatch.setattr(client_module.asyncio, "open_connection", open_connection_then_hang)
    monkeypatch.setattr(client_module, "_CONTROL_HEARTBEAT_INTERVAL_SECONDS", 0.01)
    monkeypatch.setattr(client_module, "_CONTROL_REDIAL_INTERVAL_SECONDS", 0.01)
    monkeypatch.setattr(settings, "control_heartbeat_timeout", 0.05)
    monkeypatch.setattr(settings, "control_outage_budget", 0.2)
    server = await asyncio.start_server(handler, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    try:
        with pytest.raises(ControlTimeoutError, match="did not accept a control connection"):
            async with connect(Runtime(f"tcp://127.0.0.1:{port}")):
                await asyncio.Event().wait()
    finally:
        server.close()
        await server.wait_closed()


async def test_connect_uses_runtime_ready_timeout_param(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, float | int | str] = {}

    class _FakeClient:
        async def close(self) -> None:
            pass

    async def fake_connect_ready(
        host: str,
        port: int,
        *,
        ready_timeout: float,
        workspace_routes: tuple[WorkspaceRoute, ...],
        connections: tuple[Connection, ...],
        interval: float = 0.5,
    ) -> _FakeClient:
        seen["host"] = host
        seen["port"] = port
        seen["ready_timeout"] = ready_timeout
        assert workspace_routes == ()
        assert connections == ()
        seen["interval"] = interval
        return _FakeClient()

    monkeypatch.setattr(client_module, "_connect_ready", fake_connect_ready)

    async with client_module.connect(
        Runtime("tcp://127.0.0.1:1234", params={"ready_timeout": 300.0})
    ):
        pass

    assert seen == {
        "host": "127.0.0.1",
        "port": 1234,
        "ready_timeout": 300.0,
        "interval": 0.5,
    }


@pytest.mark.parametrize("protocol_error", [False, True])
async def test_connect_reports_handshake_failure(protocol_error: bool) -> None:
    async def handler(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            await read_frame(reader)
            if protocol_error:
                await send_frame(
                    writer,
                    {
                        "jsonrpc": "2.0",
                        "id": None,
                        "error": {"code": -32000, "message": "frame exceeds size limit"},
                    },
                )
        finally:
            writer.close()

    server = await asyncio.start_server(handler, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    try:
        expected = HudProtocolError if protocol_error else EOFError
        message = (
            "frame exceeds size limit" if protocol_error else "closed connection during 'hello'"
        )
        with pytest.raises(expected, match=message):
            async with connect(Runtime(f"tcp://127.0.0.1:{port}"), ready_timeout=1.2):
                pass
    finally:
        server.close()
        await server.wait_closed()


async def test_control_connection_enables_keepalive_with_configured_timings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "control_keepalive_idle", 7)
    monkeypatch.setattr(settings, "control_keepalive_interval", 3)
    monkeypatch.setattr(settings, "control_keepalive_count", 2)
    monkeypatch.setattr(settings, "control_user_timeout", 11.0)
    applied: list[tuple[int, int, int]] = []
    real_setsockopt = socket.socket.setsockopt

    def record_setsockopt(sock: socket.socket, level: int, option: int, value: int) -> None:
        applied.append((level, option, value))
        real_setsockopt(sock, level, option, value)

    monkeypatch.setattr(socket.socket, "setsockopt", record_setsockopt)

    async def handler(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            hello = await read_frame(reader)
            assert hello is not None
            await send_frame(writer, {"jsonrpc": "2.0", "id": hello["id"], "result": HELLO_RESULT})
            await read_frame(reader)
        finally:
            writer.close()

    server = await asyncio.start_server(handler, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    try:
        async with connect(Runtime(f"tcp://127.0.0.1:{port}")):
            pass
    finally:
        server.close()
        await server.wait_closed()

    assert (socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1) in applied
    expected = {
        "TCP_KEEPINTVL": 3,
        "TCP_KEEPCNT": 2,
        "TCP_USER_TIMEOUT": 11_000,
    }
    for name, value in expected.items():
        if hasattr(socket, name):
            assert (socket.IPPROTO_TCP, getattr(socket, name), value) in applied
    assert any(
        value == 7
        for _level, option, value in applied
        if option in (getattr(socket, "TCP_KEEPIDLE", None), getattr(socket, "TCP_KEEPALIVE", None))
    )


async def test_redial_the_env_refuses_to_resume_never_continues_on_a_fresh_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connections: list[list[dict[str, object]]] = []

    async def handler(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        requests: list[dict[str, object]] = []
        connections.append(requests)
        first = len(connections) == 1
        try:
            while (request := await read_frame(reader)) is not None:
                requests.append(request)
                if first and request["method"] == "tasks.grade":
                    return
                if first or request["method"] != "hello":
                    result: dict[str, object] = HELLO_RESULT
                    await send_frame(
                        writer, {"jsonrpc": "2.0", "id": request["id"], "result": result}
                    )
                else:
                    error = {"code": -32600, "message": "unknown session: 's-1'"}
                    await send_frame(
                        writer, {"jsonrpc": "2.0", "id": request["id"], "error": error}
                    )
        finally:
            writer.close()

    monkeypatch.setattr(client_module, "_CONTROL_REDIAL_INTERVAL_SECONDS", 0.01)
    server = await asyncio.start_server(handler, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    try:
        async with connect(Runtime(f"tcp://127.0.0.1:{port}")) as client:
            for _ in range(2):
                with pytest.raises(HudProtocolError, match="unknown session"):
                    await client.grade({"answer": "done"})
    finally:
        server.close()
        await server.wait_closed()

    methods = [request["method"] for requests in connections[1:] for request in requests]
    assert methods
    assert set(methods) == {"hello"}


async def test_a_peer_the_kernel_declares_dead_fails_with_the_typed_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def handler(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            while (request := await read_frame(reader)) is not None:
                await send_frame(
                    writer, {"jsonrpc": "2.0", "id": request["id"], "result": HELLO_RESULT}
                )
        finally:
            writer.close()

    server = await asyncio.start_server(handler, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    monkeypatch.setattr(client_module, "_CONTROL_REDIAL_INTERVAL_SECONDS", 0.01)
    monkeypatch.setattr(settings, "control_outage_budget", 0.05)
    try:
        async with connect(Runtime(f"tcp://127.0.0.1:{port}")) as client:

            async def dead_peer(*_args: object, **_kwargs: object) -> None:
                raise TimeoutError(errno.ETIMEDOUT, "Connection timed out")

            monkeypatch.setattr(client_module, "read_frame", dead_peer)
            with pytest.raises(ControlTimeoutError, match=r"timed out during"):
                await client.grade({"answer": "done"})
    finally:
        server.close()
        await server.wait_closed()
