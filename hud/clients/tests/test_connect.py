"""``connect()`` readiness: the handshake retries until the env actually serves.

A provisioned substrate can sit behind a proxied port (``docker -p``, a
port-forward) that *accepts* TCP before the env behind it is up — those
connections die with EOF at the handshake. Readiness is therefore
protocol-level: ``connect`` keeps retrying through both refused connects and
handshake EOFs until ``hello`` answers or the deadline passes.
"""

from __future__ import annotations

import asyncio
import logging
from typing import ClassVar
from urllib.parse import urlsplit

import pytest

import hud.clients.client as client_module
from hud.capabilities import Capability, CapabilityClient, Connection
from hud.clients import connect
from hud.environment import WorkspaceRoute
from hud.environment.utils import read_frame, send_frame
from hud.eval.runtime import Runtime

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


async def test_tunnel_connection_failure_warns_with_peer(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
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
                                "name": "shell",
                                "protocol": "ssh/2",
                                "url": "ssh://agent@environment:22",
                            }
                        ],
                    },
                },
            )
            await read_frame(reader)
        finally:
            writer.close()

    server = await asyncio.start_server(handler, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    open_connection = asyncio.open_connection

    async def fail_tunnel_connection(host: str, peer_port: int):
        assert (host, peer_port) == ("127.0.0.1", port)
        raise OSError("peer unavailable")

    try:
        async with connect(Runtime(f"tcp://127.0.0.1:{port}")) as client:
            route = urlsplit(client.binding("shell").url)
            monkeypatch.setattr(client_module.asyncio, "open_connection", fail_tunnel_connection)
            caplog.set_level(logging.WARNING, logger="hud.clients")

            reader, writer = await open_connection(route.hostname, route.port)
            try:
                assert await reader.read() == b""
            finally:
                writer.close()
                await writer.wait_closed()
    finally:
        server.close()
        await server.wait_closed()

    assert f"tunnel peer 127.0.0.1:{port} connection failed: peer unavailable" in caplog.messages


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


async def test_heartbeat_transport_failure_interrupts_the_connection_body(
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
    monkeypatch.setattr(client_module, "_CONTROL_HEARTBEAT_TIMEOUT_SECONDS", 0.01)
    server = await asyncio.start_server(handler, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    try:
        with pytest.raises(TimeoutError):
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
