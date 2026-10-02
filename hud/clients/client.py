"""HudClient: JSON-RPC client for the HUD wire protocol."""

from __future__ import annotations

import asyncio
import contextlib
import itertools
import logging
import math
from contextlib import asynccontextmanager
from dataclasses import dataclass, replace
from functools import partial
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit, urlunsplit

from hud.capabilities import Capability, Connection
from hud.environment.utils import (
    CONTROL_FRAME_LIMIT_BYTES,
    FILE_TRACKING_FRAME_LIMIT_BYTES,
    encode_frame,
    read_frame,
    send_frame,
    splice,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Sequence

    from hud.environment.egress import WorkspaceRoute
    from hud.environment.file_tracker import FileTrackingOp
    from hud.runtime import Runtime

LOGGER = logging.getLogger("hud.clients")
_CONTROL_HEARTBEAT_INTERVAL_SECONDS = 120.0
_CONTROL_HEARTBEAT_TIMEOUT_SECONDS = 5.0
_ABANDON_REPLY_TIMEOUT_SECONDS = 2.0
# How long a client waits for a freshly provisioned env to serve when nothing sets a limit.
DEFAULT_READY_TIMEOUT = 240.0
_LEGACY_FILE_TRACKING = "filetracking/1"
_METHOD_NOT_FOUND = -32601


class HudProtocolError(RuntimeError):
    """Raised when the env returns a JSON-RPC error frame."""

    def __init__(self, code: int, message: str) -> None:
        super().__init__(f"hud rpc error {code}: {message}")
        self.code = code
        self.message = message


@dataclass(frozen=True, slots=True)
class ServerInfo:
    """Identity of the env serving this session (for compatibility / observability)."""

    name: str
    version: str


@dataclass(frozen=True, slots=True)
class Manifest:
    """Env welcome frame returned by ``HudClient.hello()``.

    ``bindings`` hold environment-native addresses. From outside the environment, use
    ``HudClient.binding`` or ``HudClient.open``, which route through the control channel.
    """

    session_id: str
    protocol_version: str  # e.g. "hud/1.0"
    server_info: ServerInfo
    bindings: list[Capability]


class HudClient:
    """The session of a served environment: JSON-RPC over its control channel.

    Prefer ``connect``, which yields a client with ``manifest`` set and closes it.
    """

    PROTOCOL_VERSION = "hud/1.0"

    def __init__(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
        *,
        endpoint: tuple[str, int] | None = None,
    ) -> None:
        self._reader = reader
        self._writer = writer
        # Dialed for capability tunnels; without one, bindings pass through unrouted.
        self._endpoint = endpoint
        self._ids = itertools.count(1)
        self._call_lock = asyncio.Lock()
        self._closed = False
        self.manifest: Manifest | None = None
        self._legacy_files: _LegacyFileTracking | None = None
        self._routes: dict[str, str] = {}
        self._forwarders: list[asyncio.Server] = []
        self._tunnels: set[asyncio.Task[None]] = set()

    @property
    def capabilities(self) -> list[Capability]:
        """The capabilities the environment published, at its own addresses."""
        if self.manifest is None:
            raise RuntimeError("call hello() before accessing capabilities")
        return list(self.manifest.bindings)

    def route(self, capability: Capability) -> Capability:
        """``capability`` at the local address that tunnels to it, when there is one."""
        url = self._routes.get(capability.name)
        return capability if url is None else replace(capability, url=url)

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        tunnels = list(self._tunnels)
        self.abort()
        if tunnels:
            await asyncio.gather(*tunnels, return_exceptions=True)
        # No `bye`: a plain disconnect leaves the env's held session for a later
        # connection to grade; `grade` itself clears the session when it completes.
        self._writer.close()
        with contextlib.suppress(Exception):
            await self._writer.wait_closed()

    async def revoke(self) -> None:
        """Cut the routes to the environment's capabilities; the control channel stays up."""
        self._cut_routes()

    def abort(self) -> None:
        """Close the transport immediately without waiting for the peer."""
        self._cut_routes()
        if self._legacy_files is not None:
            self._legacy_files.close()
        self._writer.close()

    def _cut_routes(self) -> None:
        for forwarder in self._forwarders:
            forwarder.close()
        for tunnel in list(self._tunnels):
            tunnel.cancel()

    async def hello(
        self,
        session_id: str | None = None,
        *,
        workspace_routes: Sequence[WorkspaceRoute] = (),
        connections: Sequence[Connection] = (),
    ) -> Manifest:
        """Send ``hello``; cache and return the parsed ``Manifest``.

        ``session_id`` resumes that parked session, with its suspended task, instead
        of minting a fresh one.
        """
        params: dict[str, Any] = {}
        if workspace_routes:
            params["workspace_routes"] = [route.to_wire() for route in workspace_routes]
        if connections:
            params["connections"] = [connection.to_wire() for connection in connections]
        if session_id is not None:
            params["session_id"] = session_id
        result = await self._call("hello", params)
        env = result["env"]
        bindings = [Capability.from_manifest(binding) for binding in result["bindings"]]
        legacy = [b for b in bindings if b.protocol == _LEGACY_FILE_TRACKING]
        if legacy:
            bindings = [b for b in bindings if b.protocol != _LEGACY_FILE_TRACKING]
            if self._legacy_files is None:
                self._legacy_files = _LegacyFileTracking(legacy[0], self._endpoint)
        self.manifest = Manifest(
            session_id=result["session_id"],
            protocol_version=self.PROTOCOL_VERSION,
            server_info=ServerInfo(
                name=env["name"],
                version=env["version"],
            ),
            bindings=bindings,
        )
        self._routes.clear()
        if self._endpoint is not None:
            host, port = self._endpoint

            async def forward(
                capability: Capability,
                reader: asyncio.StreamReader,
                writer: asyncio.StreamWriter,
            ) -> None:
                task = asyncio.current_task()
                assert task is not None
                self._tunnels.add(task)
                try:
                    try:
                        up_reader, up_writer = await asyncio.open_connection(host, port)
                    except OSError as exc:
                        LOGGER.warning(
                            "tunnel peer %s:%d connection failed: %s",
                            host,
                            port,
                            exc,
                        )
                        writer.close()
                        return
                    await send_frame(
                        up_writer,
                        {
                            "jsonrpc": "2.0",
                            "id": 1,
                            "method": "tunnel.open",
                            "params": {"capability": capability.name},
                        },
                        max_bytes=CONTROL_FRAME_LIMIT_BYTES,
                    )
                    opened = await read_frame(up_reader, max_bytes=CONTROL_FRAME_LIMIT_BYTES)
                    if opened is None or "error" in opened:
                        LOGGER.warning("tunnel.open %r refused: %s", capability.name, opened)
                        up_writer.close()
                        writer.close()
                        return
                    await splice((reader, writer), (up_reader, up_writer))
                finally:
                    self._tunnels.discard(task)

            for capability in bindings:
                forwarder = await asyncio.start_server(partial(forward, capability), "127.0.0.1", 0)
                self._forwarders.append(forwarder)
                local_port = forwarder.sockets[0].getsockname()[1]
                parts = urlsplit(capability.url)
                userinfo = f"{parts.username}@" if parts.username else ""
                self._routes[capability.name] = urlunsplit(
                    (
                        parts.scheme,
                        f"{userinfo}127.0.0.1:{local_port}",
                        parts.path,
                        parts.query,
                        parts.fragment,
                    )
                )
        return self.manifest

    async def list_tasks(self) -> list[dict[str, Any]]:
        """Return ``[{id, description}, ...]`` for every registered task."""
        result = await self._call("tasks.list", {})
        tasks = result.get("tasks") or []
        if not isinstance(tasks, list):
            raise HudProtocolError(-32603, "tasks.list: 'tasks' must be a list")
        return tasks

    async def start_task(
        self,
        task_id: str,
        args: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Start a task; returns the first yield (``{"prompt": ...}``)."""
        return await self._call("tasks.start", {"id": task_id, "args": args or {}})

    async def grade(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Send ``tasks.grade``; returns the evaluation dict (``{"score": ...}``)."""
        return await self._call("tasks.grade", payload)

    async def cancel(self) -> None:
        await self._call("tasks.cancel", {})

    async def _abandon_task(self) -> None:
        """Cancel the task, even mid-call, wait briefly for its teardown, then abort."""
        frame = encode_frame(
            {"jsonrpc": "2.0", "id": next(self._ids), "method": "tasks.cancel", "params": {}},
            max_bytes=CONTROL_FRAME_LIMIT_BYTES,
        )
        try:
            with contextlib.suppress(Exception):
                async with asyncio.timeout(_ABANDON_REPLY_TIMEOUT_SECONDS):
                    self._writer.write(frame)
                    await self._writer.drain()
                    async with self._call_lock:
                        await read_frame(self._reader, max_bytes=CONTROL_FRAME_LIMIT_BYTES)
        finally:
            self.abort()

    async def files(self, op: FileTrackingOp) -> dict[str, Any] | None:
        """Run one file-tracking operation; ``None`` when the environment tracks no files."""
        if self._legacy_files is not None:
            return await self._legacy_files.call(op)
        try:
            return await self._call(
                f"files.{op}", {}, reply_limit_bytes=FILE_TRACKING_FRAME_LIMIT_BYTES
            )
        except HudProtocolError as exc:
            if exc.code == _METHOD_NOT_FOUND:
                return None
            raise

    async def _call(
        self,
        method: str,
        params: dict[str, Any],
        *,
        reply_timeout: float | None = None,
        reply_limit_bytes: int = CONTROL_FRAME_LIMIT_BYTES,
    ) -> dict[str, Any]:
        async with self._call_lock:
            msg_id = next(self._ids)
            frame = encode_frame(
                {"jsonrpc": "2.0", "id": msg_id, "method": method, "params": params},
                max_bytes=CONTROL_FRAME_LIMIT_BYTES,
            )
            try:
                async with asyncio.timeout(reply_timeout):
                    self._writer.write(frame)
                    await self._writer.drain()
                    reply = await read_frame(self._reader, max_bytes=reply_limit_bytes)
                    if reply is None:
                        raise EOFError(f"env closed connection during {method!r}")
                    if reply.get("id") != msg_id:
                        self.abort()
                        if reply.get("id") is not None or "error" not in reply:
                            raise HudProtocolError(
                                -32603,
                                f"{method!r}: reply id did not match request",
                            )
                    if "error" in reply:
                        err = reply["error"]
                        raise HudProtocolError(
                            int(err.get("code", -32000)),
                            str(err.get("message", "")),
                        )
                    result = reply.get("result")
                    if not isinstance(result, dict):
                        raise HudProtocolError(-32603, f"{method!r}: result was not an object")
                    return result
            except HudProtocolError:
                raise
            except BaseException:
                self.abort()
                raise


class _LegacyFileTracking:
    """``files`` against images that publish ``filetracking/1`` instead of serving ``files.*``.

    Dialed over a private tunnel, so revoking the agent's routes leaves it up. Remove once
    images built before the control channel served ``files.*`` are rebuilt.
    """

    def __init__(self, capability: Capability, endpoint: tuple[str, int] | None) -> None:
        self._capability = capability
        self._endpoint = endpoint
        self._ids = itertools.count(1)
        self._lock = asyncio.Lock()
        self._stream: tuple[asyncio.StreamReader, asyncio.StreamWriter] | None = None

    async def call(self, op: FileTrackingOp) -> dict[str, Any]:
        method: str = op
        if op == "setup" and self._capability.params.get("setup_diff") is not True:
            method = "advance"
        async with self._lock:
            reader, writer = self._stream or await self._open()
            await send_frame(
                writer, {"jsonrpc": "2.0", "id": next(self._ids), "method": method, "params": {}}
            )
            reply = await read_frame(reader, max_bytes=FILE_TRACKING_FRAME_LIMIT_BYTES)
        if reply is None:
            raise EOFError(f"file tracking closed during {method!r}")
        if "error" in reply:
            raise HudProtocolError(-32000, str(reply["error"].get("message", "")))
        # Servers that cannot report setup changes only advance past them.
        return {} if method == "advance" else reply["result"]

    async def _open(self) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
        if self._endpoint is None:
            parts = urlsplit(self._capability.url)
            assert parts.hostname is not None and parts.port is not None
            stream = await asyncio.open_connection(parts.hostname, parts.port)
        else:
            stream = await asyncio.open_connection(*self._endpoint)
            await send_frame(
                stream[1],
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "tunnel.open",
                    "params": {"capability": self._capability.name},
                },
                max_bytes=CONTROL_FRAME_LIMIT_BYTES,
            )
            opened = await read_frame(stream[0], max_bytes=CONTROL_FRAME_LIMIT_BYTES)
            if opened is None or "error" in opened:
                stream[1].close()
                raise ConnectionError(f"file tracking tunnel refused: {opened}")
        self._stream = stream
        return stream

    def close(self) -> None:
        if self._stream is not None:
            self._stream[1].close()
            self._stream = None


async def _connect_ready(
    host: str,
    port: int,
    *,
    ready_timeout: float,
    workspace_routes: Sequence[WorkspaceRoute],
    connections: Sequence[Connection],
    interval: float = 0.5,
) -> HudClient:
    """Connect and complete ``hello``, retrying until the env is ready.

    A proxied port (``docker -p``, a port-forward) can accept before the env behind it
    serves, so a handshake EOF means not ready yet, like a refused connect.
    """
    loop = asyncio.get_event_loop()
    deadline = loop.time() + ready_timeout
    while True:
        try:
            reader, writer = await asyncio.open_connection(host, port)
        except OSError:
            if loop.time() >= deadline:
                raise
            await asyncio.sleep(interval)
            continue

        client = HudClient(reader, writer, endpoint=(host, port))
        try:
            await client.hello(workspace_routes=workspace_routes, connections=connections)
        except asyncio.CancelledError:
            client.abort()
            raise
        except (EOFError, OSError):
            # No live env behind the accepted connection yet: EOF on the reply, or a
            # reset racing the hello write.
            await client.close()
            if loop.time() >= deadline:
                raise
            await asyncio.sleep(interval)
        except BaseException:
            # A real failure still closes the socket: an unclosed connection parks the
            # env's connection handler.
            await client.close()
            raise
        else:
            return client


def _runtime_ready_timeout(runtime: Runtime, default: float) -> float:
    raw = runtime.params.get("ready_timeout")
    if raw is None:
        return default
    if isinstance(raw, bool) or not isinstance(raw, int | float):
        raise ValueError("runtime.params['ready_timeout'] must be a positive finite number")
    timeout = float(raw)
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("runtime.params['ready_timeout'] must be a positive finite number")
    return timeout


@asynccontextmanager
async def connect(
    runtime: Runtime,
    *,
    ready_timeout: float = DEFAULT_READY_TIMEOUT,
    workspace_routes: Sequence[WorkspaceRoute] = (),
    connections: Sequence[Connection] = (),
) -> AsyncIterator[HudClient]:
    """Connect a ``HudClient`` to a runtime's control channel, retrying until it answers.

    The substrate's teardown stays with its provider. While open, a periodic heartbeat
    probes the transport; a transport failure cancels the body and raises that error.
    """
    parts = urlsplit(runtime.url)
    if parts.scheme not in ("", "tcp"):
        raise NotImplementedError(
            f"control transport {parts.scheme!r} not supported yet (only tcp://)",
        )
    client = await _connect_ready(
        parts.hostname or "127.0.0.1",
        parts.port or 0,
        ready_timeout=_runtime_ready_timeout(runtime, ready_timeout),
        workspace_routes=workspace_routes,
        connections=connections,
    )
    owner = asyncio.current_task()
    assert owner is not None
    heartbeat_error: Exception | None = None

    async def heartbeat() -> None:
        nonlocal heartbeat_error
        while True:
            await asyncio.sleep(_CONTROL_HEARTBEAT_INTERVAL_SECONDS)
            assert client.manifest is not None
            try:
                params: dict[str, Any] = {"session_id": client.manifest.session_id}
                if workspace_routes:
                    params["workspace_routes"] = [route.to_wire() for route in workspace_routes]
                if connections:
                    params["connections"] = [connection.to_wire() for connection in connections]
                await client._call(
                    "hello",
                    params,
                    reply_timeout=_CONTROL_HEARTBEAT_TIMEOUT_SECONDS,
                )
            except HudProtocolError as exc:
                LOGGER.warning("control heartbeat failed: %s", exc)
                return
            except Exception as exc:
                LOGGER.warning("control heartbeat failed: %s", exc)
                heartbeat_error = exc
                owner.cancel()
                return

    heartbeat_task = asyncio.create_task(heartbeat(), name="hud-control-heartbeat")
    try:
        try:
            yield client
        finally:
            heartbeat_task.cancel()
            await asyncio.gather(heartbeat_task, return_exceptions=True)
            await client.close()
    except asyncio.CancelledError:
        if heartbeat_error is not None:
            raise heartbeat_error from None
        raise
