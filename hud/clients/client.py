"""HudClient: JSON-RPC client for the HUD wire protocol."""

from __future__ import annotations

import asyncio
import contextlib
import itertools
import logging
from dataclasses import dataclass, replace
from functools import partial
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit, urlunsplit

from hud.capabilities import Capability, Connection
from hud.environment.utils import (
    CONTROL_FRAME_LIMIT_BYTES,
    FILE_TRACKING_FRAME_LIMIT_BYTES,
    HudProtocolError,
    encode_frame,
    read_frame,
    send_frame,
    splice,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from hud.environment.egress import WorkspaceRoute
    from hud.environment.file_tracker import FileTrackingOp

LOGGER = logging.getLogger("hud.clients")
_LEGACY_FILE_TRACKING = "filetracking/1"
_METHOD_NOT_FOUND = -32601


@dataclass(frozen=True, slots=True)
class ServerInfo:
    """Identity of the env serving this session (for compatibility / observability)."""

    name: str
    version: str


@dataclass(frozen=True, slots=True)
class Manifest:
    """Env welcome frame returned by ``HudClient.hello()``.

    ``bindings`` hold environment-native addresses; ``HudClient.capabilities`` holds them
    routed through the control channel.
    """

    session_id: str
    protocol_version: str  # e.g. "hud/1.0"
    server_info: ServerInfo
    bindings: list[Capability]


class HudClient:
    """The session of a served environment: JSON-RPC over its control channel.

    Prefer ``Runtime.connect``, which yields a client with ``manifest`` set and closes it.
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
        # The routes and connections ``hello`` bound, which ``ping`` re-sends.
        self._bound: dict[str, Any] = {}
        self._legacy_files: _LegacyFileTracking | None = None
        self._routes: dict[str, str] = {}
        self._forwarders: list[asyncio.Server] = []
        self._tunnels: set[asyncio.Task[None]] = set()

    @property
    def session_id(self) -> str:
        """The environment's id for this session."""
        if self.manifest is None:
            raise RuntimeError("call hello() before accessing session_id")
        return self.manifest.session_id

    @property
    def capabilities(self) -> list[Capability]:
        """The environment's capabilities at the local addresses that tunnel to them.

        ``Capability.env_url`` keeps each one's address inside the environment.
        """
        if self.manifest is None:
            raise RuntimeError("call hello() before accessing capabilities")
        return [
            replace(cap, url=self._routes[cap.name]) if cap.name in self._routes else cap
            for cap in self.manifest.bindings
        ]

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        tunnels = list(self._tunnels)
        self.abort()
        if tunnels:
            await asyncio.gather(*tunnels, return_exceptions=True)
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
        lease: float | None = None,
        workspace_routes: Sequence[WorkspaceRoute] = (),
        connections: Sequence[Connection] = (),
    ) -> Manifest:
        """Send ``hello``; cache and return the parsed ``Manifest``.

        ``session_id`` attaches to that open session, with any task it holds, instead
        of opening a fresh one; it must be the connection's first ``hello``. ``lease``
        keeps the session open that many seconds after this connection ends; by
        default it closes with the connection, cancelling its task.
        """
        self._bound = {}
        if lease is not None:
            self._bound["lease"] = lease
        if workspace_routes:
            self._bound["workspace_routes"] = [route.to_wire() for route in workspace_routes]
        if connections:
            self._bound["connections"] = [connection.to_wire() for connection in connections]
        params = dict(self._bound)
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

    async def ping(self, *, reply_timeout: float) -> None:
        """Re-send this session's ``hello`` to probe the connection."""
        await self._call(
            "hello", {"session_id": self.session_id, **self._bound}, reply_timeout=reply_timeout
        )

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
