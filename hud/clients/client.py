"""HudClient: JSON-RPC client for the HUD wire protocol.

Transport for a served env's control channel: drives ``hello`` / ``tasks.*`` /
``bye`` and exposes capabilities via ``binding(ref)`` (wire data) /
``open(ref)`` (live client). Use the module-level ``connect(runtime)`` to
attach to a provisioned substrate.
"""

from __future__ import annotations

import asyncio
import contextlib
import itertools
import logging
import math
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, replace
from functools import partial
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit, urlunsplit

from hud.capabilities import (
    Capability,
    CapabilityClient,
    CDPClient,
    Connection,
    MCPClient,
    RFBClient,
    SSHClient,
)
from hud.environment.utils import (
    CONTROL_FRAME_LIMIT_BYTES,
    encode_frame,
    read_frame,
    send_frame,
    splice,
)
from hud.settings import settings

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Sequence

    from hud.environment.egress import WorkspaceRoute
    from hud.eval.runtime import Runtime

LOGGER = logging.getLogger("hud.clients")
_ABANDON_REPLY_TIMEOUT_SECONDS = 2.0
_CAPABILITY_CONNECT_ATTEMPTS = 3
_RETRY_BASE_DELAY_SECONDS = 0.25
_RETRY_MAX_DELAY_SECONDS = 10.0
#: How long a client waits for a freshly provisioned env to serve when nothing sets a limit.
DEFAULT_READY_TIMEOUT = 240.0

#: protocol -> CapabilityClient subclass, for ``HudClient.open``.
_CLIENT_REGISTRY: dict[str, type[CapabilityClient]] = {
    cls.protocol: cls for cls in (SSHClient, RFBClient, MCPClient, CDPClient)
}


def _retry_delay(attempt: int) -> float:
    return min(_RETRY_BASE_DELAY_SECONDS * 2**attempt, _RETRY_MAX_DELAY_SECONDS)


async def _connect_capability(
    client_cls: type[CapabilityClient],
    capability: Capability,
) -> CapabilityClient:
    for attempt in range(_CAPABILITY_CONNECT_ATTEMPTS):
        try:
            return await client_cls.connect(capability)
        except ConnectionError:
            if attempt + 1 == _CAPABILITY_CONNECT_ATTEMPTS:
                raise
            await asyncio.sleep(_retry_delay(attempt))
    raise RuntimeError("capability connect attempts must be positive")


class ControlTimeoutError(TimeoutError):
    """Raised when the env does not answer a control request within its liveness deadline."""


class HudProtocolError(RuntimeError):
    """Raised when the env returns a JSON-RPC error frame."""

    def __init__(self, code: int, message: str) -> None:
        super().__init__(f"hud rpc error {code}: {message}")
        self.code = code
        self.message = message


async def _dial_tunnel(
    host: str, port: int, capability: str
) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
    async with asyncio.timeout(settings.control_heartbeat_timeout):
        reader, writer = await asyncio.open_connection(host, port)
        try:
            await send_frame(
                writer,
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "tunnel.open",
                    "params": {"capability": capability},
                },
                max_bytes=CONTROL_FRAME_LIMIT_BYTES,
            )
            opened = await read_frame(reader, max_bytes=CONTROL_FRAME_LIMIT_BYTES)
            if opened is None:
                raise EOFError("env closed the connection during 'tunnel.open'")
            if "error" in opened:
                err = opened["error"]
                raise HudProtocolError(int(err.get("code", -32000)), str(err.get("message", "")))
        except BaseException:
            writer.close()
            raise
    return reader, writer


async def _open_tunnel(
    host: str, port: int, capability: str
) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
    """Open one capability stream, redialing for up to ``control_outage_budget``.

    Nothing of the local connection is read until the env confirms the stream, so a
    failed attempt never delivered a byte and redialing is safe. An error reply from the
    env is final.
    """
    loop = asyncio.get_running_loop()
    deadline = loop.time() + settings.control_outage_budget
    attempt = 0
    while True:
        try:
            return await _dial_tunnel(host, port, capability)
        except (OSError, EOFError, TimeoutError) as exc:
            if loop.time() >= deadline:
                raise
            LOGGER.warning("tunnel %r to %s:%d not open, retrying: %r", capability, host, port, exc)
        await asyncio.sleep(_retry_delay(attempt))
        attempt += 1


@dataclass(frozen=True, slots=True)
class ServerInfo:
    """Identity of the env serving this session (for compatibility / observability)."""

    name: str
    version: str


@dataclass(frozen=True, slots=True)
class Manifest:
    """Env welcome frame returned by ``HudClient.hello()``.

    ``bindings`` are the environment-native addresses published by the
    substrate. Agents executing inside that environment consume these raw
    endpoints; controller-side consumers use :meth:`HudClient.binding` or
    :meth:`HudClient.open`, which route them through the control channel.
    """

    session_id: str
    protocol_version: str  # e.g. "hud/1.0"
    server_info: ServerInfo
    bindings: list[Capability]


class HudClient:
    """JSON-RPC client for a served env's control channel.

    Prefer ``hud.connect(runtime)``, which owns the lifecycle (connect →
    ``hello`` → yield → ``close``) and yields one of these with ``manifest``
    ready; the raw constructor takes any connected stream pair. Task lifecycle
    wrapping (start → grade) lives in :class:`hud.eval.Run`.
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
        #: Control-channel (host, port), for tunnel connections. ``None`` for
        #: raw stream pairs (no dialable endpoint): bindings pass through.
        self._endpoint = endpoint
        self._ids = itertools.count(1)
        self._call_lock = asyncio.Lock()
        self._closed = False
        self._aborted = False
        #: The control connection failed; the next resuming call redials first.
        self._down = False
        #: ``hello`` params that attach a connection to this session (routes, connections).
        self._attach: dict[str, Any] = {}
        self.manifest: Manifest | None = None
        self._routes: dict[str, str] = {}
        self._opened: dict[str, CapabilityClient] = {}
        self._forwarders: list[asyncio.Server] = []
        self._tunnels: set[asyncio.Task[None]] = set()

    # ─── lifecycle ────────────────────────────────────────────────────

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        opened = list(self._opened.values())
        tunnels = list(self._tunnels)
        for cap_client in opened:
            with contextlib.suppress(Exception):
                await cap_client.close()
        self.abort()
        if tunnels:
            await asyncio.gather(*tunnels, return_exceptions=True)
        # No `bye`: a plain disconnect leaves the env's held session for a later
        # connection to grade; `grade` itself clears the session when it completes.
        self._writer.close()
        with contextlib.suppress(Exception):
            await self._writer.wait_closed()

    def abort(self) -> None:
        """Close the transport immediately without waiting for the peer."""
        self._aborted = True
        for forwarder in self._forwarders:
            forwarder.close()
        for tunnel in list(self._tunnels):
            tunnel.cancel()
        self._writer.close()

    # ─── handshake ────────────────────────────────────────────────────

    async def hello(
        self,
        session_id: str | None = None,
        *,
        workspace_routes: Sequence[WorkspaceRoute] = (),
        connections: Sequence[Connection] = (),
    ) -> Manifest:
        """Send ``hello``; cache and return the parsed ``Manifest``.

        ``session_id`` resumes that parked session on the env — its suspended
        task, e.g. one a prior connection started — instead of minting a
        fresh session.
        """
        self._attach = {}
        if workspace_routes:
            self._attach["workspace_routes"] = [route.to_wire() for route in workspace_routes]
        if connections:
            self._attach["connections"] = [connection.to_wire() for connection in connections]
        params = dict(self._attach)
        if session_id is not None:
            params["session_id"] = session_id
        result = await self._call("hello", params)
        env = result["env"]
        bindings = [Capability.from_manifest(binding) for binding in result["bindings"]]
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
                        up_reader, up_writer = await _open_tunnel(host, port, capability.name)
                    except (OSError, EOFError, TimeoutError, HudProtocolError) as exc:
                        LOGGER.warning(
                            "tunnel %r to %s:%d failed: %r", capability.name, host, port, exc
                        )
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

    # ─── capability access ────────────────────────────────────────────
    #
    # ``binding`` and ``open`` look up the same capability by name or protocol;
    # they differ only in what they hand back:
    #   binding(ref) -> Capability         wire data (url/params; BYO conn)
    #   open(ref)    -> CapabilityClient   live, connected, cached client

    def binding(self, ref: str) -> Capability:
        """Find the capability matching *ref* (name, protocol family, or protocol).

        Returns the wire data — use this when something else owns the
        connection (e.g. browser-use reads the CDP url). Ambiguous refs
        (multiple matches) raise; use names to disambiguate.
        """
        if self.manifest is None:
            raise RuntimeError("call hello() before accessing bindings")
        matches = [
            capability
            for capability in self.manifest.bindings
            if ref
            in (
                capability.name,
                capability.protocol,
                capability.protocol.split("/", 1)[0],
            )
        ]
        if len(matches) == 1:
            capability = matches[0]
            if capability.name in self._routes:
                return replace(capability, url=self._routes[capability.name])
            return capability
        if len(matches) > 1:
            names = ", ".join(f"{c.name} ({c.protocol})" for c in matches)
            raise KeyError(f"ambiguous capability {ref!r}; matches: {names}")
        available = ", ".join(f"{c.name} ({c.protocol})" for c in self.manifest.bindings)
        raise KeyError(f"no capability {ref!r} (available: {available or '<none>'})")

    async def open(self, ref: str) -> CapabilityClient:
        """Open (and cache) a live ``CapabilityClient`` for a capability.

        Resolves like ``binding`` but connects and returns a live client, owned
        by this connection and closed on ``close()``.
        """
        cap = self.binding(ref)
        cap_client = self._opened.get(cap.name)
        if cap_client is None:
            client_cls = _CLIENT_REGISTRY.get(cap.protocol)
            if client_cls is None and cap.protocol.split("/", 1)[0] == "openpi":
                # RobotClient pulls optional deps (numpy/openpi-client — the ``robot``
                # extra), so it joins the registry on first open, not at import.
                from hud.capabilities.robot import RobotClient

                client_cls = _CLIENT_REGISTRY[RobotClient.protocol] = RobotClient
            if client_cls is None:
                raise ValueError(
                    f"no client registered for protocol {cap.protocol!r}; "
                    f"use binding({ref!r}) for raw access",
                )
            cap_client = await _connect_capability(client_cls, cap)
            self._opened[cap.name] = cap_client
        return cap_client

    # ─── tasks ────────────────────────────────────────────────────────

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
        """Start a task; returns the first yield (``{"prompt": ...}``).

        Survives a dropped control connection: the call is re-sent on the redialed
        connection under the same ``call_id``, so the env joins the start it already
        has instead of starting the task again.
        """
        return await self._call_resuming(
            "tasks.start", {"id": task_id, "args": args or {}, "call_id": uuid.uuid4().hex}
        )

    async def grade(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Send ``tasks.grade``; returns the evaluation dict (``{"score": ...}``).

        Survives a dropped control connection: the env keeps the grade it is running
        for the session, so the re-sent call collects it instead of grading twice.
        """
        return await self._call_resuming("tasks.grade", payload)

    async def heartbeat(self) -> None:
        """Re-send ``hello`` for this session, redialing the control connection if it is down."""
        assert self.manifest is not None
        await self._call_resuming(
            "hello",
            {**self._attach, "session_id": self.manifest.session_id},
            reply_timeout=settings.control_heartbeat_timeout,
        )

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

    # ─── JSON-RPC plumbing ────────────────────────────────────────────

    async def _call(
        self,
        method: str,
        params: dict[str, Any],
        *,
        reply_timeout: float | None = None,
    ) -> dict[str, Any]:
        async with self._call_lock:
            return await self._exchange(method, params, reply_timeout)

    async def _call_resuming(
        self,
        method: str,
        params: dict[str, Any],
        *,
        reply_timeout: float | None = None,
    ) -> dict[str, Any]:
        """Call a method the env answers idempotently for a session, across lost connections.

        A failed or silent control connection is replaced by a redialed one that resumes
        the session, and the call is sent again. Gives up with the last error once the
        control channel has been down for ``control_outage_budget``.
        """
        loop = asyncio.get_running_loop()
        outage_started: float | None = None
        attempt = 0
        async with self._call_lock:
            while True:
                try:
                    if self._down:
                        await self._redial()
                    result = await self._exchange(method, params, reply_timeout)
                except (EOFError, OSError, TimeoutError) as exc:
                    now = loop.time()
                    outage_started = now if outage_started is None else outage_started
                    if (
                        self._aborted
                        or self._endpoint is None
                        or self.manifest is None
                        or now - outage_started >= settings.control_outage_budget
                    ):
                        raise
                    LOGGER.warning(
                        "control link down for %.0fs during %r, redialing: %r",
                        now - outage_started,
                        method,
                        exc,
                    )
                else:
                    if outage_started is not None:
                        LOGGER.warning(
                            "control link restored after %.0fs", loop.time() - outage_started
                        )
                    return result
                await asyncio.sleep(_retry_delay(attempt))
                attempt += 1

    async def _redial(self) -> None:
        """Open a fresh control connection and resume this session on it.

        Only the control connection is replaced: forwarders and tunnels dial the env
        themselves, and the env keeps the session's task parked across the drop.
        ``takeover`` tells the env to drop the old connection, which it may not have
        noticed is dead.
        """
        assert self._endpoint is not None and self.manifest is not None
        timeout = settings.control_heartbeat_timeout
        try:
            self._reader, self._writer = await asyncio.wait_for(
                asyncio.open_connection(*self._endpoint), timeout
            )
        except TimeoutError:
            raise ControlTimeoutError(
                f"env did not accept a control connection within {timeout:g}s"
            ) from None
        try:
            await self._exchange(
                "hello",
                {**self._attach, "session_id": self.manifest.session_id, "takeover": True},
                timeout,
            )
        except HudProtocolError:
            self._writer.close()
            raise
        self._down = False

    async def _exchange(
        self,
        method: str,
        params: dict[str, Any],
        reply_timeout: float | None,
    ) -> dict[str, Any]:
        msg_id = next(self._ids)
        frame = encode_frame(
            {"jsonrpc": "2.0", "id": msg_id, "method": method, "params": params},
            max_bytes=CONTROL_FRAME_LIMIT_BYTES,
        )
        deadline = asyncio.timeout(reply_timeout)
        try:
            async with deadline:
                self._writer.write(frame)
                await self._writer.drain()
                reply = await read_frame(self._reader, max_bytes=CONTROL_FRAME_LIMIT_BYTES)
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
        except BaseException as exc:
            self._down = True
            self._writer.close()
            if isinstance(exc, TimeoutError) and deadline.expired():
                raise ControlTimeoutError(
                    f"env did not reply to {method!r} within {reply_timeout:g}s"
                ) from None
            raise


# ─── module-level entry points ────────────────────────────────────────


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

    Readiness is protocol-level, and the client owns waiting for it: a
    freshly-provisioned substrate may refuse the connect, and a proxied port
    (``docker -p``, a port-forward) can *accept* before the env behind it is
    serving — that connection just dies at the handshake. Both mean
    not-ready-yet. Returns a client whose ``manifest`` is populated.
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
            # The accepted connection had no live env behind it: EOF on the
            # reply, or a reset racing our hello write. Still not-ready.
            await client.close()
            if loop.time() >= deadline:
                raise
            await asyncio.sleep(interval)
        except BaseException:
            # Real failure (error frame, cancellation): don't leak the socket —
            # an unclosed connection parks the env's connection handler.
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
    """Connect a :class:`HudClient` to a provisioned substrate's control channel.

    Takes the :class:`~hud.eval.runtime.Runtime` a provider yielded (or
    one constructed directly for a substrate served elsewhere) and retries
    connect + handshake until the channel answers. Does not tear the substrate
    down — lifecycle belongs to whichever provider brought it up.
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
            await asyncio.sleep(settings.control_heartbeat_interval)
            try:
                await client.heartbeat()
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


__all__ = [
    "DEFAULT_READY_TIMEOUT",
    "ControlTimeoutError",
    "HudClient",
    "HudProtocolError",
    "Manifest",
    "ServerInfo",
    "connect",
]
