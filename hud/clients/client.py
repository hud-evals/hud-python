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
from contextlib import asynccontextmanager
from dataclasses import dataclass, replace
from functools import partial
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit, urlunsplit

from hud.capabilities import (
    Capability,
    CapabilityClient,
    CDPClient,
    MCPClient,
    RFBClient,
    SSHClient,
)
from hud.environment.platform_inference import InferenceBinding
from hud.environment.utils import read_frame, send_frame, splice

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from hud.eval.runtime import Runtime

LOGGER = logging.getLogger("hud.clients")
_CONTROL_HEARTBEAT_INTERVAL_SECONDS = 120.0
_CONTROL_HEARTBEAT_TIMEOUT_SECONDS = 5.0
_CAPABILITY_CONNECT_ATTEMPTS = 3
_CAPABILITY_CONNECT_BASE_DELAY_SECONDS = 0.25

#: protocol -> CapabilityClient subclass, for ``HudClient.open``.
_CLIENT_REGISTRY: dict[str, type[CapabilityClient]] = {
    cls.protocol: cls for cls in (SSHClient, RFBClient, MCPClient, CDPClient)
}


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
            await asyncio.sleep(_CAPABILITY_CONNECT_BASE_DELAY_SECONDS * 2**attempt)
    raise RuntimeError("capability connect attempts must be positive")


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
        self.manifest: Manifest | None = None
        self._routes: dict[str, str] = {}
        self._opened: dict[str, CapabilityClient] = {}
        self._forwarders: list[asyncio.Server] = []
        self._tunnels: set[asyncio.Task[None]] = set()
        self.inference: InferenceBinding | None = None

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
        for forwarder in self._forwarders:
            forwarder.close()
        for tunnel in list(self._tunnels):
            tunnel.cancel()
        self._writer.close()

    # ─── handshake ────────────────────────────────────────────────────

    async def hello(self, session_id: str | None = None) -> Manifest:
        """Send ``hello``; cache and return the parsed ``Manifest``.

        ``session_id`` resumes that parked session on the env — its suspended
        task, e.g. one a prior connection started — instead of minting a
        fresh session.
        """
        params: dict[str, Any] = {} if session_id is None else {"session_id": session_id}
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
                    )
                    opened = await read_frame(up_reader)
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
        """Start a task; returns the first yield (``{"prompt": ...}``)."""
        return await self._call("tasks.start", {"id": task_id, "args": args or {}})

    async def grade(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Send ``tasks.grade``; returns the evaluation dict (``{"score": ...}``)."""
        return await self._call("tasks.grade", payload)

    async def cancel(self) -> None:
        await self._call("tasks.cancel", {})

    async def bind_inference(
        self,
        *,
        upstream_url: str,
        token: str,
        trace_id: str | None = None,
    ) -> InferenceBinding:
        """Ask the environment server to expose scoped inference inside its workspace."""
        result = await self._call(
            "platform.inference.bind",
            {
                "upstream_url": upstream_url,
                "token": token,
                "trace_id": trace_id,
            },
        )
        base_url = result.get("base_url")
        api_key = result.get("api_key")
        if not isinstance(base_url, str) or not isinstance(api_key, str):
            raise HudProtocolError(-32603, "platform inference binding was malformed")
        self.inference = InferenceBinding(base_url=base_url, api_key=api_key)
        return self.inference

    # ─── JSON-RPC plumbing ────────────────────────────────────────────

    async def _call(
        self,
        method: str,
        params: dict[str, Any],
        *,
        reply_timeout: float | None = None,
    ) -> dict[str, Any]:
        async with self._call_lock:
            try:
                async with asyncio.timeout(reply_timeout):
                    msg_id = next(self._ids)
                    await send_frame(
                        self._writer,
                        {"jsonrpc": "2.0", "id": msg_id, "method": method, "params": params},
                    )
                    reply = await read_frame(self._reader)
                    if reply is None:
                        raise EOFError(f"env closed connection during {method!r}")
                    if reply.get("id") != msg_id:
                        self.abort()
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


# ─── module-level entry points ────────────────────────────────────────


async def _connect_ready(
    host: str,
    port: int,
    *,
    ready_timeout: float,
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
            await client.hello()
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
async def connect(runtime: Runtime, *, ready_timeout: float = 240.0) -> AsyncIterator[HudClient]:
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
    )
    try:
        if runtime.inference is not None:
            await client.bind_inference(
                upstream_url=runtime.inference.upstream_url,
                token=runtime.inference.token,
                trace_id=runtime.inference.trace_id,
            )
    except BaseException:
        await client.close()
        raise
    owner = asyncio.current_task()
    assert owner is not None
    heartbeat_error: Exception | None = None

    async def heartbeat() -> None:
        nonlocal heartbeat_error
        while True:
            await asyncio.sleep(_CONTROL_HEARTBEAT_INTERVAL_SECONDS)
            assert client.manifest is not None
            try:
                await client._call(
                    "hello",
                    {"session_id": client.manifest.session_id},
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


__all__ = [
    "HudClient",
    "HudProtocolError",
    "InferenceBinding",
    "Manifest",
    "ServerInfo",
    "connect",
]
