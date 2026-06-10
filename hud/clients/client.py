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
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit

from hud.capabilities import (
    Capability,
    CapabilityClient,
    CDPClient,
    MCPClient,
    RFBClient,
    SSHClient,
)
from hud.environment.utils import read_frame, send_frame

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from hud.environment.runtime import Runtime

LOGGER = logging.getLogger("hud.clients")

#: protocol -> CapabilityClient subclass, for ``HudClient.open``.
_CLIENT_REGISTRY: dict[str, type[CapabilityClient]] = {
    cls.protocol: cls for cls in (SSHClient, RFBClient, MCPClient, CDPClient)
}


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

    ``bindings`` carry concrete connection data: the env resolves backed
    declarations (materializing their daemons) when it answers ``hello``.
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
    ) -> None:
        self._reader = reader
        self._writer = writer
        self._ids = itertools.count(1)
        self._closed = False
        self.manifest: Manifest | None = None
        self._opened: dict[str, CapabilityClient] = {}

    # ─── lifecycle ────────────────────────────────────────────────────

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for cap_client in self._opened.values():
            with contextlib.suppress(Exception):
                await cap_client.close()
        self._opened.clear()
        # No `bye`: a plain disconnect leaves the env's held session for a later
        # connection to grade; `grade` itself clears the session when it completes.
        self._writer.close()
        with contextlib.suppress(Exception):
            await self._writer.wait_closed()

    # ─── handshake ────────────────────────────────────────────────────

    async def hello(self) -> Manifest:
        """Send ``hello``; cache and return the parsed ``Manifest``."""
        result = await self._call("hello", {})
        env = result.get("env") or {}
        bindings = [Capability.from_manifest(b) for b in (result.get("bindings") or [])]
        self.manifest = Manifest(
            session_id=result["session_id"],
            protocol_version=self.PROTOCOL_VERSION,
            server_info=ServerInfo(
                name=env.get("name", "unknown"),
                version=env.get("version", "0.0.0"),
            ),
            bindings=bindings,
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
            c
            for c in self.manifest.bindings
            if ref in (c.name, c.protocol, c.protocol.split("/", 1)[0])
        ]
        if len(matches) == 1:
            return matches[0]
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
            if client_cls is None:
                raise ValueError(
                    f"no client registered for protocol {cap.protocol!r}; "
                    f"use binding({ref!r}) for raw access",
                )
            cap_client = await client_cls.connect(cap)
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

    # ─── JSON-RPC plumbing ────────────────────────────────────────────

    async def _call(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        msg_id = next(self._ids)
        await send_frame(
            self._writer,
            {"jsonrpc": "2.0", "id": msg_id, "method": method, "params": params},
        )
        reply = await read_frame(self._reader)
        if reply is None:
            raise HudProtocolError(-32000, f"env closed connection during {method!r}")
        if "error" in reply:
            err = reply["error"]
            raise HudProtocolError(int(err.get("code", -32000)), str(err.get("message", "")))
        result = reply.get("result")
        if not isinstance(result, dict):
            raise HudProtocolError(-32603, f"{method!r}: result was not an object")
        return result


# ─── module-level entry points ────────────────────────────────────────


async def _connect_ready(
    host: str,
    port: int,
    *,
    ready_timeout: float,
    interval: float = 0.5,
) -> HudClient:
    """Connect to a control channel, retrying until it accepts or ``ready_timeout``.

    A freshly-provisioned substrate may not be serving yet; the client owns
    waiting for readiness by retrying the connect.
    """
    loop = asyncio.get_event_loop()
    deadline = loop.time() + ready_timeout
    while True:
        try:
            reader, writer = await asyncio.open_connection(host, port)
            return HudClient(reader, writer)
        except OSError:
            if loop.time() >= deadline:
                raise
            await asyncio.sleep(interval)


@asynccontextmanager
async def connect(runtime: Runtime, *, ready_timeout: float = 120.0) -> AsyncIterator[HudClient]:
    """Connect a :class:`HudClient` to a provisioned substrate's control channel.

    Takes the :class:`~hud.environment.runtime.Runtime` a provider yielded (or
    one constructed directly for a substrate served elsewhere) and retries the
    connect until the channel is ready. Does not tear the substrate down —
    lifecycle belongs to whichever provider brought it up.
    """
    parts = urlsplit(runtime.url)
    if parts.scheme not in ("", "tcp"):
        raise NotImplementedError(
            f"control transport {parts.scheme!r} not supported yet (only tcp://)",
        )
    client = await _connect_ready(
        parts.hostname or "127.0.0.1",
        parts.port or 0,
        ready_timeout=ready_timeout,
    )
    try:
        await client.hello()
        yield client
    finally:
        await client.close()


__all__ = ["HudClient", "HudProtocolError", "Manifest", "ServerInfo", "connect"]
