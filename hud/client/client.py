"""HudClient: JSON-RPC client for the HUD wire protocol.

Transport + ergonomics for an ``Env.serve()`` endpoint. Drives the
``hello`` / ``tasks.list`` / ``tasks.start`` / ``tasks.evaluate`` /
``tasks.cancel`` / ``bye`` methods, and exposes capability access:

* ``binding(name)`` — the raw ``Capability`` declaration (BYO connection).
* ``open(name)``    — a live, cached ``CapabilityClient`` (we own the socket).
* ``task(id, **args)`` — a ``Trace`` run-handle (async context manager).

Two module-level entry points sit on top:

* ``connect(endpoint)`` — attach to an already-running env (borrow; no teardown).
* ``launch(ref)``       — provision + attach (own; tears down what it started).
"""

from __future__ import annotations

import asyncio
import contextlib
import itertools
import logging
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any, Self

from hud.capabilities import (
    Capability,
    CapabilityClient,
    CDPClient,
    MCPClient,
    RFBClient,
    SSHClient,
)
from hud.env.utils import read_frame, send_frame

from . import Manifest, ServerInfo
from .run import Run

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from types import TracebackType

LOGGER = logging.getLogger("hud.client")

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


class HudClient:
    """JSON-RPC client for an ``Env.serve()`` endpoint.

    Prefer the module-level ``hud.connect`` / ``hud.launch`` helpers; this class
    is the transport they sit on. ``hello`` runs on ``__aenter__`` so
    ``manifest`` is ready immediately::

        async with await HudClient.connect("127.0.0.1", 9001) as client:
            async with client.task("write_hello") as run:
                ssh = await run.client.open("shell")
                ...
                run.trace.content = "done"          # the answer, graded on exit
            print(run.trace.reward)
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

    @classmethod
    async def connect(cls, host: str = "127.0.0.1", port: int = 0) -> Self:
        reader, writer = await asyncio.open_connection(host, port)
        return cls(reader, writer)

    async def __aenter__(self) -> Self:
        await self.hello()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.close()

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for cap_client in self._opened.values():
            with contextlib.suppress(Exception):
                await cap_client.close()
        self._opened.clear()
        try:
            await self._call("bye", {})
        except Exception:
            LOGGER.debug("bye failed (env may have already closed)", exc_info=True)
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
    # ``binding`` and ``open`` resolve the same capability *by protocol*; they
    # differ only in what they hand back:
    #   binding(proto) -> Capability        raw declaration (url/params; BYO conn)
    #   open(proto)    -> CapabilityClient   live, connected, cached client

    def binding(self, protocol: str) -> Capability:
        """Resolve a ``Capability`` by protocol (family ``"cdp"`` or full ``"cdp/1.3"``).

        Returns the raw declaration — use this when something else owns the
        connection (e.g. browser-use reads the CDP url). Ambiguous protocols
        (multiple bindings) raise; publish distinct protocols to disambiguate.
        """
        if self.manifest is None:
            raise RuntimeError("call hello() before accessing bindings")
        matches = [
            c
            for c in self.manifest.bindings
            if c.protocol == protocol or c.protocol.split("/", 1)[0] == protocol
        ]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            protos = ", ".join(c.protocol for c in matches)
            raise KeyError(f"ambiguous protocol {protocol!r}; matches: {protos}")
        available = ", ".join(c.protocol for c in self.manifest.bindings) or "<none>"
        raise KeyError(f"no binding for protocol {protocol!r} (available: {available})")

    async def open(self, protocol: str) -> CapabilityClient:
        """Open (and cache) a live ``CapabilityClient`` for a protocol.

        Resolves like ``binding`` but connects and returns a live client, owned by
        this connection and closed on ``close()``.
        """
        cap = self.binding(protocol)
        cap_client = self._opened.get(cap.protocol)
        if cap_client is None:
            client_cls = _CLIENT_REGISTRY.get(cap.protocol)
            if client_cls is None:
                raise ValueError(
                    f"no client registered for protocol {cap.protocol!r}; "
                    f"use binding({protocol!r}) for raw access",
                )
            cap_client = await client_cls.connect(cap)
            self._opened[cap.protocol] = cap_client
        return cap_client

    # ─── tasks ────────────────────────────────────────────────────────

    def task(self, task_id: str, **args: Any) -> Run:
        """Return a ``Run`` handle for a task (async context manager).

        ``async with client.task("sum_column", sheet="q3.xlsx") as run: ...``
        starts the task on enter (populating ``run.trace.prompt``) and grades it on
        exit (populating ``run.trace.reward``).
        """
        return Run(self, task_id, args)

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

    async def evaluate(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Send ``tasks.evaluate``; returns the final evaluation dict."""
        return await self._call("tasks.evaluate", payload)

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


@asynccontextmanager
async def connect(host: str = "127.0.0.1", port: int = 0) -> AsyncIterator[HudClient]:
    """Attach to an already-running env (borrow; does not tear down the substrate)."""
    client = await HudClient.connect(host, port)
    async with client:
        yield client


__all__ = ["HudClient", "HudProtocolError", "connect"]
