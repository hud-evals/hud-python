"""CDPClient — Chrome DevTools Protocol over a single page-target WebSocket.

Thin transport: opens one WebSocket to a Chromium *page* target and speaks CDP
JSON-RPC. A background reader demuxes command replies (matched by ``id``) from
protocol events. The only verb is ``send(method, params)`` — callers build
higher-level helpers (navigate, evaluate, screenshot, click, type, …) on top of
it.

Discovery
---------
A ``cdp/1.3`` capability publishes the DevTools endpoint as ``ws://host:port``.
On connect we resolve a concrete page target:

* an explicit ``params['target_id']`` → ``ws://host:port/devtools/page/<id>``,
* a full ``/devtools/`` URL is used verbatim,
* otherwise ``GET http://host:port/json`` picks the first ``page`` target
  (creating one via ``/json/new`` if none exist).
"""

from __future__ import annotations

import asyncio
import contextlib
import itertools
import json
import logging
from typing import TYPE_CHECKING, Any, ClassVar, Self
from urllib.parse import urlsplit

import httpx
from websockets.asyncio.client import connect as ws_connect
from websockets.exceptions import ConnectionClosed

from .base import Capability, CapabilityClient

if TYPE_CHECKING:
    from websockets.asyncio.client import ClientConnection

LOGGER = logging.getLogger("hud.capabilities.cdp")


class CDPError(RuntimeError):
    """Raised when Chrome returns a CDP error frame for a command."""

    def __init__(self, method: str, error: dict[str, Any]) -> None:
        code = error.get("code")
        message = error.get("message", "")
        super().__init__(f"CDP {method!r} failed [{code}]: {message}")
        self.code = code
        self.message = message


class CDPClient(CapabilityClient):
    """Live CDP session bound to one Chromium page target."""

    protocol: ClassVar[str] = "cdp/1.3"

    def __init__(self, capability: Capability, ws: ClientConnection) -> None:
        self.capability = capability
        self._ws = ws
        self._ids = itertools.count(1)
        self._pending: dict[int, asyncio.Future[dict[str, Any]]] = {}
        self._reader: asyncio.Task[None] | None = None

    @classmethod
    async def connect(cls, cap: Capability) -> Self:
        parts = urlsplit(cap.url)
        host = parts.hostname or "127.0.0.1"
        port = parts.port or 9222
        ws_url = await cls._resolve_ws_url(host, port, cap.params.get("target_id"), cap.url)
        ws = await ws_connect(ws_url, max_size=None)
        client = cls(cap, ws)
        client._reader = asyncio.create_task(client._read_loop())
        # Enable the domains every browser-driving tool relies on.
        await client.send("Page.enable")
        await client.send("Runtime.enable")
        await client.send("DOM.enable")
        return client

    @staticmethod
    async def _resolve_ws_url(
        host: str,
        port: int,
        target_id: str | None,
        raw_url: str,
    ) -> str:
        if "/devtools/" in raw_url:
            return raw_url
        if target_id:
            return f"ws://{host}:{port}/devtools/page/{target_id}"
        async with httpx.AsyncClient(timeout=10.0) as http:
            resp = await http.get(f"http://{host}:{port}/json")
            targets: list[dict[str, Any]] = resp.json()
            pages = [
                t for t in targets if t.get("type") == "page" and t.get("webSocketDebuggerUrl")
            ]
            if pages:
                return str(pages[0]["webSocketDebuggerUrl"])
            created = await http.put(f"http://{host}:{port}/json/new?about:blank")
            ws_url = created.json().get("webSocketDebuggerUrl")
        if not ws_url:
            raise ValueError(f"no CDP page target available at {host}:{port}")
        return str(ws_url)

    # ─── JSON-RPC plumbing ────────────────────────────────────────────

    async def send(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        """Issue one CDP command and await its result frame."""
        msg_id = next(self._ids)
        future: asyncio.Future[dict[str, Any]] = asyncio.get_running_loop().create_future()
        self._pending[msg_id] = future
        await self._ws.send(json.dumps({"id": msg_id, "method": method, "params": params or {}}))
        try:
            return await future
        finally:
            self._pending.pop(msg_id, None)

    async def _read_loop(self) -> None:
        try:
            async for raw in self._ws:
                msg = json.loads(raw)
                msg_id = msg.get("id")
                future = self._pending.get(msg_id) if msg_id is not None else None
                if future is None or future.done():
                    continue  # protocol event (no waiter) — ignored for now
                if "error" in msg:
                    future.set_exception(CDPError(str(msg.get("method", "")), msg["error"]))
                else:
                    future.set_result(msg.get("result", {}))
        except (ConnectionClosed, OSError) as exc:
            LOGGER.debug("CDP read loop ended: %s", exc)
        finally:
            for future in self._pending.values():
                if not future.done():
                    future.set_exception(ConnectionError("CDP connection closed"))

    async def close(self) -> None:
        if self._reader is not None:
            self._reader.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._reader
        with contextlib.suppress(Exception):
            await self._ws.close()


__all__ = ["CDPClient", "CDPError"]
