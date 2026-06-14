"""``RobotEndpoint`` — the env-side control handle for a :class:`RobotBridge`.

The single surface an env uses to drive a bridge through an episode (``start`` /
``stop`` / ``reset`` / ``result`` / ``url``). Its whole point is to make *where the
bridge runs* irrelevant — the env code is identical either way:

- **Same process** — ``RobotEndpoint(bridge)``: calls go straight through.
- **Different process** — ``RobotEndpoint.remote(host, port)`` on the env side,
  ``RobotEndpoint(bridge).serve(host, port)`` in the process that owns the sim (e.g.
  Isaac/Omniverse, which pins the main thread); calls are forwarded over JSON-RPC.

Control plane only: the agent's step/observation loop tunnels straight to the bridge's
``robot`` WebSocket, and the wire contract stays env-side.

    async def my_task(task_id: int, seed: int = 0):
        prompt = await endpoint.reset(task_id=task_id, seed=seed)
        yield {"prompt": prompt}
        yield await endpoint.result()
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import TYPE_CHECKING, Any

from hud.environment.utils import error, read_frame, reply, send_frame

if TYPE_CHECKING:
    from hud.capabilities import Capability

    from .bridge import RobotBridge


class RobotEndpoint:
    """Drive a simulation bridge - even if it's in another process.

    Build it one of two ways and use the *identical* methods either way:
    ``RobotEndpoint(bridge)`` (local) or ``RobotEndpoint.remote(host, port)`` (a handle
    on a bridge that another process exposes via :meth:`serve` defined here).
    """

    def __init__(
        self,
        bridge: RobotBridge | None = None,
        *,
        host: str | None = None,
        port: int | None = None,
    ) -> None:
        self._bridge = bridge  # set => local; None => remote (dial host:port)
        self._host = host
        self._port = port
        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None

    @classmethod
    def remote(cls, host: str, port: int) -> RobotEndpoint:
        """A handle on a bridge served by another process; :meth:`connect` once it's up."""
        return cls(host=host, port=port)

    @property
    def _is_remote(self) -> bool:
        return self._bridge is None

    def _local_bridge(self) -> RobotBridge:
        bridge = self._bridge
        if bridge is None:
            raise RuntimeError("local bridge required")
        return bridge

    # ── control surface (same whether local or remote) ───────────────────
    async def url(self) -> str:
        """The bridge's ``ws://`` address — publish it as the robot capability."""
        if self._is_remote:
            return (await self._call("url"))["url"]
        return self._local_bridge().url

    async def capability(self, *, name: str = "robot", contract: dict[str, Any]) -> Capability:
        """The ``robot`` capability for this bridge — mirrors ``Workspace.capability()``.

        Publish it from an ``@env.initialize`` hook after :meth:`start` (the URL only
        exists once the bridge has bound its socket)::

            @env.initialize
            async def _up():
                await endpoint.start()
                env.add_capability(await endpoint.capability(contract=CONTRACT))
        """
        from hud.capabilities import Capability

        return Capability.robot(name=name, url=await self.url(), contract=contract)

    async def start(self) -> None:
        if self._is_remote:
            await self._call("start")
        else:
            await self._local_bridge().start()

    async def stop(self) -> None:
        if self._is_remote:
            await self._call("stop")
        else:
            await self._local_bridge().stop()

    async def reset(self, **task_args: Any) -> str:
        """Start a new episode; return the task prompt."""
        if self._is_remote:
            return (await self._call("reset", task_args))["prompt"]
        return await self._local_bridge()._reset(**task_args)

    async def result(self, **extra: Any) -> dict[str, Any]:
        """The episode score dict, merged with any caller ``extra`` metadata."""
        res = await self._call("result") if self._is_remote else self._local_bridge().result()
        res = {**res, **extra}
        print(
            f"[env] result: success={res.get('success')} "
            f"total_reward={res.get('total_reward', 0.0):.3f}",
            flush=True,
        )
        return res

    """ in your simulation program where bridge is started """

    # ── serving: expose a local bridge so a remote endpoint can drive it ──
    async def serve(self, host: str = "127.0.0.1", port: int = 9100) -> asyncio.AbstractServer:
        """Serve this (local) bridge's control surface over JSON-RPC.

        The process that owns the sim calls this; a ``remote()`` endpoint elsewhere then
        drives the bridge through it. Await the returned server's ``wait_closed()`` to run
        for the process's lifetime. Calls dispatch on *this* loop — the sim's — so e.g.
        ``reset`` runs inline on the sim thread.
        """
        if self._bridge is None:
            raise RuntimeError("serve() needs a local bridge: RobotEndpoint(bridge)")
        server = await asyncio.start_server(self._handle, host, port)
        print(f"[env] control endpoint listening on {host}:{port}", flush=True)
        return server

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        with contextlib.suppress(ConnectionResetError, asyncio.IncompleteReadError):
            while (msg := await read_frame(reader)) is not None:
                try:
                    result = await self._dispatch(msg["method"], msg.get("params") or {})
                    await send_frame(writer, reply(msg["id"], result))
                except Exception as exc:  # surface to the caller, keep serving the link
                    await send_frame(writer, error(msg["id"], -32000, str(exc)))
        writer.close()
        with contextlib.suppress(Exception):
            await writer.wait_closed()

    async def _dispatch(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        b = self._local_bridge()
        if method == "url":
            return {"url": b.url}
        if method == "reset":
            return {"prompt": await b._reset(**params)}
        if method == "result":
            return b.result()
        if method == "start":
            await b.start()
            return {}
        if method == "stop":
            await b.stop()
            return {}
        raise ValueError(f"unknown method {method!r}")

    # ── remote link (no-ops when local) ──────────────────────────────────
    async def connect(self, *, connect_timeout_s: float = 240.0, retry_every: float = 2.0) -> None:
        """Dial the serving process, retrying until it's up. No-op for a local endpoint."""
        if not self._is_remote:
            return
        try:
            async with asyncio.timeout(connect_timeout_s):
                while True:
                    try:
                        self._reader, self._writer = await asyncio.open_connection(
                            self._host, self._port
                        )
                        return
                    except OSError:
                        await asyncio.sleep(retry_every)
        except TimeoutError as exc:
            raise TimeoutError(
                f"timed out connecting to {self._host}:{self._port} after {connect_timeout_s}s"
            ) from exc

    async def close(self) -> None:
        """Drop the link (no-op when local; does not stop the bridge)."""
        if self._writer is not None:
            self._writer.close()
            with contextlib.suppress(Exception):
                await self._writer.wait_closed()
            self._reader = self._writer = None

    async def _call(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        # Strictly request/reply, one call at a time, so a constant id is enough.
        if self._writer is None or self._reader is None:
            raise RuntimeError("not connected; call connect() first")
        await send_frame(
            self._writer, {"jsonrpc": "2.0", "id": 1, "method": method, "params": params or {}}
        )
        msg = await read_frame(self._reader)
        if msg is None:
            raise ConnectionError(f"connection closed awaiting {method!r} reply")
        if "error" in msg:
            raise RuntimeError(f"{method} failed: {msg['error']['message']}")
        return msg["result"]


__all__ = ["RobotEndpoint"]
