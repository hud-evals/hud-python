"""The ``openpi/0`` protocol: wire codec + the agent-side client.

``openpi/0`` is openpi-like — it reuses openpi's msgpack-numpy wire format and flat
observation/action naming — but flips the roles: here the *env* is the WebSocket
server (it owns the world) and the *agent* is the client (it acts in the world).
:class:`RobotClient` is that agent-side client; it dials a robot env and exchanges
observations/actions over the socket.

The *env-side* counterpart — the server bridge that owns the simulator
(:class:`~hud.environment.robot.bridge.RobotBridge`) — lives in
:mod:`hud.environment.robot`, and reuses the wire codec defined here.
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import Any, ClassVar, Self

import numpy as np
import websockets
import websockets.exceptions
from openpi_client import msgpack_numpy

from .base import Capability, CapabilityClient

# ─── wire codec ──────────────────────────────────────────────────────────────
# openpi's msgpack-numpy codec: numpy arrays nested anywhere in a message serialize
# transparently and recursively, so neither end wraps obs/action fields by hand.
_packb = msgpack_numpy.packb
_unpackb = msgpack_numpy.unpackb


# ─── agent-side client ───────────────────────────────────────────────────────


class RobotClient(CapabilityClient):
    """Live ``openpi/0`` connection: send actions, receive observations."""

    protocol: ClassVar[str] = "openpi"

    def __init__(self, capability: Capability, ws: Any) -> None:
        self.capability = capability
        self._ws = ws
        self._queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=1)
        self._mailman = asyncio.create_task(self._recv_loop())

    @property
    def contract(self) -> dict[str, Any]:
        """The env's full contract from the manifest (robot_type, control_rate, features, ...)."""
        return dict(self.capability.params.get("contract") or {})

    def spaces(self) -> tuple[dict[str, Any], dict[str, Any]]:
        """Split the contract's ``features`` into ``(action_space, observation_space)`` by role.

        ``action_space`` is the single ``role == "action"`` feature; the
        observation space is the ordered ``name -> feature`` map of the
        ``role == "observation"`` features. Full feature dicts (type/dtype/shape/
        names/stats) are preserved, so agents wire policies with no shared config.
        """
        features = self.contract.get("features", {})
        action = next((f for f in features.values() if f.get("role") == "action"), {})
        observations = {n: f for n, f in features.items() if f.get("role") == "observation"}
        return action, observations

    @classmethod
    async def connect(cls, cap: Capability) -> Self:
        ws = await websockets.connect(cap.url, max_size=None)
        # Consume the connect-time metadata frame (always first); a string frame is the env's error convention.
        raw = await ws.recv()
        if isinstance(raw, str):
            raise RuntimeError(f"robot env error on connect:\n{raw}")
        return cls(cap, ws)

    async def get_observation(self) -> dict[str, Any]:
        """Await the latest observation: ``{"data": {name: ndarray}, "terminated": bool}``.

        On the wire the env sends an openpi-style *flat* dict (``{name: ndarray, ...}``)
        with ``terminated`` (and, for realtime bridges, ``meta``) as sibling keys; we
        regroup the array fields under ``"data"`` for the agent harness. Arrays — nested
        anywhere, including inside ``meta`` (e.g. ``unexecuted_chunk``) — are already
        decoded by the codec.

        Realtime (free-running) bridges attach a ``"meta"`` block carrying the realtime
        control state used for async/RTC inference (``obs_index``, ``queue_remaining``,
        ``delay``, ``unexecuted_chunk``); sync bridges omit it.

        Raises if the env reported an error (a string traceback frame).
        """
        msg = await self._queue.get()
        if "error" in msg:
            raise RuntimeError(f"robot env error:\n{msg['error']}")
        terminated = bool(msg.pop("terminated", False))
        meta = msg.pop("meta", None)
        out: dict[str, Any] = {"data": msg, "terminated": terminated}
        if meta is not None:
            out["meta"] = meta
        return out

    async def send_action(self, action: Any) -> None:
        """Send a single action under the openpi ``"actions"`` key (sync path)."""
        await self._ws.send(_packb({"actions": np.asarray(action, dtype=np.float32)}))

    async def send_chunk(
        self, chunk: Any, *, obs_index: int | None = None, delay_used: int | None = None
    ) -> None:
        """Send a whole action chunk ``[chunk_len, action_dim]`` to a realtime bridge.

        ``obs_index`` echoes the observation the chunk was inferred from so the env
        can measure the real inference delay (ticks consumed in flight); ``delay_used``
        is the delay the agent conditioned on (informational).
        """
        msg: dict[str, Any] = {"actions": np.asarray(chunk, dtype=np.float32)}
        if obs_index is not None:
            msg["obs_index"] = int(obs_index)
        if delay_used is not None:
            msg["delay_used"] = int(delay_used)
        await self._ws.send(_packb(msg))

    async def close(self) -> None:
        self._mailman.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self._mailman
        with contextlib.suppress(Exception):
            await self._ws.close()

    async def _recv_loop(self) -> None:
        try:
            async for raw in self._ws:
                # A string frame is the env's error convention (a traceback), not an obs.
                msg = {"error": raw} if isinstance(raw, str) else _unpackb(raw)
                if self._queue.full():
                    self._queue.get_nowait()
                await self._queue.put(msg)
        except websockets.exceptions.ConnectionClosed:
            pass
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # never silently stop draining the socket
            import traceback

            print(f"[agent] robot recv loop crashed: {exc!r}", flush=True)
            traceback.print_exc()
            raise


__all__ = ["RobotClient"]
