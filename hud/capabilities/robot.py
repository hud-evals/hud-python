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

    def get_control_rate(self, default: int = 10) -> int:
        """The env's control rate in Hz (frames/actions per second), rounded to at least 1."""
        return max(1, round(self.contract.get("control_rate") or default))

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
    async def connect(cls, cap: Capability, *, token: str | None = None) -> Self:
        """Dial the robot WebSocket; ``token`` claims a sim slot after the metadata frame.

        HUD bridges require the claim and send nothing until it arrives — pass the
        token from ``endpoint.reset()``. Omit it only for servers without slots.
        """
        ws = await websockets.connect(cap.url, max_size=None, ping_interval=None)
        # Consume initial metadata; string means env error.
        raw = await ws.recv()
        if isinstance(raw, str):
            raise RuntimeError(f"robot env error on connect:\n{raw}")
        # Bind this connection to a claimed episode slot (scalar openpi from here).
        if token is not None:
            await ws.send(_packb({"claim": token}))
        return cls(cap, ws)

    async def get_observation(self) -> dict[str, Any]:
        """Await the latest observation: ``{"data": {name: ndarray}, "terminated": bool}``.

        Wire format is a flat openpi dict (``{name: ndarray, ...}``) with ``terminated``
        as a sibling; array fields are regrouped under ``"data"``. Realtime bridges also
        attach ``"meta"`` (``obs_index``, ``queue_remaining``, ``delay``,
        ``unexecuted_chunk``); sync bridges omit it. Raises on an env error frame.
        """
        msg = await self._queue.get()
        if "error" in msg:
            raise RuntimeError(f"robot env error:\n{msg['error']}")
        # Scalar openpi: each connection is one slot; terminated is always a bool.
        terminated = msg.pop("terminated", False)
        meta = msg.pop("meta", None)
        reward = msg.pop("reward", None)
        out: dict[str, Any] = {"data": msg, "terminated": terminated}
        if reward is not None:
            out["reward"] = reward  # per-step reward sibling (RL collection)
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
