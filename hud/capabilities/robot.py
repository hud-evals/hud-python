"""The ``openpi/0`` wire codec and agent-side client; unlike openpi, the env is the server."""

from __future__ import annotations

import asyncio
import contextlib
from typing import Any, ClassVar, Self, cast

import numpy as np
import websockets
import websockets.exceptions
from openpi_client import msgpack_numpy

from .base import Capability, CapabilityClient

# openpi's msgpack-numpy codec serializes nested numpy arrays, so neither end wraps
# obs/action fields by hand.
_packb = msgpack_numpy.packb
_unpackb = msgpack_numpy.unpackb


class RobotClient(CapabilityClient):
    """Live ``openpi/0`` connection: send actions, receive observations."""

    protocol: ClassVar[str] = "openpi"

    def __init__(
        self, capability: Capability, ws: Any, *, seed: dict[str, Any] | None = None
    ) -> None:
        self.capability = capability
        self._ws = ws
        self._queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=1)
        if seed is not None:
            self._queue.put_nowait(seed)  # first obs from connect's claim handshake
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

        The action space is the single action feature; the observation space maps
        names to observation features, in contract order.
        """
        features = self.contract.get("features", {})
        action = next((f for f in features.values() if f.get("role") == "action"), {})
        observations = {n: f for n, f in features.items() if f.get("role") == "observation"}
        return action, observations

    @classmethod
    async def connect(cls, cap: Capability, *, token: str | None = None) -> Self:
        """Dial the robot WebSocket; ``token`` claims a sim slot after the metadata frame.

        A ``None`` token binds the sole slot of a single-env bridge; vectorized bridges
        need the token from ``endpoint.reset()``. A rejected claim raises here rather
        than hanging the first ``get_observation``.
        """
        # Bounded dial: an undialed claim stalls the bridge's barrier for every peer.
        ws = await websockets.connect(cap.url, max_size=None, ping_interval=None, open_timeout=30.0)
        # Consume initial metadata; string means env error.
        raw = await ws.recv()
        if isinstance(raw, str):
            raise RuntimeError(f"robot env error on connect:\n{raw}")
        # HUD bridges require a claim frame binding this connection to an episode slot;
        # plain openpi servers have no slots.
        meta = _unpackb(raw)
        seed: dict[str, Any] | None = None
        if token is not None or (isinstance(meta, dict) and meta.get("claim_required")):
            await ws.send(cast("Any", _packb({"claim": token})))
            try:
                first = await asyncio.wait_for(ws.recv(), timeout=30.0)
            except TimeoutError as exc:
                with contextlib.suppress(Exception):
                    await ws.close()
                raise RuntimeError("robot env timed out after slot claim") from exc
            if isinstance(first, str):
                with contextlib.suppress(Exception):
                    await ws.close()
                raise RuntimeError(f"robot env error on claim:\n{first}")
            seed = _unpackb(first)
        return cls(cap, ws, seed=seed)

    async def get_observation(self) -> dict[str, Any]:
        """Await the latest observation: ``{"data": {name: ndarray}, "terminated": bool}``.

        A per-step ``"reward"`` and, from realtime bridges, ``"meta"`` ride alongside.
        Raises on an env error frame.
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

        ``obs_index`` names the observation the chunk was inferred from, so the env can
        measure the real inference delay; ``delay_used`` is informational.
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
