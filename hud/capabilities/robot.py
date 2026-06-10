"""The ``robot/1`` protocol: wire codec + the agent-side client.

This module defines the ``robot/1`` wire format (msgpack + raw numpy array buffers) and
:class:`RobotClient`, the agent-side capability client that dials a robot env and exchanges
observations/actions over it.

The *env-side* counterpart — the server bridges that own the simulator
(:class:`~hud.environment.robots.bridge.RobotBridge` /
:class:`~hud.environment.robots.bridge.RealtimeRobotBridge`) — lives in
:mod:`hud.environment.robots`, and reuses the wire codec defined here.
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import Any, ClassVar, Self

import numpy as np
import websockets
import websockets.exceptions

from .base import Capability, CapabilityClient


# ─── wire codec (msgpack + raw array buffers, no base64) ─────────────────────


def _encode_array(arr: Any) -> dict[str, Any]:
    a = np.ascontiguousarray(arr)
    return {"shape": list(a.shape), "dtype": str(a.dtype), "data": a.tobytes()}


def _decode_array(d: dict[str, Any]) -> np.ndarray:
    return np.frombuffer(d["data"], dtype=np.dtype(d["dtype"])).reshape(d["shape"]).copy()


def _packb(obj: Any) -> bytes:
    import msgpack

    return msgpack.packb(obj, use_bin_type=True)


def _unpackb(data: bytes) -> Any:
    import msgpack

    return msgpack.unpackb(data, raw=False)


# ─── agent-side client ───────────────────────────────────────────────────────


class RobotClient(CapabilityClient):
    """Live ``robot/1`` connection: send actions, receive observations."""

    protocol: ClassVar[str] = "robot/1"

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
        return cls(cap, ws)

    async def get_observation(self) -> dict[str, Any]:
        """Await the latest observation: ``{"data": {name: ndarray}, "terminated": bool}``.

        Realtime (free-running) bridges also attach a ``"meta"`` block carrying the
        realtime control state used for async/RTC inference::

            {"obs_index": int,            # episode control-tick counter at emit time
             "queue_remaining": int,      # actions still buffered env-side
             "delay": int,                # env's conservative inference-delay estimate (ticks)
             "unexecuted_chunk": ndarray|None}  # [T, A] not-yet-executed tail (executable space); RTC prefix source

        Legacy sync bridges omit ``"meta"`` entirely, so it is only present when the
        env is realtime.
        """
        msg = await self._queue.get()
        data = {name: _decode_array(d) for name, d in msg["data"].items()}
        out: dict[str, Any] = {"data": data, "terminated": bool(msg.get("terminated", False))}
        meta = msg.get("meta")
        if meta is not None:
            decoded = dict(meta)
            unexecuted_chunk = meta.get("unexecuted_chunk")
            decoded["unexecuted_chunk"] = (
                _decode_array(unexecuted_chunk) if unexecuted_chunk is not None else None
            )
            out["meta"] = decoded
        return out

    async def send_action(self, action: Any) -> None:
        """Encode the action and send it (legacy single-action sync path)."""
        arr = np.asarray(action, dtype=np.float32)
        await self._ws.send(_packb({"data": _encode_array(arr)}))

    async def send_chunk(
        self, chunk: Any, *, obs_index: int | None = None, delay_used: int | None = None
    ) -> None:
        """Send a whole action chunk ``[chunk_len, action_dim]`` to a realtime bridge.

        ``obs_index`` echoes the observation the chunk was inferred from so the env
        can measure the real inference delay (ticks consumed in flight); ``delay_used``
        is the delay the agent conditioned on (informational).
        """
        arr = np.asarray(chunk, dtype=np.float32)
        msg: dict[str, Any] = {"chunk": _encode_array(arr)}
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
                if self._queue.full():
                    self._queue.get_nowait()
                await self._queue.put(_unpackb(raw))
        except websockets.exceptions.ConnectionClosed:
            pass
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # never silently stop draining the socket
            import traceback

            print(f"[agent] robot/1 recv loop crashed: {exc!r}", flush=True)
            traceback.print_exc()
            raise


__all__ = ["RobotClient"]
