"""The ``Model``: wraps a policy and owns its inference mechanics.

A ``Model`` knows *how to run* a policy (preprocess → forward → postprocess); the
harness only awaits ``model.ainfer(batch)``. Use :class:`LeRobotModel` for stock
LeRobot checkpoints; subclass :class:`Model` and implement ``infer`` otherwise.

:meth:`Model.infer` is batch-shaped (one batch dict in, an ``[N, T, A]`` chunk out) and
stateless across calls, so one model can be shared and batched across concurrent rollouts
(see :mod:`hud.agents.robot.batching`); per-episode state belongs on the agent.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from ._types import ActionArray


class Model:
    """Owns a policy and its inference mechanics.

    Driven by :class:`~hud.agents.robot.agent.RobotAgent`: :meth:`reset` once per
    episode, then :meth:`ainfer` (awaited; one rollout) each inference.
    """

    def reset(self) -> None:
        """Reset per-episode model state. Override when the policy is stateful."""

    def infer(self, batch: Any) -> ActionArray:
        """runs policy on a batch, returns [N, T, A] action chunk"""
        raise NotImplementedError

    async def ainfer(self, batch: Any) -> ActionArray:
        """Awaited single-rollout entry: run :meth:`infer` in a thread, return its ``[T, A]``."""
        return (await asyncio.to_thread(self.infer, batch))[0]


class LeRobotModel(Model):
    """LeRobot policy with pre/post-processors: ``preprocess`` → ``predict_action_chunk`` →
    ``postprocess``. ``preprocess`` adds the batch dim for an unbatched sample and is a no-op
    for an already-stacked one, so :meth:`infer` handles both single and batched inputs.
    """

    def __init__(self, policy: Any, preprocess: Any, postprocess: Any) -> None:
        self.policy = policy
        self.preprocess = preprocess
        self.postprocess = postprocess
        #: Flipped to False after the first forward; used to print the one-time
        #: CUDA/flow-matching warmup message.
        self._first_inference = True

    def reset(self) -> None:
        """Reset LeRobot's open-loop action queue for the new episode."""
        if hasattr(self.policy, "reset"):
            self.policy.reset()

    def infer(self, batch: Any) -> ActionArray:
        """run batch dict (N dim) → [N, T, A] chunk"""
        import torch  # pyright: ignore[reportMissingImports]
        if self._first_inference:
            print("[agent] first inference — flow-matching/CUDA warmup; this may take a while", flush=True)
        with torch.no_grad():
            chunk = self.postprocess(self.policy.predict_action_chunk(self.preprocess(batch)))
        if self._first_inference:
            print("[agent] first inference done — inference is now fast", flush=True)
            self._first_inference = False
        return chunk.float().cpu().numpy()
   


class RemoteModel(Model):
    """Weightless client to an OpenPI-WebSocket policy server: ships the adapter's request
    dict, returns the server's chunk. All pre/post-processing lives in the adapter + server.
    """

    def __init__(self, host: str = "localhost", port: int = 8000, *, response_key: str = "actions") -> None:
        self.host = host
        self.port = port
        #: Key under which the server returns the chunk — "actions" (stock OpenPI) or "action" (Cosmos).
        self.response_key = response_key
        self._client: Any = None

    def connect(self) -> None:
        """Open the websocket (idempotent); blocks until the server is up."""
        if self._client is None:
            from openpi_client import websocket_client_policy

            print(f"[agent] connecting to openpi server ws://{self.host}:{self.port} — on hold...", flush=True)
            self._client = websocket_client_policy.WebsocketClientPolicy(self.host, self.port)

    def reset(self) -> None:
        """Connect before the act loop (once per episode), so blocking happens at a known point."""
        self.connect()

    def infer(self, batch: Any) -> ActionArray:
        """Ship one request dict → the server's ``[T, A]`` chunk, returned as ``[1, T, A]``."""
        self.connect()  # safety net if reset() wasn't called
        chunk = np.asarray(self._client.infer(batch)[self.response_key], dtype=np.float32)
        return chunk[None]  # add the leading N=1 batch dim


__all__ = [
    "LeRobotModel",
    "Model",
    "RemoteModel",
]
