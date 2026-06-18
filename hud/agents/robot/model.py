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
import importlib
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from ._types import ActionArray


class Model:
    """Owns a policy and its inference mechanics.

    Stateless by contract: the agent owns all per-episode state (the open-loop chunk), so a
    single model can be shared and batched across concurrent rollouts. There is deliberately
    no ``reset`` hook — anything that resets per episode belongs on the agent, not here.
    Driven by :class:`~hud.agents.robot.agent.RobotAgent`, which awaits :meth:`ainfer`.
    """

    def infer(self, batch: Any) -> ActionArray:
        """Run the policy on an ``[N, ...]`` batch, return an ``[N, T, A]`` chunk.

        Implementations MUST keep the leading batch dim ``N`` (even for ``N == 1``):
        :meth:`ainfer` indexes ``[0]`` and :class:`~hud.agents.robot.batching.BatchedModel`
        scatters rows along it, so a squeezed ``[T, A]`` silently breaks both.
        """
        raise NotImplementedError

    async def ainfer(self, batch: Any) -> ActionArray:
        """Awaited single-rollout entry: run :meth:`infer` in a thread, return its single
        ``[T, A]`` row. Indexing ``[0]`` assumes :meth:`infer` honors the ``[N, T, A]`` contract.
        """
        return (await asyncio.to_thread(self.infer, batch))[0]


class LeRobotModel(Model):
    """LeRobot policy with pre/post-processors: ``preprocess`` → ``predict_action_chunk`` →
    ``postprocess``. ``preprocess`` adds the batch dim for an unbatched sample and is a no-op
    for an already-stacked one, so :meth:`infer` handles both single and batched inputs.

    Stateless: ``predict_action_chunk`` is a pure forward and the agent owns the open-loop
    chunk, so LeRobot's internal action queue is never consumed here — hence no ``reset``.
    """

    def __init__(self, policy: Any, preprocess: Any, postprocess: Any) -> None:
        self.policy = policy
        self.preprocess = preprocess
        self.postprocess = postprocess
        #: Flipped to False after the first forward; used to print the one-time
        #: CUDA/flow-matching warmup message.
        self._first_inference = True

    def infer(self, batch: Any) -> ActionArray:
        """run batch dict (N dim) → [N, T, A] chunk"""
        torch: Any = importlib.import_module("torch")
        if self._first_inference:
            print(
                "[agent] first inference — flow-matching/CUDA warmup; this may take a while",
                flush=True,
            )
        with torch.no_grad():
            chunk = self.postprocess(self.policy.predict_action_chunk(self.preprocess(batch)))
        if self._first_inference:
            print("[agent] first inference done — inference is now fast", flush=True)
            self._first_inference = False
        arr = chunk.float().cpu().numpy()
        assert arr.ndim == 3, (
            f"expected [N, T, A] chunk, got {arr.shape}"
        )  # LeRobot keeps the N dim
        return arr


class RemoteModel(Model):
    """Weightless client to an OpenPI-WebSocket policy server: ships the adapter's request
    dict, returns the server's chunk. All pre/post-processing lives in the adapter + server.

    Not batchable: each :meth:`infer` is one WebSocket request for one env and always adds a
    single leading batch dim, and the OpenPI server protocol currently has no batched-request
    shape. Do not wrap in :class:`~hud.agents.robot.batching.BatchedModel` — use one
    :class:`~hud.agents.robot.agent.RobotAgent` per concurrent rollout instead.
    """

    def __init__(
        self, host: str = "localhost", port: int = 8000, *, response_key: str = "actions"
    ) -> None:
        self.host = host
        self.port = port
        #: Key under which the server returns the chunk — "actions" (stock OpenPI) or "action" (Cosmos).
        self.response_key = response_key
        self._client: Any = None

    def connect(self) -> None:
        """Open the websocket (idempotent); blocks until the server is up."""
        if self._client is None:
            mod: Any = importlib.import_module("openpi_client.websocket_client_policy")

            print(
                f"[agent] connecting to openpi server ws://{self.host}:{self.port} — on hold...",
                flush=True,
            )
            self._client = mod.WebsocketClientPolicy(self.host, self.port)

    def infer(self, batch: Any) -> ActionArray:
        """Ship one request dict → the server's ``[T, A]`` chunk, returned as ``[1, T, A]``."""
        self.connect()  # lazy connect on first call (blocks until the server is up)
        chunk = np.asarray(self._client.infer(batch)[self.response_key], dtype=np.float32)
        return chunk[None]  # add the leading N=1 batch dim


__all__ = [
    "LeRobotModel",
    "Model",
    "RemoteModel",
]
