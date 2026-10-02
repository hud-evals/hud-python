"""Robot policies and the inference mechanics that run them."""

from __future__ import annotations

import asyncio
import importlib
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from .adapter import ActionArray


class Model:
    """A policy and its inference mechanics.

    Stateless, so one model can be shared and batched across concurrent rollouts; per-episode
    state such as the open-loop chunk belongs on the agent, hence no ``reset``.
    """

    def infer(self, batch: Any) -> ActionArray:
        """Run the policy on an ``[N, ...]`` batch and return an ``[N, T, A]`` chunk.

        Keep the leading ``N`` even when it is 1: ``ainfer`` and ``BatchedModel`` index along it.
        """
        raise NotImplementedError

    async def ainfer(self, batch: Any) -> ActionArray:
        """Run ``infer`` in a thread and return its single ``[T, A]`` row."""
        return (await asyncio.to_thread(self.infer, batch))[0]


class LeRobotModel(Model):
    """A LeRobot policy with its processors; ``infer`` takes one sample or a stacked batch.

    ``predict_action_chunk`` bypasses LeRobot's internal action queue, so nothing needs a reset.
    """

    def __init__(self, policy: Any, preprocess: Any, postprocess: Any) -> None:
        self.policy = policy
        self.preprocess = preprocess
        self.postprocess = postprocess
        self._first_inference = True

    def infer(self, batch: Any) -> ActionArray:
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
        assert arr.ndim == 3, f"expected [N, T, A] chunk, got {arr.shape}"
        return arr


class RemoteModel(Model):
    """Client to an OpenPI WebSocket policy server; the adapter and server own all processing.

    Not batchable: each ``infer`` is one request for one env, so use one ``RobotAgent`` per
    concurrent rollout instead of wrapping it in ``BatchedModel``.
    """

    def __init__(
        self, host: str = "localhost", port: int = 8000, *, response_key: str = "actions"
    ) -> None:
        self.host = host
        self.port = port
        # Server chunk key — "actions" (stock OpenPI) or "action" (Cosmos).
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
        self.connect()
        chunk = np.asarray(self._client.infer(batch)[self.response_key], dtype=np.float32)
        return chunk[None]
