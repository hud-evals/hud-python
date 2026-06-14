"""The ``Model``: wraps a policy and owns its inference mechanics.

A ``Model`` knows *how to run* a policy (preprocess → forward → postprocess); the
harness only awaits ``model.ainfer(batch)``. Use :class:`LeRobotModel` for stock
LeRobot checkpoints; subclass :class:`Model` and implement ``infer`` otherwise.
"""

from __future__ import annotations

import asyncio
from collections import deque
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from ._types import ActionArray

# ─── LeRobot convention (isolated, explicit, pure function) ──────────────────


def lerobot_infer(policy: Any, preprocess: Any, postprocess: Any, batch: Any) -> ActionArray:
    """Infer one ``[T, A]`` chunk: ``preprocess`` → ``predict_action_chunk`` →
    ``postprocess``."""
    import torch  # pyright: ignore[reportMissingImports]

    torch_mod: Any = torch
    with torch_mod.no_grad():
        chunk = postprocess(policy.predict_action_chunk(preprocess(batch)))
    return chunk.squeeze(0).float().cpu().numpy()


# ─── the abstraction ──────────────────────────────────────────────────────────


class Model:
    """Owns a policy and its inference mechanics.

    Driven by :class:`~hud.agents.robot.agent.RobotAgent`: :meth:`reset` once per
    episode, then :meth:`ainfer` (awaited; defaults to :meth:`infer` in a thread) each
    inference. Returns a ``[T, A]`` chunk (``T = 1`` for single-action policies).
    """

    def reset(self) -> None:
        """Reset per-episode model state. Override when the policy is stateful."""

    def infer(self, batch: Any) -> ActionArray:
        """Run the policy on a prepared batch → a ``[T, A]`` action chunk. Must implement."""
        raise NotImplementedError

    async def ainfer(self, batch: Any) -> ActionArray:
        """Awaited entry point; runs blocking :meth:`infer` in a worker thread."""
        return await asyncio.to_thread(self.infer, batch)


# TODO: define a general chunk -> action class model side. `Ensembler` is the
class Ensembler:
    """Temporal action ensembling: reduce overlapping action chunks to one action
    per step. Used by chunked policies (ACT, CogACT, pi0, VLA-JEPA).
    """

    def __init__(self, horizon: int = 7, alpha: float = 0.1) -> None:
        self.horizon = int(horizon)
        self.alpha = float(alpha)
        self._history: deque[ActionArray] = deque(maxlen=self.horizon)

    def reset(self) -> None:
        """Clear the per-episode chunk history."""
        self._history.clear()

    def __call__(self, chunk: ActionArray) -> ActionArray:
        """Push the freshly inferred ``[chunk_size, action_dim]`` chunk; return one action."""
        self._history.append(np.asarray(chunk, dtype=np.float32))
        n = len(self._history)
        # Time-align: the chunk pushed i steps ago contributes its row i (its
        # forecast for the current timestep); the newest chunk contributes row 0.
        preds = np.stack([c[i] for i, c in zip(range(n - 1, -1, -1), self._history, strict=False)])
        ref = preds[-1]  # newest opinion = inferred from the freshest observation
        cos = np.sum(preds * ref, axis=1) / (
            np.linalg.norm(preds, axis=1) * np.linalg.norm(ref) + 1e-7
        )
        weights = np.exp(self.alpha * cos)
        weights = weights / weights.sum()
        return np.sum(weights[:, None] * preds, axis=0)


class LeRobotModel(Model):
    """LeRobot policy with pre/post-processors; infers via :func:`lerobot_infer`.

    Pass an :class:`Ensembler` to reduce overlapping chunks to one action per step.
    """

    def __init__(
        self, policy: Any, preprocess: Any, postprocess: Any, ensembler: Ensembler | None = None
    ) -> None:
        self.policy = policy
        self.preprocess = preprocess
        self.postprocess = postprocess
        #: Optional chunk->action reducer. When set, :meth:`infer` ensembles each
        #: freshly inferred chunk into a single action (a length-1 chunk).
        self.ensembler = ensembler
        #: Flipped to False after the first forward; used to print the one-time
        #: CUDA/flow-matching warmup message.
        self._first_inference = True

    def reset(self) -> None:
        """Reset LeRobot's open-loop action queue (and the ensembler) for the new episode."""
        if hasattr(self.policy, "reset"):
            self.policy.reset()
        if self.ensembler is not None:
            self.ensembler.reset()

    def infer(self, batch: Any) -> ActionArray:
        """Infer one ``[T, A]`` chunk; with an :attr:`ensembler`, reduce to length 1."""
        if self._first_inference:
            print(
                "[agent] first inference — flow-matching/CUDA warmup on this call, "
                "may take a while; subsequent steps will be fast",
                flush=True,
            )

        chunk = lerobot_infer(self.policy, self.preprocess, self.postprocess, batch)
        if self.ensembler is not None:
            chunk = self.ensembler(chunk)[None, :]  # [A] -> length-1 chunk [1, A]

        if self._first_inference:
            print("[agent] first inference done — inference is now fast", flush=True)
            self._first_inference = False

        return chunk


__all__ = [
    "Ensembler",
    "LeRobotModel",
    "Model",
    "lerobot_infer",
]
