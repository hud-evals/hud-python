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
    from .tracer import RobotTracer

# ─── LeRobot convention (isolated, explicit, pure function) ──────────────────


def lerobot_infer(policy: Any, preprocess: Any, postprocess: Any, batch: Any) -> np.ndarray:
    """Full LeRobot inference: ``preprocess`` → ``select_action`` → ``postprocess``."""
    import torch

    with torch.no_grad():
        action = postprocess(policy.select_action(preprocess(batch)))
    return action.squeeze(0).cpu().numpy()


def lerobot_chunk_infer(policy: Any, preprocess: Any, postprocess: Any, batch: Any) -> np.ndarray:
    """Chunked sibling of :func:`lerobot_infer`"""
    import torch

    with torch.no_grad():
        chunk = postprocess(policy.predict_action_chunk(preprocess(batch)))
    return chunk.squeeze(0).float().cpu().numpy()


# ─── the abstraction ──────────────────────────────────────────────────────────


class Model:
    """Owns a policy and its inference mechanics.

    Lifecycle (driven by :class:`~hud.agents.robot.agent.RobotAgent`):

    - :meth:`reset` once per episode — reset per-episode model state (e.g.
      LeRobot's open-loop action queue).
    - :meth:`ainfer` every step — the awaited entry point the harness calls;
      defaults to running :meth:`infer` in a worker thread.
    - :meth:`infer` every step — run the policy on a prepared batch.
    """

    #: Optional per-step platform tracer (one span per env step, keyframes at
    #: fresh chunks). The harness attaches a default one when HUD telemetry is
    #: configured; models that know their chunk boundaries emit through it.
    tracer: RobotTracer | None = None

    def reset(self) -> None:
        """Reset per-episode model state. Override when the policy is stateful."""

    def infer(self, batch: Any) -> np.ndarray:
        """Run the policy on a prepared batch → a 1-D action vector. Must implement."""
        raise NotImplementedError

    async def ainfer(self, batch: Any) -> np.ndarray:
        """Awaited inference entry point — what the harness calls each step.

        Default: run the blocking :meth:`infer` in a worker thread so the event
        loop stays free.
        """
        return await asyncio.to_thread(self.infer, batch)


# TODO: define a general chunk -> action class model side. `Ensembler` is the
class Ensembler:
    """Temporal action ensembling: reduce overlapping action chunks to one action
    per step. Used by chunked policies (ACT, CogACT, pi0, VLA-JEPA).
    """

    def __init__(self, horizon: int = 7, alpha: float = 0.1) -> None:
        self.horizon = int(horizon)
        self.alpha = float(alpha)
        self._history: deque[np.ndarray] = deque(maxlen=self.horizon)

    def reset(self) -> None:
        """Clear the per-episode chunk history."""
        self._history.clear()

    def __call__(self, chunk: np.ndarray) -> np.ndarray:
        """Push the freshly inferred ``[chunk_size, action_dim]`` chunk; return one action."""
        self._history.append(np.asarray(chunk, dtype=np.float32))
        n = len(self._history)
        # Time-align: the chunk pushed i steps ago contributes its row i (its
        # forecast for the current timestep); the newest chunk contributes row 0.
        preds = np.stack([c[i] for i, c in zip(range(n - 1, -1, -1), self._history)])
        ref = preds[-1]  # newest opinion = inferred from the freshest observation
        cos = np.sum(preds * ref, axis=1) / (
            np.linalg.norm(preds, axis=1) * np.linalg.norm(ref) + 1e-7
        )
        weights = np.exp(self.alpha * cos)
        weights = weights / weights.sum()
        return np.sum(weights[:, None] * preds, axis=0)


class LeRobotModel(Model):
    """Wraps a LeRobot policy with its pre- and post-processor pipelines.

    Ships the LeRobot inference convention via :func:`lerobot_infer`. Subclass and
    override :meth:`infer` for non-standard policies (e.g. realtime chunk models),
    while keeping :meth:`reset` and ``policy`` / ``preprocess`` / ``postprocess``.

    Pass an :class:`Ensembler` to swap the default open-loop path (``select_action``
    pops a chunk, executed step-by-step) for per-step re-inference + temporal
    ensembling. ``ensembler=None`` (the default) keeps the pop-the-queue path.
    """

    def __init__(
        self, policy: Any, preprocess: Any, postprocess: Any, ensembler: Ensembler | None = None
    ) -> None:
        self.policy = policy
        self.preprocess = preprocess
        self.postprocess = postprocess
        #: Optional chunk->action reducer. When set, :meth:`infer` re-infers a
        #: chunk every step and ensembles it instead of popping ``select_action``.
        self.ensembler = ensembler
        #: Flipped to False after the first forward; used to print the one-time
        #: CUDA/flow-matching warmup message.
        self._first_inference = True
        self._step = 0  # env-step index within the episode (for the tracer)

    def reset(self) -> None:
        """Reset LeRobot's open-loop action queue (and the ensembler) for the new episode."""
        if hasattr(self.policy, "reset"):
            self.policy.reset()
        if self.ensembler is not None:
            self.ensembler.reset()
        self._step = 0

    def _queue_len(self) -> int | None:
        """Length of LeRobot's open-loop action queue, or ``None`` if unknown.

        Handles both conventions: the old single deque ``policy._action_queue``
        (pi05) and the new per-key dict ``policy._queues[ACTION]`` (VLA-JEPA).
        """
        queue = getattr(self.policy, "_action_queue", None)
        if queue is None:
            # Newer convention: a dict of deques keyed by feature constant. The
            # action key is the literal "action" (lerobot.utils.constants.ACTION).
            queues = getattr(self.policy, "_queues", None)
            if isinstance(queues, dict):
                queue = queues.get("action")
        try:
            return None if queue is None else len(queue)
        except TypeError:
            return None

    def infer(self, batch: Any) -> np.ndarray:
        """Run one inference step work also with a ``batch`` (with first-inference log + tracing).

        Default (no :attr:`ensembler`): :func:`lerobot_infer` pops the open-loop
        queue; fresh chunk iff the queue was empty. Ensembling: re-infer every
        step via :func:`lerobot_chunk_infer`, reduced to one action. A step that
        computes a fresh chunk is flagged as a tracer keyframe.
        """
        if self._first_inference:
            print(
                "[agent] first inference — flow-matching/CUDA warmup on this call, "
                "may take a while; subsequent steps will be fast",
                flush=True,
            )

        if self.ensembler is not None:
            chunk = lerobot_chunk_infer(self.policy, self.preprocess, self.postprocess, batch)
            result = self.ensembler(chunk)
            keyframe, chunk_len = True, len(chunk)
        else:
            before = self._queue_len()
            result = lerobot_infer(self.policy, self.preprocess, self.postprocess, batch)
            # Fresh chunk iff the queue was empty going in. The queued actions are
            # pre-postprocess (normalized), so only the horizon is recorded: the
            # popped action + whatever select_action left queued.
            after = self._queue_len()
            keyframe = (before == 0) or (before is None and self._step == 0)
            chunk_len = (after + 1) if (keyframe and after is not None) else None

        if self._first_inference:
            print("[agent] first inference done — inference is now fast", flush=True)
            self._first_inference = False

        # TODO Clean
        if self.tracer is not None:
            self.tracer.emit_step(
                batch, result, step=self._step, keyframe=bool(keyframe), chunk_len=chunk_len
            )
        self._step += 1
        return result


__all__ = [
    "Ensembler",
    "LeRobotModel",
    "Model",
    "lerobot_chunk_infer",
    "lerobot_infer",
]
