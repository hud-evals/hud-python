"""The ``Model``: wraps a policy and owns its inference mechanics.

The ``Model`` is the object that knows *how to run* a policy — preprocessing the
input batch, calling the forward pass, postprocessing the output. The agent harness
knows nothing about these details; it only awaits ``model.ainfer(batch)`` (which by
default just runs ``model.infer(batch)`` in a worker thread).

The framework ships :class:`LeRobotModel`, backed by :func:`lerobot_infer` — the
preprocess → ``policy.select_action`` → postprocess sandwich that every LeRobot
checkpoint needs. The free function is named explicitly so custom models can reuse
parts of it. A non-LeRobot policy just subclasses :class:`Model` and implements
``infer``.

Agent harness usage::

    batch  = adapter.adapt_observation(obs, prompt)   # Adapter's job
    raw    = await model.ainfer(batch)                 # Model's job
    action = adapter.adapt_action(raw, obs)            # Adapter's job
"""

from __future__ import annotations

import asyncio
from collections import deque
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from .tracer import RobotTracer

# ─── throughput counter (shared by the baseline + batched paths) ─────────────


class StepCounter:
    """Counts per-step model inferences for throughput (obs/s) measurement.

    One ``ainfer`` call == one env step for that lane, so summing across lanes
    (they all share this single module-level counter) gives the cell's total env
    steps. The asyncio loop is single-threaded, so a plain ``+= 1`` is race-free
    even with K lanes interleaving.
    """

    def __init__(self) -> None:
        self.count = 0

    def reset(self) -> None:
        self.count = 0

    def incr(self) -> None:
        self.count += 1


#: Process-wide step counter; reset around each cell by the runner.
STEP_COUNTER = StepCounter()


# ─── LeRobot convention (isolated, explicit, pure function) ──────────────────


def lerobot_infer(policy: Any, preprocess: Any, postprocess: Any, batch: Any) -> np.ndarray:
    """Run the LeRobot preprocess → forward → postprocess sandwich.

    This is the exact call sequence every LeRobot checkpoint requires for
    single-step inference: the ``preprocess`` pipeline (normalization, tokenization,
    device transfer), ``policy.select_action`` (the model forward + action-queue
    pop), and ``postprocess`` (unnormalization, absolute-action reconstruction).

    Pure by design (all dependencies passed in) so custom models can reuse it.
    """
    import torch

    with torch.no_grad():
        action = postprocess(policy.select_action(preprocess(batch)))
    return action.squeeze(0).cpu().numpy()


def lerobot_chunk_infer(policy: Any, preprocess: Any, postprocess: Any, batch: Any) -> np.ndarray:
    """Run the LeRobot preprocess → chunk-forward → postprocess sandwich.

    The chunked sibling of :func:`lerobot_infer`: calls
    ``policy.predict_action_chunk`` (not ``select_action``), so the postprocessor
    unnormalizes the whole ``[B, chunk_size, action_dim]`` chunk in one pass.
    Returns a ``[chunk_size, action_dim]`` array (batch dim squeezed). The policy
    must implement ``predict_action_chunk``.

    Pure by design (all dependencies passed in) so custom models can reuse it —
    e.g. feeding the chunk to an :class:`Ensembler`.
    """
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

        Default: run the blocking :meth:`infer` in a worker thread, so the event
        loop stays free (identical behavior to the old ``to_thread(infer)`` path).
        Override to await a shared resource instead — e.g. a ``BatchedModel`` parks
        the batch on a coalescing batcher and awaits its row.
        """
        STEP_COUNTER.incr()  # one ainfer == one env step (baseline lanes=1 path)
        return await asyncio.to_thread(self.infer, batch)


# TODO: define a general chunk -> action class model side. `Ensembler` is the
# first instance of that abstraction — a reducer that consumes the stream of
# (overlapping) action chunks a chunked policy emits and yields one action per
# step. Other reducers (open-loop pop-the-queue, RTC-style prefix stitching)
# should eventually share this interface so `LeRobotModel` can be parameterized
# by the chunk->action strategy instead of hardcoding `select_action`.
class Ensembler:
    """Reduce a stream of overlapping action chunks to one action per step.

    Temporal action ensembling (ACT's idea, with CogACT's adaptive weighting):
    a chunked policy predicts a ``[chunk_size, action_dim]`` chunk every step,
    and the chunk produced ``i`` steps ago made a forecast for *now* in its row
    ``i``. :meth:`__call__` keeps the last ``horizon`` chunks, time-aligns those
    forecasts, and returns their weighted average — closed-loop reactivity with
    the smoothness of consensus.

    Weights are ``softmax(alpha * cos_sim)`` against the newest prediction, so
    predictions that disagree with the freshest evidence are down-weighted
    (``alpha=0`` recovers ACT's uniform average). Port of the starVLA SimplerEnv
    eval client's ``AdaptiveEnsembler`` (``adaptive_ensemble.py``).

    Space-agnostic: it averages in whatever space it is fed, so place it AFTER
    the policy's postprocessor (chunks already in env/native units). Note any
    discretized dim (e.g. a binarized gripper) is averaged to a continuous value
    the caller must re-threshold.
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

    Ships the LeRobot inference convention via :func:`lerobot_infer`. A policy
    that deviates from the standard pipeline (e.g. a realtime chunk model) can
    subclass this and override :meth:`infer`, while still getting :meth:`reset`
    and access to ``policy`` / ``preprocess`` / ``postprocess`` for free.

    Pass an :class:`Ensembler` to switch from the default open-loop behavior
    (``select_action`` pops a chunk it executes step-by-step) to per-step
    re-inference + temporal ensembling: every step runs the full
    preprocess -> ``predict_action_chunk`` -> postprocess sandwich and reduces
    the resulting chunk to one action. ``ensembler=None`` (the default) keeps the
    original pop-the-queue path untouched.
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

        Handles both LeRobot queue conventions: the older single-deque form
        ``policy._action_queue`` (e.g. pi05) and the newer per-key dict form
        ``policy._queues[ACTION]`` (e.g. VLA-JEPA). Returns ``None`` only when
        neither form is present.
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
        """Run one inference step, with a one-time first-inference log + tracing.

        Two paths share the same logging / tracer / step-counter scaffolding and
        differ only in how the action is produced:

        - default (:attr:`ensembler` is ``None``) — :func:`lerobot_infer`
          (``select_action`` pops the open-loop queue). The step is a fresh chunk
          iff the queue was empty going in.
        - ensembling (:attr:`ensembler` set) — :func:`lerobot_chunk_infer` every
          step, reduced to one action by the ensembler. Every step re-infers, so
          every step is a fresh chunk.

        When a :attr:`tracer` is attached, each step emits a platform span; fresh
        chunks are stamped as keyframes carrying the chunk horizon — the
        decision-point markers in the trace viewer.
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

        if self.tracer is not None:
            self.tracer.emit_step(
                batch, result, step=self._step, keyframe=bool(keyframe), chunk_len=chunk_len
            )
        self._step += 1
        return result


__all__ = [
    "STEP_COUNTER",
    "Ensembler",
    "LeRobotModel",
    "Model",
    "StepCounter",
    "lerobot_chunk_infer",
    "lerobot_infer",
]
