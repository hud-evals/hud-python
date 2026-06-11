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
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np

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


class LeRobotModel(Model):
    """Wraps a LeRobot policy with its pre- and post-processor pipelines.

    Ships the LeRobot inference convention via :func:`lerobot_infer`. A policy
    that deviates from the standard pipeline (e.g. a realtime chunk model) can
    subclass this and override :meth:`infer`, while still getting :meth:`reset`
    and access to ``policy`` / ``preprocess`` / ``postprocess`` for free.
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

    def infer(self, batch: Any) -> np.ndarray:
        """Run :func:`lerobot_infer`, with a one-time first-inference log."""
        if self._first_inference:
            print(
                "[agent] first inference — flow-matching/CUDA warmup on this call, "
                "may take a while; subsequent steps will be fast",
                flush=True,
            )
        result = lerobot_infer(self.policy, self.preprocess, self.postprocess, batch)
        if self._first_inference:
            print("[agent] first inference done — inference is now fast", flush=True)
            self._first_inference = False
        return result


__all__ = ["STEP_COUNTER", "LeRobotModel", "Model", "StepCounter", "lerobot_infer"]
