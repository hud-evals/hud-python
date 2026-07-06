"""``RobotAgent`` — the one robot harness, driving N >= 1 env slots.

Subclass, set ``self.model`` and ``self.adapter`` in ``__init__``, and the base
owns the rest: connect to the ``robot`` capability, read the contract, run the
open-loop chunk queue until the env terminates. A plain single env is a batch
of one — the same loop drives a vectorized env's whole ``[N, ...]`` batch over
one connection (one batched forward per refill).

Two entries, one loop:

- ``__call__(run)`` — the generic rollout contract (one run, one trace).
- ``drive(runs, client)`` — the grouped-eval entry
  (:func:`hud.eval.run.rollout_group`): N runs sharing one env instance, spans
  recorded per slot onto each run's trace.

Most policies use :class:`~.adapter.LeRobotAdapter`; a policy whose spaces
match the env natively can set ``adapter = None`` (raw pass-through).
"""

from __future__ import annotations

import asyncio
from collections import deque
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

from hud.agents.base import Agent
from hud.capabilities.robot import RobotClient
from hud.telemetry.robot import TraceRecorder

if TYPE_CHECKING:
    from hud.clients.client import HudClient
    from hud.eval.run import Run

    from .adapter import Adapter
    from .model import Model

ROBOT_PROTOCOL = "openpi/0"


class RobotAgent(Agent):
    """Drive a ``robot`` env — single or vectorized — with one open-loop chunk queue.

    **Subclass contract:** in ``__init__`` set ``self.model`` (a
    :class:`~.model.Model`) and ``self.adapter`` (an :class:`~.adapter.Adapter`,
    or ``None`` for raw pass-through). ``model.infer`` is batch-shaped
    (``[N, ...] -> [N, T, A]``), so the same subclass drives both shapes.
    """

    robot_protocol: ClassVar[str] = ROBOT_PROTOCOL
    #: Max control ticks before the episode is cut off.
    max_steps: ClassVar[int] = 520
    #: How often (in steps) to print a step-progress line. 0 = off.
    log_every: ClassVar[int] = 20
    #: Opt-in: also save a LeRobot v3 dataset of every (obs, action) pair
    #: (single-env runs only). Telemetry streams regardless; see :mod:`.dataset`.
    save: bool = False

    #: Runs the policy (preprocess -> forward -> postprocess). Subclasses set this.
    model: Model | None = None
    #: Translates env<->policy spaces. Subclasses set this; ``None`` = raw pass-through.
    adapter: Adapter | None = None

    async def __call__(self, run: Run, *, max_steps: int | None = None) -> None:
        """The generic rollout contract: one run, one trace."""
        await self.drive([run], run.client, max_steps=max_steps)
        run.trace.status = "completed"
        run.trace.content = "done"

    async def drive(
        self, runs: list[Run], client: HudClient, *, max_steps: int | None = None
    ) -> None:
        """Drive every env slot to termination, recording onto ``runs[i]``'s trace.

        ``len(runs)`` must match the env's batch size (the wire framing tells us:
        scalar ``terminated`` = 1, an ``[N]`` mask = N).
        """
        if self.model is None:
            raise RuntimeError(f"{type(self).__name__} must set self.model in __init__")
        prompt = runs[0].prompt
        if not isinstance(prompt, str):
            raise TypeError(f"run.prompt must be a str, got {type(prompt).__name__}: {prompt!r}")

        robot = await RobotClient.connect(client.binding(self.robot_protocol))
        try:
            _, obs_space = robot.spaces()
            if self.adapter is not None:
                self.adapter.bind(*robot.spaces())
                self.adapter.reset()

            obs = await robot.get_observation()
            single = np.ndim(obs["terminated"]) == 0  # wire framing: scalar vs [N] mask
            n = 1 if single else int(np.asarray(obs["terminated"]).shape[0])
            if len(runs) != n:
                raise ValueError(f"got {len(runs)} runs for an env batch of {n}")

            fps = robot.get_control_rate()
            recorders = [
                # A live run (single path) records through it so steps land on
                # run.trace for training; grouped receipts emit by trace id.
                TraceRecorder(run=r, fps=fps, obs_space=obs_space)
                if r._client is not None
                else TraceRecorder(trace_id=r.trace_id, fps=fps, obs_space=obs_space)
                for r in runs
            ]
            writer = None
            if self.save:
                if n == 1:
                    from .dataset import DatasetWriter

                    writer = DatasetWriter(robot.contract, fps=fps)
                else:
                    print("[agent] save=True is single-env only; streaming telemetry", flush=True)

            print(f"[agent] episode started: {prompt!r} (n={n})", flush=True)
            await self._loop(
                robot,
                obs,
                prompt,
                recorders,
                writer,
                single=single,
                max_steps=max_steps or self.max_steps,
            )
            for rec in recorders:
                rec.close()
            if writer is not None:
                writer.end_episode()
        finally:
            await robot.close()

    async def _loop(
        self,
        robot: RobotClient,
        obs: dict[str, Any],
        prompt: str,
        recorders: list[TraceRecorder],
        writer: Any,
        *,
        single: bool,
        max_steps: int,
    ) -> None:
        """One batched forward per refill; execute chunks open-loop per slot."""
        adapter = self.adapter
        n = len(recorders)
        chunks: list[deque[np.ndarray]] = [deque() for _ in range(n)]
        ever_done = np.zeros(n, dtype=bool)

        for step in range(max_steps):
            done = np.atleast_1d(np.asarray(obs["terminated"], dtype=bool)).reshape(-1)
            for i in np.nonzero(done)[0]:  # a reset slot re-infers for its new episode
                chunks[i].clear()
            ever_done |= done
            if step and ever_done.all():
                print(f"[agent] all slots terminated at step {step}", flush=True)
                break

            # Batched view of the observation: single framing lifts to a batch of one.
            data = obs["data"] if not single else {k: v[None] for k, v in obs["data"].items()}
            for i, rec in enumerate(recorders):
                if not ever_done[i]:
                    rec.record_observation({k: v[i] for k, v in data.items()}, tick=step)

            if any(not c for c in chunks):  # refill spent slots with a fresh forward
                # The adapter sees the wire framing (unbatched when single), so a
                # BatchedModel can still stack samples across concurrent rollouts.
                batch = adapter.adapt_observation(obs, prompt) if adapter else obs
                if n == 1:
                    # ainfer is the coalescing point for cross-rollout batching
                    # (BatchedModel), so the single slot goes through it.
                    chunk = np.atleast_2d(await self.model.ainfer(batch))[None]  # [1, T, A]
                else:
                    chunk = np.asarray(await asyncio.to_thread(self.model.infer, batch))
                for i, c in enumerate(chunks):
                    if not c:
                        rows = chunk[i]
                        if adapter is not None:  # e.g. deltas -> absolute vs the query obs
                            slot = {"data": {k: v[i] for k, v in data.items()}}
                            rows = adapter.adapt_chunk(rows, slot)
                        c.extend(rows)
                        recorders[i].record_inference(rows, tick=step)

            raw = [chunks[i].popleft() for i in range(n)]
            if adapter is not None:  # per-step execution-time hook (default identity)
                raw = [
                    adapter.adapt_action(a, {"data": {k: v[i] for k, v in data.items()}})
                    for i, a in enumerate(raw)
                ]
            action = raw[0] if single else np.stack(raw)
            if writer is not None:
                writer.add(obs["data"], np.asarray(raw[0]), task=prompt)
            await robot.send_action(action)

            if self.log_every and step % self.log_every == 0:
                live = int((~ever_done).sum())
                print(f"[agent] step {step}/{max_steps} live={live}/{n}", flush=True)
            obs = await robot.get_observation()
        else:
            print(f"[agent] reached max_steps={max_steps}", flush=True)


__all__ = ["ROBOT_PROTOCOL", "RobotAgent"]
