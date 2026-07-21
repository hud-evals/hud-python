"""``RobotAgent`` — drive one ``robot`` env with an open-loop chunk queue.

Subclass, set ``self.model`` and ``self.adapter`` in ``__init__``, and the base
owns the rest: connect to the ``robot`` capability (claiming a slot token from
``run.started`` when present), read the contract, run until the env terminates.

Vectorized sims still look like one scalar connection per rollout; concurrent
rollouts coalesce GPU forwards through :class:`~.batching.BatchedModel`.

Most policies use :class:`~.adapter.LeRobotAdapter`; a policy whose spaces
match the env natively can set ``adapter = None`` (raw pass-through).
"""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

from hud.agents.base import Agent
from hud.capabilities.robot import RobotClient
from hud.telemetry.robot import TraceRecorder

if TYPE_CHECKING:
    from hud.eval.run import Run

    from .adapter import ActionArray, Adapter
    from .model import Model

ROBOT_PROTOCOL = "openpi/0"


class RobotAgent(Agent):
    """Drive a ``robot`` env with one open-loop chunk queue.

    **Subclass contract:** in ``__init__`` set ``self.model`` (a
    :class:`~.model.Model`) and ``self.adapter`` (an :class:`~.adapter.Adapter`,
    or ``None`` for raw pass-through).
    """

    robot_protocol: ClassVar[str] = ROBOT_PROTOCOL
    #: Max control ticks before the episode is cut off. Subclasses may override.
    max_steps: ClassVar[int] = 520
    #: How often (in steps) to print a step-progress line. 0 = off.
    log_every: ClassVar[int] = 20
    #: Opt-in: also save a LeRobot v3 dataset of every (obs, action) pair.
    #: Telemetry streams regardless; see :mod:`.dataset`.
    save: bool = False

    #: Runs the policy (preprocess -> forward -> postprocess). Subclasses set this.
    model: Model | None = None
    #: Translates env<->policy spaces. Subclasses set this; ``None`` = raw pass-through.
    adapter: Adapter | None = None

    async def __call__(self, run: Run, *, max_steps: int | None = None) -> None:
        """The generic rollout contract: one run, one scalar robot connection.

        ``max_steps`` caps control ticks; omit it to use the class ``max_steps`` (520).
        """
        if self.model is None:
            raise RuntimeError(f"{type(self).__name__} must set self.model in __init__")
        prompt = run.prompt
        if not isinstance(prompt, str):
            raise TypeError(f"run.prompt must be a str, got {type(prompt).__name__}: {prompt!r}")

        # Per-episode slot token from tasks.start (opaque; env put it under "robot").
        # Single-env templates may omit it — a None claim binds the sole claimed
        # slot; vectorized bridges reject the ambiguity at connect.
        robot_info = run.started.get("robot")
        raw_token = robot_info.get("token") if isinstance(robot_info, dict) else None
        token = raw_token if isinstance(raw_token, str) else None

        robot = await RobotClient.connect(run.client.binding(self.robot_protocol), token=token)
        try:
            action_space, obs_space = robot.spaces()
            if self.adapter is not None:
                self.adapter.bind(action_space, obs_space)
                self.adapter.reset()

            obs = await robot.get_observation()
            fps = robot.get_control_rate()
            # Contract labels: obs_space for state, action names for InferenceStep plots.
            recorder = TraceRecorder(
                run=run,
                fps=fps,
                obs_space=obs_space,
                action_names=action_space.get("names"),
            )
            writer = None
            if self.save:
                from .dataset import DatasetWriter

                writer = DatasetWriter(robot.contract, fps=fps)

            print(f"[agent] episode started: {prompt!r}", flush=True)
            try:
                await self._loop(
                    robot,
                    obs,
                    prompt,
                    recorder,
                    writer,
                    max_steps=self.max_steps if max_steps is None else max_steps,
                )
            finally:
                # Flush video tails / commit the buffered episode even when the
                # rollout raises mid-loop.
                recorder.close()
                if writer is not None:
                    writer.end_episode()
        finally:
            await robot.close()
        run.trace.status = "completed"
        run.trace.content = "done"

    async def _loop(
        self,
        robot: RobotClient,
        obs: dict[str, Any],
        prompt: str,
        recorder: TraceRecorder,
        writer: Any,
        *,
        max_steps: int,
    ) -> None:
        """Open-loop chunk queue: ainfer refills, then execute one action per tick."""
        model = self.model
        assert model is not None
        adapter = self.adapter
        chunk: deque[ActionArray] = deque()

        for step in range(max_steps):
            # Record every frame, including the terminal one.
            recorder.record_observation(obs["data"], tick=step)
            # Already done (including a pre-terminated first obs) → don't act.
            if bool(np.asarray(obs["terminated"]).reshape(-1)[0]):
                if step:
                    print(f"[agent] terminated at step {step}", flush=True)
                break

            if not chunk:  # refill with a fresh forward (BatchedModel coalesces ainfer)
                batch = adapter.adapt_observation(obs, prompt) if adapter else obs
                rows = np.atleast_2d(await model.ainfer(batch))
                if adapter is not None:  # e.g. deltas -> absolute vs the query obs
                    rows = adapter.adapt_chunk(rows, obs)
                chunk.extend(rows)
                recorder.record_inference(rows, tick=step)

            action = chunk.popleft()
            if adapter is not None:  # per-step execution-time hook (default identity)
                action = adapter.adapt_action(action, obs)
            if writer is not None:
                writer.add(obs["data"], np.asarray(action), task=prompt)
            await robot.send_action(action)

            if self.log_every and step % self.log_every == 0:
                print(f"[agent] step {step}/{max_steps}", flush=True)
            obs = await robot.get_observation()
        else:
            print(f"[agent] reached max_steps={max_steps}", flush=True)


__all__ = ["ROBOT_PROTOCOL", "RobotAgent"]
