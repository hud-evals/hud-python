"""``RobotAgent``: drive one ``robot`` env with an open-loop chunk queue."""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

from hud.agents.base import Agent
from hud.capabilities.robot import RobotClient
from hud.telemetry.robot.recorder import TraceRecorder

if TYPE_CHECKING:
    from hud.eval.run import Run

    from .adapter import ActionArray, Adapter
    from .model import Model

ROBOT_PROTOCOL = "openpi/0"


class RobotAgent(Agent):
    """Drive a ``robot`` env with one open-loop chunk queue.

    Subclasses set ``model`` and ``adapter`` in ``__init__``; a ``None`` adapter passes data raw.
    """

    robot_protocol: ClassVar[str] = ROBOT_PROTOCOL
    # Control ticks before the episode is cut off.
    max_steps: ClassVar[int] = 520
    # How often (in steps) to print a step-progress line. 0 = off.
    log_every: ClassVar[int] = 20
    # Also record a LeRobot v3 dataset of every (obs, action) pair; telemetry streams regardless.
    save: bool = False

    model: Model | None = None
    adapter: Adapter | None = None

    def should_stop(self, obs: dict[str, Any], *, step: int, max_steps: int) -> bool:
        """Return whether to stop before selecting the next action."""
        return bool(np.asarray(obs["terminated"]).reshape(-1)[0])

    async def __call__(self, run: Run, *, max_steps: int | None = None) -> None:
        """Run one episode; ``max_steps`` overrides the class's control-tick cap."""
        if self.model is None:
            raise RuntimeError(f"{type(self).__name__} must set self.model in __init__")
        prompt = run.prompt
        if not isinstance(prompt, str):
            raise TypeError(f"run.prompt must be a str, got {type(prompt).__name__}: {prompt!r}")

        cap = run.client.binding(self.robot_protocol)
        # The episode's slot token. Single-env templates may omit it: a ``None`` claim binds the
        # sole slot, and vectorized bridges reject it at connect.
        token = run.bindings.get(cap.name, {}).get("token")
        robot = await RobotClient.connect(cap, token=token)
        try:
            action_space, obs_space = robot.spaces()
            if self.adapter is not None:
                self.adapter.bind(action_space, obs_space)
                self.adapter.reset()

            obs = await robot.get_observation()
            fps = robot.get_control_rate()
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
                # Drop the wire before telemetry flush so a claimed slot with no
                # further actions doesn't stall co-located peers at the barrier.
                await robot.close()
                recorder.close()
                if writer is not None:
                    writer.end_episode()
        finally:
            await robot.close()
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
            # Stop before acting, including on a pre-terminated first observation.
            if self.should_stop(obs, step=step, max_steps=max_steps):
                if step:
                    print(f"[agent] stopped at step {step}", flush=True)
                break

            if not chunk:
                batch = adapter.adapt_observation(obs, prompt) if adapter else obs
                rows = np.atleast_2d(await model.ainfer(batch))
                if adapter is not None:
                    rows = adapter.adapt_chunk(rows, obs)
                    if adapter.chunk_size is not None:
                        rows = rows[: adapter.chunk_size]
                chunk.extend(rows)
                recorder.record_inference(rows, tick=step)

            action = chunk.popleft()
            if adapter is not None:
                action = adapter.adapt_action(action, obs)
            if writer is not None:
                writer.add(obs["data"], np.asarray(action), task=prompt)
            await robot.send_action(action)

            if self.log_every and step % self.log_every == 0:
                print(f"[agent] step {step}/{max_steps}", flush=True)
            obs = await robot.get_observation()
            if (reward := obs.get("reward")) is not None:
                recorder.add_reward(reward)
        else:
            print(f"[agent] reached max_steps={max_steps}", flush=True)
