"""Base v6 agent for any env that exposes a ``robot`` capability.

Subclass :class:`RobotAgent`, set ``self.model`` and ``self.adapter`` in
``__init__``, and the base owns the rest.

The base calls the adapter and model at the right moments::

    setup_robot      -> adapter.bind(spaces)                          # once after connect
    on_episode_start -> model.reset(); adapter.reset()                # once per episode
    select_action    -> adapt_observation -> model.ainfer -> pop chunk -> adapt_action

``model.ainfer`` always returns a ``[T, A]`` chunk; :meth:`RobotAgent.select_action`
executes it open-loop, re-inferring only once the active chunk is spent.

Most policies use :class:`~hud.agents.robot.adapter.DefaultAdapter`; a policy whose
spaces match the env natively can set ``adapter = None`` (raw pass-through).
"""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

from hud.agents.base import Agent
from hud.agents.types import InferenceStep, ObservationStep
from hud.capabilities.robot import RobotClient

if TYPE_CHECKING:
    from hud.eval.rollout import Run

    from .adapter import Adapter
    from .model import Model

ROBOT_PROTOCOL = "robot/0.1"


class RobotAgent(Agent):
    """Drive a ``robot`` side-channel for one :class:`~hud.client.Run`.

    **Subclass contract:** in ``__init__`` set ``self.model`` (a
    :class:`~hud.agents.robot.model.Model`) and ``self.adapter`` (an
    :class:`~hud.agents.robot.adapter.Adapter`, or ``None`` for raw pass-through).

    **Override if needed:**

    - :attr:`robot_protocol` — class attr if not ``robot/0.1``
    - :meth:`on_episode_start` — mostly internal; override (with ``super()``) to
      add per-episode setup (e.g. reading the env contract).
    - :meth:`should_stop` — custom early-exit condition beyond ``obs["terminated"]``
    - :meth:`select_action` — only for a wholly different inference path
    - :attr:`log_every` — class-level print frequency (0 = off)
    """

    robot_protocol: ClassVar[str] = ROBOT_PROTOCOL
    #: How often (in steps) to print a step-progress line. 0 = off.
    log_every: ClassVar[int] = 20

    #: Runs the policy (preprocess → forward → postprocess). Subclasses set this.
    model: Model | None = None
    #: Translates env<->policy spaces. Subclasses set this; ``None`` = raw pass-through.
    adapter: Adapter | None = None

    _prompt: str = ""
    #: The env's action / observation contract features (from ``client.spaces()``),
    #: named ``_env_*`` to mark them as env-side values (not the policy's spaces).
    _env_action_space: dict[str, Any]
    _env_obs_space: dict[str, Any]
    #: Unexecuted tail of the current policy chunk; popped one action per step.
    _active_chunk: deque[np.ndarray]
    #: The live run + control-tick index, so ``select_action`` can record its own InferenceStep.
    _run: Run
    _tick: int


    def setup_robot(self, client: RobotClient) -> None:
        """Discover the env's action/observation layout and bind the adapter to it."""
        self._env_action_space, self._env_obs_space = client.spaces()
        if self.adapter is not None:
            self.adapter.bind(self._env_action_space, self._env_obs_space)

    def on_episode_start(self, run: Run, client: RobotClient, *, prompt: str) -> None:
        """Called once before the observe/act loop begins.

        Stores the prompt, resets the model and adapter. Mostly internal — the base
        always calls it. Override (calling ``super()`` first) only when per-episode
        env-contract reading or extra setup is needed (e.g. a realtime chunk-streaming
        agent reads inference mode/threshold from the contract here).
        """
        self._prompt = prompt
        self._active_chunk = deque()
        self._run = run
        self._tick = 0
        if self.model is not None:
            self.model.reset()
        if self.adapter is not None:
            self.adapter.reset()

    def should_stop(self, obs: dict[str, Any], *, step: int, max_steps: int) -> bool:
        """Return True to break out of the step loop (before ``select_action``)."""
        return bool(obs.get("terminated"))

    async def select_action(self, obs: dict[str, Any]) -> np.ndarray:
        """pop the next model action — re-inferring a fresh ``[T, A]`` chunk via ``model.ainfer`` once the active one is spent (a length-1 chunk just re-infers every step) — and adapt it to env space; override only for a wholly different inference path"""
        if self.model is None:
            raise RuntimeError(f"{type(self).__name__} must set self.model in __init__")
        if not self._active_chunk:
            batch = (
                obs if self.adapter is None else self.adapter.adapt_observation(obs, self._prompt)
            )
            chunk = np.atleast_2d(await self.model.ainfer(batch))  # [T, A]
            self._active_chunk = deque(chunk)
            self._run.record(
                InferenceStep(tick=self._tick, chunk=chunk.tolist(), chunk_length=len(chunk))
            )
        self._tick += 1
        raw = self._active_chunk.popleft()
        return raw if self.adapter is None else self.adapter.adapt_action(raw, obs)

    async def __call__(self, run: Run, *, max_steps: int | None = None) -> None:
        if max_steps is None:
            max_steps = getattr(self, "max_steps", 520)
        cap = run.client.binding(self.robot_protocol)
        client = await RobotClient.connect(cap)
        try:
            self.setup_robot(client)
            prompt = run.prompt
            if not isinstance(prompt, str):
                raise TypeError(
                    f"run.prompt must be a str, got {type(prompt).__name__}: {prompt!r}"
                )
            self.on_episode_start(run, client, prompt=prompt)
            print(f"[agent] episode started: {prompt!r} (max_steps={max_steps})", flush=True)

            for step in range(max_steps):
                obs = await client.get_observation()
                run.record(
                    ObservationStep.from_obs(obs, tick=step, obs_space=self._env_obs_space)
                )

                if self.should_stop(obs, step=step, max_steps=max_steps):
                    print(f"[agent] env reported terminated at step {step}", flush=True)
                    break

                action = await self.select_action(obs)
                await client.send_action(action)

                if self.log_every and step % self.log_every == 0:
                    preview = np.array2string(action, precision=3, suppress_small=True)
                    print(f"[agent] step {step}/{max_steps} action={preview}", flush=True)
            else:
                print(f"[agent] reached max_steps={max_steps}", flush=True)

            run.trace.status = "completed"
            run.trace.content = "done"
        finally:
            await client.close()


__all__ = ["ROBOT_PROTOCOL", "RobotAgent"]
