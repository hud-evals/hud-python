"""``VecRobotAgent`` — drive a whole vectorized robot env over one connection.

The agent-side counterpart to :class:`~hud.environment.robot.isaac_bridge.IsaacBridge`: one
batched forward per tick. Receive an ``[N, ...]`` observation, run the model once to an
``[N, T, A]`` chunk, execute it open-loop per slot, and send one ``[N, A]`` action. The N
parallel episodes stream to the platform as one **Job** of per-episode traces via
:class:`~hud.agents.robot.record.VecRecorder` (slots split into fresh traces on each reset).

Set ``self.model`` (and ``self.adapter``) exactly as for :class:`~hud.agents.robot.agent.RobotAgent`;
the model's ``infer`` is already ``[N, ...] -> [N, T, A]``, so :class:`~hud.agents.robot.model.LeRobotModel`
works unchanged. Pair with :class:`~hud.agents.robot.adapter.VecLeRobotAdapter` for the batched obs.
"""

from __future__ import annotations

import asyncio
from collections import deque
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

from hud.capabilities.robot import RobotClient

from .record import VecRecorder

if TYPE_CHECKING:
    from hud.capabilities.base import Capability

    from .adapter import Adapter
    from .model import Model


class VecRobotAgent:
    """Drive an N-env robot batch as one Job of per-episode traces.

    **Subclass contract:** set ``self.model`` (a :class:`~hud.agents.robot.model.Model`) and
    ``self.adapter`` (a :class:`~hud.agents.robot.adapter.Adapter`) in ``__init__``.
    """

    model: Model | None = None
    adapter: Adapter | None = None
    #: Max control ticks before the run is cut off (the env auto-resets episodes within this).
    max_steps: ClassVar[int] = 520
    #: Step-progress print frequency. 0 = off.
    log_every: ClassVar[int] = 20

    async def run(
        self,
        cap: Capability,
        prompt: str,
        *,
        name: str,
        num_record: int = 4,
        seed: int | None = None,
        group_id: str | None = None,
        model_name: str | None = None,
    ) -> str:
        """Connect, drive the batch to ``max_steps`` (or all-terminated), return the Job URL."""
        if self.model is None:
            raise RuntimeError(f"{type(self).__name__} must set self.model in __init__")
        client = await RobotClient.connect(cap)
        try:
            action_space, obs_space = client.spaces()
            if self.adapter is not None:
                self.adapter.bind(action_space, obs_space)
            return await self._drive(
                client, prompt, name=name, num_record=num_record, seed=seed,
                group_id=group_id, model_name=model_name,
            )
        finally:
            await client.close()

    async def _drive(
        self, client: RobotClient, prompt: str, *, name: str, num_record: int,
        seed: int | None, group_id: str | None, model_name: str | None,
    ) -> str:
        adapter = self.adapter
        state_key = adapter.state_key if adapter else None
        image_keys = adapter.image_keys if adapter else []

        obs = await client.get_observation()
        # Batch size from the leading dim of any obs array (state if present, else the first).
        probe = obs["data"][state_key] if state_key else next(iter(obs["data"].values()))
        n = int(np.asarray(probe).shape[0])
        rec = VecRecorder(
            name, num_envs=n, record_indices=list(range(min(num_record, n))),
            fps=client.get_control_rate(), seed=seed, group_id=group_id, model=model_name,
        )
        print(f"[vec-agent] job: {rec.job_url}  (N={n})", flush=True)

        chunks: list[deque[np.ndarray]] = [deque() for _ in range(n)]
        for step in range(self.max_steps):
            done = np.asarray(obs["terminated"], dtype=bool).reshape(-1)
            for i in np.nonzero(done)[0]:  # a freshly-reset slot re-infers for its new episode
                chunks[i].clear()
            if step and done.all():
                break

            if any(not c for c in chunks):  # refill spent slots with a fresh batched chunk
                batch = adapter.adapt_observation(obs, prompt) if adapter else obs
                chunk = np.asarray(await asyncio.to_thread(self.model.infer, batch))  # [N, T, A]
                for i, c in enumerate(chunks):
                    if not c:
                        rows = chunk[i]
                        if adapter is not None:  # e.g. DROID delta -> absolute against slot i's query obs
                            rows = adapter.adapt_chunk(rows, {"data": {k: v[i] for k, v in obs["data"].items()}})
                        c.extend(rows)
            action = np.stack([chunks[i].popleft() for i in range(n)])

            state = {state_key: obs["data"][state_key]} if state_key else None
            frames = {k: obs["data"][k] for k in image_keys} if image_keys else None
            rec.record(obs=state, frames=frames, action=action, done=done)
            await client.send_action(action)

            if self.log_every and step % self.log_every == 0:
                print(f"[vec-agent] step {step}/{self.max_steps} live={int((~done).sum())}/{n}", flush=True)
            obs = await client.get_observation()

        rec.close()
        return rec.job_url


__all__ = ["VecRobotAgent"]
