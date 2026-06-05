"""HUD training client: turn rewarded rollouts into training signals.

Agent-agnostic: take rewarded rollouts (``Run``s), compute **GRPO advantages** over
the group, and POST ``{trace_id, advantage}`` to the backend (which holds the
token-level trajectories keyed by ``trace_id`` and runs the optimizer)::

    trainer = HudTrainingClient(TrainingConfig(learning_rate=1e-5))
    runs = await Taskset(task(x) for x in xs).run(agent, group=16)
    await trainer.reward(runs)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Protocol, runtime_checkable

import httpx

from hud.settings import settings


@runtime_checkable
class Rewarded(Protocol):
    """The minimal surface ``reward`` needs — "this rollout got this reward".

    Anything carrying a ``trace_id`` and a ``reward`` satisfies it (a ``Run`` does,
    but so does a lightweight stand-in).
    """

    trace_id: str | None
    reward: float


@dataclass(slots=True)
class TrainingConfig:
    """Managed-tier training params. GRPO is the only method for now.

    The backend computes group-relative advantages over each submitted group and
    runs ``forward_backward`` + ``optim_step`` internally; ``batch_groups``
    accumulates that many groups before one step.
    """

    learning_rate: float = 1e-5
    kl_coef: float = 0.0
    max_grad_norm: float | None = 1.0
    batch_groups: int = 1  # accumulate N groups → one optim_step
    normalize_advantage: bool = True  # divide group advantages by std (GRPO)


def group_relative(
    rewards: list[float],
    *,
    normalize_std: bool = True,
    eps: float = 1e-6,
) -> list[float]:
    """GRPO advantages over one group: ``reward - mean``, optionally ``/ std``."""
    if not rewards:
        return []
    mean = sum(rewards) / len(rewards)
    advs = [r - mean for r in rewards]
    if normalize_std:
        std = (sum(a * a for a in advs) / len(advs)) ** 0.5
        if std > eps:
            advs = [a / std for a in advs]
    return advs


@dataclass
class HudTrainingClient:
    """Send rewarded rollouts to the HUD training backend. Agent-agnostic."""

    config: TrainingConfig = field(default_factory=TrainingConfig)
    base_url: str | None = None
    api_key: str | None = None

    async def reward(self, group: list[Rewarded]) -> None:
        """Reward a group of rollouts (the model updates in the background).

        Computes GRPO advantages over the group and POSTs ``{trace_id, advantage}``
        to ``{base_url}/train/advantages``. Each item just needs ``trace_id`` +
        ``reward`` (a ``Run`` qualifies); only those signals cross the wire, never
        token data. Returns once enqueued — it does not wait for an optim step.
        """
        advantages = group_relative(
            [r.reward for r in group],
            normalize_std=self.config.normalize_advantage,
        )
        signals = [
            {"trace_id": r.trace_id, "advantage": adv}
            for r, adv in zip(group, advantages, strict=True)
            if r.trace_id is not None
        ]
        if not signals:
            return

        base_url = self.base_url or settings.hud_api_url
        api_key = self.api_key or settings.api_key
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        async with httpx.AsyncClient(base_url=base_url, timeout=30.0) as client:
            resp = await client.post(
                "/train/advantages",
                json={"config": asdict(self.config), "signals": signals},
                headers=headers,
            )
            resp.raise_for_status()


__all__ = ["HudTrainingClient", "Rewarded", "TrainingConfig", "group_relative"]
