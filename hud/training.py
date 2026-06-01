"""HUD training client: turn rewarded rollouts into training signals.

Decoupled from the agent. The agent's inference runs through a backend that
collects token-level logprobs server-side (keyed by ``trace_id``); this client
takes the resulting rewarded ``Trace``s, computes **GRPO advantages** over the
group (group-relative; the SDK owns the estimator), and sends
``{trace_id, advantage}`` to the backend. The backend then attaches each
self-contained advantage to its stored trajectory and runs
``forward_backward`` + ``optim_step`` in the background — no grouping needed
server-side.

(Contrast with Tinker, which *is* tied to the agent: there the agent samples from
the very policy you train. Here the agent only produces ``Trace``s; training
consumes them.)

    trainer = HudTrainingClient(TrainingConfig(learning_rate=1e-5))
    traces = await asyncio.gather(*(rollout(v) for v in expand(tasks, group=16)))
    await trainer.reward(traces)          # this trace got this reward; group → backend (async)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Protocol, runtime_checkable

import httpx

from hud.settings import settings


@runtime_checkable
class Rewarded(Protocol):
    """The minimal surface ``reward`` needs — "this trace got this reward".

    Smaller than a full ``Trace``: anything carrying a ``trace_id`` and a
    ``reward`` satisfies it (a ``Trace`` does, but so does a lightweight stand-in).
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
        """Reward a group of rollouts; the model updates in the background.

        Each item just needs a ``trace_id`` and a ``reward`` (the ``Rewarded``
        protocol — a ``Trace`` qualifies). Computes GRPO advantages over the group
        (group-relative; the SDK owns the estimator) and posts
        ``{trace_id, advantage}`` to the backend, which attaches each
        self-contained advantage to its stored trajectory and runs
        ``forward_backward`` / ``optim_step`` per ``config`` — asynchronously.
        Returns once the signals are enqueued; it does not wait for a step.

        The group is structural: the rollouts you gathered for one task. Only
        ``{trace_id, advantage}`` crosses the wire — never token data, and the
        backend needs no grouping of its own.

        Backend contract: ``POST {base_url}/train/advantages`` with
        ``{"config": {...}, "signals": [{"trace_id", "advantage"}, ...]}``.
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
