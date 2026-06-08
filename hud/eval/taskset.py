"""Taskset: a collection of ``Variant``s you evaluate one agent over.

Launches each variant, lets ``agent(run)`` fill ``run.trace``, grades it, and
gathers the :class:`Run`s — with optional GRPO grouping + a concurrency cap. HUD
job/trace reporting lives in :mod:`hud.eval.job`::

    runs = await Taskset(fix_bug(difficulty=d) for d in range(5)).run(agent, group=8)
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import replace
from typing import TYPE_CHECKING, Any

from hud.client import Run

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from hud.agents.base import Agent

    from .variant import Variant

logger = logging.getLogger("hud.eval.taskset")


async def _rollout(
    variant: Variant,
    agent: Agent,
    *,
    job_id: str | None = None,
    group_id: str | None = None,
) -> Run:
    """Drive one variant to a graded :class:`Run` (the rollout atom).

    Launch the env, let ``agent(run)`` fill ``run.trace``, and grade it on exit
    (``run.reward``). The rollout is wrapped in :func:`hud.eval.job.trace`,
    which binds the per-rollout ``trace_id`` into the trace context (so ``@instrument``
    spans upload to it) and reports the trace to HUD. A launch/connect failure is
    isolated into a failed ``Run`` so one bad rollout never collapses a batch.
    """
    from hud.eval.job import trace as report_trace

    trace_id = uuid.uuid4().hex
    async with report_trace(trace_id, job_id=job_id, group_id=group_id) as recorded:
        try:
            async with variant as run:
                await agent(run)
            run.trace.trace_id = trace_id
        except (TimeoutError, asyncio.CancelledError, KeyboardInterrupt):
            raise
        except Exception as exc:
            logger.warning("rollout failed: %s", exc)
            run = Run.failed(str(exc), trace_id=trace_id)
        run.job_id = job_id
        run.group_id = group_id
        recorded.append(run)
    return run


def _job_name(variants: list[Variant], group: int) -> str:
    suffix = f" ({group} times)" if group > 1 else ""
    if len(variants) == 1:
        return f"Task Run: {variants[0].task}{suffix}"
    return f"Batch Run: {len(variants)} tasks{suffix}"


class Taskset:
    """A collection of :class:`~hud.eval.variant.Variant`s to evaluate an agent over."""

    def __init__(self, variants: Iterable[Variant]) -> None:
        self.variants: list[Variant] = list(variants)

    def __len__(self) -> int:
        return len(self.variants)

    def __iter__(self) -> Iterator[Variant]:
        return iter(self.variants)

    async def run(
        self,
        agent: Any,
        *,
        group: int = 1,
        max_concurrent: int | None = None,
    ) -> list[Run]:
        """Gather rollouts over every variant x ``group`` with an optional concurrency cap.

        One shared (stateless) ``agent`` drives every rollout; each rollout gets a
        fresh env (via the variant) and its own :class:`Run`. Registers one HUD job
        for the batch and reports each rollout's trace under it. Returns the runs in
        expansion order (variant-major, then group).
        """
        if group < 1:
            raise ValueError("group must be >= 1")
        from hud.eval.job import job_enter

        # Fresh Variant per rollout (the Variant CM holds per-enter state); the
        # ``group`` repeats of one variant share a group_id (the GRPO group).
        expanded: list[tuple[Variant, str]] = []
        for variant in self.variants:
            group_id = uuid.uuid4().hex
            expanded.extend((replace(variant), group_id) for _ in range(group))

        job_id = uuid.uuid4().hex
        await job_enter(job_id, name=_job_name(self.variants, group), group=group)

        sem = asyncio.Semaphore(max_concurrent) if max_concurrent else None

        async def _one(variant: Variant, group_id: str) -> Run:
            if sem is None:
                return await _rollout(variant, agent, job_id=job_id, group_id=group_id)
            async with sem:
                return await _rollout(variant, agent, job_id=job_id, group_id=group_id)

        logger.info(
            "running %d rollouts (%d variants x %d group)%s",
            len(expanded),
            len(self.variants),
            group,
            f", max_concurrent={max_concurrent}" if max_concurrent else "",
        )
        return list(await asyncio.gather(*(_one(v, gid) for v, gid in expanded)))


__all__ = ["Taskset"]
