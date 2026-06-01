"""Taskset: a collection of Variants you run an agent over.

A :class:`~hud.client.Variant` is one parameterized task bound to an env/sandbox.
A ``Taskset`` groups many of them so a single (stateless) agent can be evaluated
across the set — optionally with GRPO-style grouping and a concurrency cap::

    ts = Taskset(fix_bug(difficulty=d) for d in range(1, 6))
    runs = await ts.run(agent, group=8, max_concurrent=16)
    await trainer.reward(runs)        # each Run carries reward + trace_id

The contract is just ``agent(run)`` filling ``run.trace``; the taskset owns
launching each variant, grading it, and gathering the resulting :class:`Run`s
(the episode: prompt + trace + reward).
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
    from hud.client import Variant

logger = logging.getLogger("hud.taskset")


async def _rollout(variant: Variant, agent: Agent) -> Run:
    """Drive one variant to a graded :class:`Run` (the rollout atom).

    Launch the env, let ``agent(run)`` fill ``run.trace``, and grade it on exit
    (``run.reward``). A per-rollout ``trace_id`` is bound into the trace context so
    ``@instrument`` spans and Mode-B training key correctly, then stamped on the
    trace. A failure while launching/connecting is isolated into a failed ``Run``
    so one bad rollout never collapses a batch.
    """
    from hud.eval.context import set_trace_context  # lazy: avoid legacy import at module load

    trace_id = uuid.uuid4().hex
    try:
        with set_trace_context(trace_id):
            async with variant as run:
                await agent(run)
    except (TimeoutError, asyncio.CancelledError, KeyboardInterrupt):
        raise
    except Exception as exc:
        logger.warning("rollout failed: %s", exc)
        return Run.failed(str(exc), trace_id=trace_id)
    run.trace.trace_id = trace_id
    return run


class Taskset:
    """A collection of :class:`~hud.client.Variant`s to evaluate an agent over."""

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
        fresh env (via the variant) and its own :class:`Run`. Returns the runs in
        expansion order (variant-major, then group).
        """
        if group < 1:
            raise ValueError("group must be >= 1")
        # Fresh Variant per rollout: the Variant context manager holds per-enter
        # state, so concurrent rollouts must not share one instance.
        expanded = [replace(v) for v in self.variants for _ in range(group)]
        sem = asyncio.Semaphore(max_concurrent) if max_concurrent else None

        async def _one(v: Variant) -> Run:
            if sem is None:
                return await _rollout(v, agent)
            async with sem:
                return await _rollout(v, agent)

        logger.info(
            "running %d rollouts (%d variants x %d group)%s",
            len(expanded), len(self.variants), group,
            f", max_concurrent={max_concurrent}" if max_concurrent else "",
        )
        return list(await asyncio.gather(*(_one(v) for v in expanded)))


__all__ = ["Taskset"]
