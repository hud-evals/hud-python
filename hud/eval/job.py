"""Job: the platform receipt for one execution — there are no standalone traces.

A ``Job`` collects the graded traces (:class:`~hud.types.Trace`) of one batch
under one platform job id. Every trace reports under a job: the scheduler's batch
job, or the single-run job :func:`rollout` registers when called bare.

Backend reporting contract:
- ``POST /trace/job/{job_id}/enter`` — register the batch job.
- ``POST /trace/{trace_id}/enter``   — a rollout started.
- ``POST /trace/{trace_id}/exit``    — a rollout finished (reward / success).

All three are best-effort no-ops without telemetry + an API key, so local runs
never depend on the platform.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from hud.utils.platform import PlatformClient

if TYPE_CHECKING:
    from hud.types import Trace

logger = logging.getLogger("hud.eval.job")


@dataclass(slots=True)
class Job:
    """Platform receipt for one execution: the graded traces under one job id."""

    id: str
    name: str
    traces: list[Trace] = field(default_factory=list)
    group: int = 1
    #: Platform taskset id this job runs, when it's a synced taskset
    #: (``Taskset.from_api``). Links the job to that taskset on the platform.
    taskset_id: str | None = None

    @classmethod
    async def start(cls, name: str, *, group: int = 1, taskset_id: str | None = None) -> Job:
        """Open a job spanning multiple scheduler calls.

        A scheduler call mints its own job by default; pass a started job as
        ``job=`` to ``Task.run`` / ``Taskset.run`` to accumulate every trace of a
        longer arc — a training session, a chat conversation — under one id.
        """
        job = cls(id=uuid.uuid4().hex, name=name, group=group, taskset_id=taskset_id)
        await job_enter(job.id, name=name, group=group, taskset_id=taskset_id)
        return job

    @property
    def reward(self) -> float:
        """Mean reward across traces whose grade is not an error (0.0 when there are none)."""
        graded = [trace.reward for trace in self.traces if not trace.grade.is_error]
        if not graded:
            return 0.0
        return sum(graded) / len(graded)

    @property
    def errors(self) -> list[Trace]:
        """Traces whose grade is an error, excluded from :attr:`reward`."""
        return [trace for trace in self.traces if trace.grade.is_error]

    @property
    def results(self) -> dict[str, list[Trace]]:
        """Traces grouped by task slug — the safe alternative to positional zip.

        List-valued because ``group > 1`` produces several traces per task, and
        in slug order within each group. Use this instead of ``zip(tasks,
        job.traces)``, which silently misaligns once grouping or task ordering
        changes.
        """
        out: dict[str, list[Trace]] = {}
        for trace in self.traces:
            out.setdefault(trace.slug or "", []).append(trace)
        return out


def _reporting_enabled() -> bool:
    from hud.settings import settings

    return bool(settings.telemetry_enabled and settings.api_key)


async def job_enter(job_id: str, *, name: str, group: int, taskset_id: str | None = None) -> None:
    """Register a batch job with the platform.

    ``taskset_id`` links the job to a synced taskset (set when running
    ``Taskset.from_api``); ``None`` for ad-hoc/local tasksets. The platform
    creates no taskset on its own — remote rollouts carry the scenario inline.
    """
    if not _reporting_enabled():
        return
    await _report(
        f"/trace/job/{job_id}/enter",
        {"name": name, "group": group, "taskset_id": taskset_id},
    )
    from hud.settings import settings

    logger.info("job: %s/jobs/%s", settings.hud_web_url, str(uuid.UUID(job_id)))


async def trace_enter(
    trace_id: str,
    *,
    job_id: str | None,
    group_id: str | None,
    task_slug: str,
    model: str | None = None,
    parent_trace_id: str | None = None,
) -> None:
    """Report that one rollout started.

    ``task_slug`` identifies the logical task independently of the execution
    environment. ``model`` is the model string the agent will sample (when
    known); the platform resolves it and attributes the trace immediately on
    enter.
    """
    if not _reporting_enabled():
        return
    await _report(
        f"/trace/{trace_id}/enter",
        {
            "job_id": job_id,
            "group_id": group_id,
            "task_slug": task_slug,
            "model": model,
            "parent_trace_id": parent_trace_id,
        },
    )


async def trace_exit(trace: Trace) -> None:
    """Report one finished rollout (status / reward / error / metadata) from its trace."""
    if not _reporting_enabled() or trace.trace_id is None:
        return
    await _report(
        f"/trace/{trace.trace_id}/exit",
        {
            "status": trace.status or "completed",
            "reward": trace.reward,
            # Recovered step errors stay on the steps; only an errored trace
            # reports a trace-level error.
            "error": trace.error if trace.is_error else None,
            "evaluation_result": trace.evaluation or None,
            "stop_reason": trace.stop_reason,
            # Trajectory metadata the platform stores on the trace for display;
            # never load-bearing.
            "metadata": trace.extra or None,
        },
    )


async def _report(path: str, payload: dict[str, Any]) -> None:
    body = {k: v for k, v in payload.items() if v is not None}
    # One bounded retry: reporting is fire-and-forget, and concurrent rollout
    # bursts have been observed to draw transient rejections (including
    # spurious 401s) from the platform that succeed moments later.
    for attempt in (1, 2):
        try:
            await PlatformClient.from_settings().apost(path, json=body)
            return
        except Exception as exc:
            if attempt == 2:
                logger.warning("platform report %s failed: %s", path, exc)
            else:
                await asyncio.sleep(0.5)
