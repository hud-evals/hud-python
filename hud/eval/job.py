"""Jobs, each trace's root span, and their platform reporting."""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

from hud.graders.base import Grade
from hud.settings import settings
from hud.telemetry.exporter import export
from hud.telemetry.span import TRACE_SCHEMA
from hud.utils.platform import PlatformClient

if TYPE_CHECKING:
    from hud.types import Trace
    from hud.utils.serialization import JsonObject

logger = logging.getLogger("hud.eval.job")


@dataclass(slots=True)
class Job:
    """The graded traces reported under one platform job id."""

    id: str
    name: str
    traces: list[Trace] = field(default_factory=list)
    group: int = 1
    taskset_id: str | None = None

    @classmethod
    async def start(cls, name: str, *, group: int = 1, taskset_id: str | None = None) -> Job:
        """Open a job that several ``run`` calls share through ``job=``."""
        job = cls(id=uuid.uuid4().hex, name=name, group=group, taskset_id=taskset_id)
        await job_enter(job.id, name=name, group=group, taskset_id=taskset_id)
        return job

    @property
    def reward(self) -> float:
        """Mean reward over traces whose grade is not an error (0.0 when there are none)."""
        graded = [trace.reward for trace in self.traces if not trace.grade.is_error]
        if not graded:
            return 0.0
        return sum(graded) / len(graded)

    @property
    def errors(self) -> list[Trace]:
        """Traces whose grade is an error, which ``reward`` excludes."""
        return [trace for trace in self.traces if trace.grade.is_error]

    @property
    def results(self) -> dict[str, list[Trace]]:
        """Traces by task slug, a list per slug since ``group`` repeats each task."""
        out: dict[str, list[Trace]] = {}
        for trace in self.traces:
            out.setdefault(trace.slug or "", []).append(trace)
        return out


async def job_enter(job_id: str, *, name: str, group: int, taskset_id: str | None = None) -> None:
    if not settings.reporting_enabled:
        return
    await _report(
        f"/trace/job/{job_id}/enter",
        {"name": name, "group": group, "taskset_id": taskset_id},
    )
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
    if not settings.reporting_enabled:
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


async def trace_exit(
    trace: Trace, *, model: str | None, parent_trace_id: str | None, started_at: str
) -> None:
    """Export the trace's root span, who ran it and how it ended, and report the end."""
    assert trace.trace_id is not None
    # A step error the agent recovered from isn't a trace error.
    error = trace.error if trace.is_error else None
    outcome = {
        "status": trace.status or "completed",
        "reward": trace.reward,
        "error": error,
        "evaluation_result": (None if trace.grade == Grade.ungraded() else trace.evaluation),
        "stop_reason": trace.stop_reason,
        "metadata": trace.extra or None,
    }
    identity = {
        "job_id": trace.job_id,
        "group_id": trace.group_id,
        "task_slug": trace.slug,
        "model": model,
        "parent_trace_id": parent_trace_id,
    }
    root = {k: v for k, v in {**identity, **outcome}.items() if v is not None}
    export(
        trace.trace_id,
        name="trace",
        schema=TRACE_SCHEMA,
        payload=cast("JsonObject", root),
        started_at=started_at,
        error=error,
    )
    if settings.reporting_enabled:
        await _report(f"/trace/{trace.trace_id}/exit", outcome)


async def _report(path: str, payload: dict[str, Any]) -> None:
    body = {k: v for k, v in payload.items() if v is not None}
    # Retried once on any rejection, a plain 500 included; a report still lost is only
    # logged, since telemetry never fails the run.
    for attempt in (1, 2):
        try:
            await PlatformClient.from_settings().apost(path, json=body)
            return
        except Exception as exc:
            if attempt == 2:
                logger.warning("platform report %s failed: %s", path, exc)
            else:
                await asyncio.sleep(0.5)
