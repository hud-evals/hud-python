"""Jobs and their platform reporting, a no-op without telemetry and an API key."""

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


def _reporting_enabled() -> bool:
    from hud.settings import settings

    return bool(settings.telemetry_enabled and settings.api_key)


async def job_enter(job_id: str, *, name: str, group: int, taskset_id: str | None = None) -> None:
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
    if not _reporting_enabled() or trace.trace_id is None:
        return
    await _report(
        f"/trace/{trace.trace_id}/exit",
        {
            "status": trace.status or "completed",
            "reward": trace.reward,
            # A step error the agent recovered from isn't a trace error.
            "error": trace.error if trace.is_error else None,
            "evaluation_result": trace.evaluation or None,
            "stop_reason": trace.stop_reason,
            "metadata": trace.extra or None,
        },
    )


async def _report(path: str, payload: dict[str, Any]) -> None:
    body = {k: v for k, v in payload.items() if v is not None}
    # One retry: bursts of concurrent rollouts draw transient rejections, even 401s.
    for attempt in (1, 2):
        try:
            await PlatformClient.from_settings().apost(path, json=body)
            return
        except Exception as exc:
            if attempt == 2:
                logger.warning("platform report %s failed: %s", path, exc)
            else:
                await asyncio.sleep(0.5)
