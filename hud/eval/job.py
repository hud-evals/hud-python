"""Job: the platform receipt for one execution — there are no standalone traces.

The live execution atom remains :class:`hud.eval.Run`; a ``Job`` collects the
graded runs of one batch under one platform job id. Every trace reports under
a job: the scheduler's batch job, or the single-run job :func:`rollout`
registers when called bare.

Backend reporting contract:
- ``POST /trace/job/{job_id}/enter`` — register the batch job.
- ``POST /trace/{trace_id}/enter``   — a rollout started.
- ``POST /trace/{trace_id}/exit``    — a rollout finished (reward / success).

All three are best-effort no-ops without telemetry + an API key, so local runs
never depend on the platform.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from hud.utils.platform import PlatformClient

if TYPE_CHECKING:
    from .rollout import Run

logger = logging.getLogger("hud.eval.job")


@dataclass(slots=True)
class Job:
    """Platform receipt for one execution: the graded runs under one job id."""

    id: str
    name: str
    runs: list[Run] = field(default_factory=list)
    group: int = 1

    @classmethod
    async def start(cls, name: str, *, group: int = 1) -> Job:
        """Open a job spanning multiple scheduler calls.

        A scheduler call mints its own job by default; pass a started job as
        ``job=`` to ``Task.run`` / ``Taskset.run`` to accumulate every run of a
        longer arc — a training session, a chat conversation — under one id.
        """
        job = cls(id=uuid.uuid4().hex, name=name, group=group)
        await job_enter(job.id, name=name, group=group)
        return job

    @property
    def reward(self) -> float:
        """Mean reward across runs (0.0 for an empty job)."""
        if not self.runs:
            return 0.0
        return sum(run.reward for run in self.runs) / len(self.runs)


def _reporting_enabled() -> bool:
    from hud.settings import settings

    return bool(settings.telemetry_enabled and settings.api_key)


async def job_enter(job_id: str, *, name: str, group: int) -> None:
    """Register a batch job with the platform."""
    if not _reporting_enabled():
        return
    await _report(f"/trace/job/{job_id}/enter", {"name": name, "group": group})
    logger.info("job: https://hud.ai/jobs/%s", job_id)


async def trace_enter(trace_id: str, *, job_id: str | None, group_id: str | None) -> None:
    """Report that one rollout started."""
    if not _reporting_enabled():
        return
    await _report(f"/trace/{trace_id}/enter", {"job_id": job_id, "group_id": group_id})


async def trace_exit(run: Run) -> None:
    """Report one finished rollout (reward / success / error) from its ``Run``."""
    if not _reporting_enabled() or run.trace.trace_id is None:
        return
    await _report(
        f"/trace/{run.trace.trace_id}/exit",
        {
            "prompt": run.prompt,
            "job_id": run.job_id,
            "group_id": run.group_id,
            "reward": run.reward,
            "success": not run.trace.isError,
            "error_message": run.trace.content if run.trace.isError else None,
            "evaluation_result": run.evaluation or None,
        },
    )


async def _report(path: str, payload: dict[str, Any]) -> None:
    try:
        await PlatformClient.from_settings().apost(
            path,
            json={k: v for k, v in payload.items() if v is not None},
        )
    except Exception as exc:
        logger.warning("platform report %s failed: %s", path, exc)


__all__ = ["Job"]
