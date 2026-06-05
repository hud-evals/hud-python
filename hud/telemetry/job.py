"""HUD platform reporting for the v6 flow: jobs + per-rollout traces.

Self-contained (depends only on ``hud.settings`` / ``hud.shared`` / the trace
contextvars) so the ``Run`` / ``Taskset`` flow reports to HUD without importing
the legacy ``hud.eval`` / ``hud.environment`` stack. The runner wraps each rollout
in :func:`trace` and registers the batch with :func:`job_enter`.

Backend contract (unchanged from v5):
- ``POST /trace/job/{job_id}/enter``  — register the batch job.
- ``POST /trace/{trace_id}/enter``    — a rollout started.
- ``POST /trace/{trace_id}/exit``     — a rollout finished (reward / success).
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from hud.settings import settings
from hud.shared import make_request
from hud.telemetry import flush
from hud.telemetry.context import _current_api_key, set_trace_context

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from hud.client import Run

logger = logging.getLogger("hud.telemetry.job")


def _enabled() -> bool:
    return bool(settings.telemetry_enabled and settings.api_key)


async def job_enter(job_id: str, *, name: str, group: int) -> None:
    """Register a batch job with the platform (no-op without telemetry/api key)."""
    if not _enabled():
        return
    try:
        await make_request(
            method="POST",
            url=f"{settings.hud_api_url}/trace/job/{job_id}/enter",
            json={"name": name, "group": group},
            api_key=settings.api_key,
        )
        logger.info("job: https://hud.ai/jobs/%s", job_id)
    except Exception as exc:
        logger.warning("job enter failed: %s", exc)


@asynccontextmanager
async def trace(
    trace_id: str,
    *,
    job_id: str | None = None,
    group_id: str | None = None,
) -> AsyncIterator[list[Run]]:
    """Report one rollout's trace to HUD around the body.

    Binds ``trace_id`` into the trace context (so ``@instrument`` spans attribute
    to it — always, even with telemetry off, for local training), and when
    telemetry is on posts trace-enter, then on exit posts trace-exit (reward /
    success / error from the recorded :class:`Run`) and flushes. The caller appends
    the resulting ``Run`` to the yielded list.
    """
    box: list[Run] = []
    if not _enabled():
        with set_trace_context(trace_id):
            yield box
        return

    api_key = settings.api_key
    assert api_key is not None  # _enabled() guarantees it
    key_token = _current_api_key.set(api_key)
    try:
        with set_trace_context(trace_id):
            await _post(f"/trace/{trace_id}/enter", {"job_id": job_id, "group_id": group_id}, api_key)
            try:
                yield box
            finally:
                if box:
                    await _post(f"/trace/{trace_id}/exit", _exit_payload(box[0], job_id, group_id), api_key)
                flush(trace_id)
    finally:
        _current_api_key.reset(key_token)


def _exit_payload(run: Run, job_id: str | None, group_id: str | None) -> dict[str, object]:
    trace_data = run.trace
    return {
        "prompt": run.prompt,
        "job_id": job_id,
        "group_id": group_id,
        "reward": run.reward,
        "success": not trace_data.isError,
        "error_message": trace_data.content if trace_data.isError else None,
        "evaluation_result": run.evaluation or None,
    }


async def _post(path: str, payload: dict[str, object], api_key: str) -> None:
    try:
        await make_request(
            method="POST",
            url=f"{settings.hud_api_url}{path}",
            json={k: v for k, v in payload.items() if v is not None},
            api_key=api_key,
        )
    except Exception as exc:
        logger.warning("telemetry %s failed: %s", path, exc)


__all__ = ["job_enter", "trace"]
