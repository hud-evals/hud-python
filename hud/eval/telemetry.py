"""HUD platform telemetry for the new eval flow: jobs + per-rollout traces.

Reuses the existing backend contract (``/trace/job/{id}/enter``,
``/trace/{id}/enter`` / ``/exit``) and the trace-context contextvars (so
``@instrument`` spans upload under the right trace). Kept out of ``Taskset`` /
``Run`` so those stay transport-only — the runner just wraps each rollout in
:func:`trace` and registers the batch with :func:`job_enter`.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from hud.eval.context import _current_api_key, set_trace_context
from hud.eval.manager import _send_job_enter
from hud.eval.types import EvalExitPayload, EvalPayload
from hud.settings import settings
from hud.shared import make_request
from hud.telemetry import flush

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from hud.client import Run

logger = logging.getLogger("hud.eval.telemetry")


def _enabled() -> bool:
    return bool(settings.telemetry_enabled and settings.api_key)


async def job_enter(job_id: str, *, name: str, group: int) -> None:
    """Register a batch job with the platform (no-op without telemetry/api key)."""
    if not _enabled():
        return
    try:
        await _send_job_enter(job_id, name, None, group, None)
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
            await _trace_enter(trace_id, job_id, group_id, api_key)
            try:
                yield box
            finally:
                if box:
                    await _trace_exit(trace_id, box[0], job_id, group_id, api_key)
                flush(trace_id)
    finally:
        _current_api_key.reset(key_token)


async def _trace_enter(
    trace_id: str,
    job_id: str | None,
    group_id: str | None,
    api_key: str,
) -> None:
    try:
        await make_request(
            method="POST",
            url=f"{settings.hud_api_url}/trace/{trace_id}/enter",
            json=EvalPayload(job_id=job_id, group_id=group_id).model_dump(exclude_none=True),
            api_key=api_key,
        )
    except Exception as exc:
        logger.warning("trace enter failed: %s", exc)


async def _trace_exit(
    trace_id: str,
    run: Run,
    job_id: str | None,
    group_id: str | None,
    api_key: str,
) -> None:
    trace_data = run.trace
    try:
        payload = EvalExitPayload(
            prompt=run.prompt,
            job_id=job_id,
            group_id=group_id,
            reward=run.reward,
            success=not trace_data.isError,
            error_message=trace_data.content if trace_data.isError else None,
            evaluation_result=run.evaluation or None,
        )
        await make_request(
            method="POST",
            url=f"{settings.hud_api_url}/trace/{trace_id}/exit",
            json=payload.model_dump(exclude_none=True),
            api_key=api_key,
        )
    except Exception as exc:
        logger.warning("trace exit failed: %s", exc)


__all__ = ["job_enter", "trace"]
