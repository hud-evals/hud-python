"""Workspace diffs from the environment's file tracking, streamed as spans."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any, Literal, TypeAlias

from hud.telemetry.context import get_current_trace_id
from hud.telemetry.exporter import queue_span
from hud.telemetry.span import (
    PAYLOAD_ATTRIBUTE,
    SCHEMA_ATTRIBUTE,
    TASK_RUN_ID_ATTRIBUTE,
    Span,
    new_span_id,
    normalize_trace_id,
)
from hud.utils.time import now_iso

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from .run import EnvironmentSession

logger = logging.getLogger("hud.eval.file_tracking")

_FILETRACKING_SCHEMA = "hud.filetracking.v1"
_FileTrackingSpanName: TypeAlias = Literal[
    "filetracking.capture",
    "filetracking.diff",
    "filetracking.setup",
    "filetracking.snapshot",
]


@asynccontextmanager
async def file_tracking_observer(session: EnvironmentSession) -> AsyncIterator[None]:
    """Stream workspace diffs during the block and capture final artifacts on exit.

    The environment opts in by tracking a workspace. Calls are never cancelled mid-flight:
    they share the session's channel with grading.
    """
    from hud.settings import settings

    if not settings.telemetry_enabled or not await _anchor(session):
        yield
        return

    stop = asyncio.Event()

    async def poll_diffs() -> None:
        while not stop.is_set():
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(stop.wait(), timeout=settings.file_tracking_interval)
            if stop.is_set():
                return
            started_at = now_iso()
            try:
                diff = await session.files("diff") or {}
            except Exception as exc:
                logger.debug("file tracking diff failed: %s", exc)
                continue
            if diff.get("files_changed"):
                _emit_file_tracking("filetracking.diff", diff, started_at=started_at)

    poller = asyncio.create_task(poll_diffs())
    try:
        yield
    finally:
        stop.set()
        await asyncio.gather(poller, return_exceptions=True)
        try:
            flush = await session.files("flush") or {}
        except Exception as exc:
            logger.debug("file tracking flush failed: %s", exc)
            flush = {}
        started_at = now_iso()
        diff = flush.get("diff")
        if isinstance(diff, dict) and diff.get("files_changed"):
            _emit_file_tracking("filetracking.diff", diff, started_at=started_at)
        capture = flush.get("capture")
        if isinstance(capture, dict) and (
            capture.get("files_captured")
            or capture.get("files_skipped")
            or capture.get("files_eligible")
            or capture.get("truncated")
        ):
            _emit_file_tracking("filetracking.capture", capture, started_at=started_at)


async def _anchor(session: EnvironmentSession) -> bool:
    """Emit the setup changes and the file manifest; ``False`` when nothing is tracked."""
    try:
        setup = await session.files("setup")
        snapshot = None if setup is None else await session.files("snapshot")
    except Exception as exc:
        logger.warning("file tracking setup failed; not tracking this rollout: %s", exc)
        return False
    if setup is None or snapshot is None:
        return False
    if setup:
        _emit_file_tracking("filetracking.setup", setup, started_at=now_iso())
    _emit_file_tracking("filetracking.snapshot", snapshot, started_at=now_iso())
    return True


def _emit_file_tracking(
    span_name: _FileTrackingSpanName,
    payload: dict[str, Any],
    *,
    started_at: str,
    ended_at: str | None = None,
) -> bool:
    task_run_id = get_current_trace_id()
    if task_run_id is None:
        return False
    span = Span(
        name=span_name,
        trace_id=normalize_trace_id(task_run_id),
        span_id=new_span_id(),
        start_time=started_at,
        end_time=ended_at or now_iso(),
        status_code="OK",
        attributes={
            SCHEMA_ATTRIBUTE: _FILETRACKING_SCHEMA,
            TASK_RUN_ID_ATTRIBUTE: task_run_id,
            PAYLOAD_ATTRIBUTE: payload,
        },
    )
    queue_span(span.model_dump(mode="json"))
    return True
