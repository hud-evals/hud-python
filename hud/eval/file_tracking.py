"""Workspace diffs from the environment's file tracking, streamed as spans."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from hud.settings import settings
from hud.telemetry.exporter import export
from hud.utils.time import now_iso

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from .run import EnvironmentSession

logger = logging.getLogger("hud.eval.file_tracking")

_FILETRACKING_SCHEMA = "hud.filetracking.v1"


@asynccontextmanager
async def file_tracking_observer(
    session: EnvironmentSession, trace_id: str | None
) -> AsyncGenerator[None]:
    """Stream workspace diffs and final artifacts as spans of ``trace_id`` during the block.

    A no-op without a trace id, when the trace is recorded nowhere (no local file and no
    upload), or when the environment tracks no workspace. A failed tracking setup logs a
    warning and the block runs untracked. Calls are never cancelled mid-flight: they share
    the session's channel with grading.
    """
    if trace_id is None or (settings.trace_dir is None and not settings.reporting_enabled):
        yield
        return
    if not await _anchor(session, trace_id):
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
                export(
                    trace_id,
                    name="filetracking.diff",
                    schema=_FILETRACKING_SCHEMA,
                    payload=diff,
                    started_at=started_at,
                )

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
            export(
                trace_id,
                name="filetracking.diff",
                schema=_FILETRACKING_SCHEMA,
                payload=diff,
                started_at=started_at,
            )
        capture = flush.get("capture")
        if isinstance(capture, dict) and (
            capture.get("files_captured")
            or capture.get("files_skipped")
            or capture.get("files_eligible")
            or capture.get("truncated")
        ):
            export(
                trace_id,
                name="filetracking.capture",
                schema=_FILETRACKING_SCHEMA,
                payload=capture,
                started_at=started_at,
            )


async def _anchor(session: EnvironmentSession, trace_id: str) -> bool:
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
        export(
            trace_id,
            name="filetracking.setup",
            schema=_FILETRACKING_SCHEMA,
            payload=setup,
            started_at=now_iso(),
        )
    export(
        trace_id,
        name="filetracking.snapshot",
        schema=_FILETRACKING_SCHEMA,
        payload=snapshot,
        started_at=now_iso(),
    )
    return True
