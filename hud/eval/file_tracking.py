"""Rollout-level file-tracking observer.

Wraps the agent loop: if the env published a ``filetracking/1`` capability and
file tracking is on, open it, skip the scenario-setup churn, then sample diffs
on a fixed interval and emit each as a ``hud.filetracking.v1`` span. Decoupled
from the tool loop — spans are self-timestamped and the viewer correlates them
to steps by time.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, cast

from hud.telemetry.filetracking import emit_file_diff, emit_file_snapshot
from hud.utils.time import now_iso

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from hud.capabilities import FileTrackingClient
    from hud.clients.client import HudClient

logger = logging.getLogger("hud.eval.file_tracking")

_DRAIN_TIMEOUT = 10.0


@asynccontextmanager
async def file_tracking_observer(client: HudClient) -> AsyncIterator[None]:
    """Stream workspace diffs to telemetry for the duration of the ``with`` block.

    A no-op unless telemetry is enabled and the manifest has a ``filetracking``
    binding. The binding's presence is the authoritative opt-in: it is published
    iff the workspace was served with ``track_files=True`` (which itself defaults
    to ``HUD_FILE_TRACKING_ENABLED``), so honoring it here means an explicit
    ``track_files=True`` streams even when the global setting is off. The opened
    client is owned by ``client`` and closed on its teardown, so this never
    closes it directly.
    """
    from hud.settings import settings

    if not settings.telemetry_enabled:
        yield
        return
    try:
        client.binding("filetracking")
    except (KeyError, RuntimeError):
        yield
        return

    ft = cast("FileTrackingClient", await client.open("filetracking"))
    # Re-baseline past scenario setup (so the first emitted diff is the agent's,
    # not setup churn) and emit the post-setup manifest as the reconstruction
    # anchor (paths + hashes, no content). Both are preconditions for correct
    # telemetry: a failed re-baseline misattributes scenario-setup edits to the
    # agent, and a missing anchor leaves the streamed diffs with no baseline to
    # reconstruct against. If either fails, skip tracking this rollout rather
    # than stream misleading data.
    try:
        await ft.advance()
        emit_file_snapshot(await ft.snapshot(), started_at=now_iso())
    except Exception as exc:
        logger.warning("file tracking setup failed; not tracking this rollout: %s", exc)
        yield
        return

    stop = asyncio.Event()
    task = asyncio.create_task(_poll(ft, settings.file_tracking_interval, stop))
    try:
        yield
    finally:
        stop.set()
        # Let the current iteration finish cleanly (never cancel mid-request, which
        # would desync the connection); fall back to cancel only if it wedges.
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(asyncio.shield(task), _DRAIN_TIMEOUT)
        if not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        # Trailing diff: edits since the last successful sample. Attempt it in
        # both paths (clean drain or forced cancel); bound it so a connection
        # desynced by the cancel above can't wedge teardown. ``_emit_once`` logs
        # and swallows its own failures.
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(_emit_once(ft), _DRAIN_TIMEOUT)


async def _poll(ft: FileTrackingClient, interval: float, stop: asyncio.Event) -> None:
    while not stop.is_set():
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(stop.wait(), timeout=interval)
        if stop.is_set():
            return
        await _emit_once(ft)


async def _emit_once(ft: FileTrackingClient) -> None:
    started_at = now_iso()
    try:
        result = await ft.diff()
    except Exception as exc:
        logger.debug("file tracking diff failed: %s", exc)
        return
    if result.get("files_changed"):
        emit_file_diff(result, started_at=started_at)


__all__ = ["file_tracking_observer"]
