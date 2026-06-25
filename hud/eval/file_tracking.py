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
import json
import logging
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any, Self
from urllib.parse import urlsplit

from hud.telemetry.filetracking import emit_file_diff, emit_file_snapshot
from hud.utils.time import now_iso

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from hud.capabilities import Capability
    from hud.clients.client import HudClient

logger = logging.getLogger("hud.eval.file_tracking")

_DRAIN_TIMEOUT = 10.0


class FileTrackingClient:
    """Live ``filetracking/1`` connection: ``diff`` / ``snapshot`` / ``advance``."""

    def __init__(
        self,
        capability: Capability,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        self.capability = capability
        self._reader = reader
        self._writer = writer
        self._id = 0
        self._lock = asyncio.Lock()

    @classmethod
    async def connect(cls, cap: Capability) -> Self:
        parts = urlsplit(cap.url)
        if parts.hostname is None or parts.port is None:
            raise ValueError(f"filetracking capability missing host or port: {cap.url!r}")
        reader, writer = await asyncio.open_connection(parts.hostname, parts.port)
        return cls(cap, reader, writer)

    async def diff(self) -> dict[str, Any]:
        return await self._call("diff")

    async def snapshot(self) -> dict[str, Any]:
        return await self._call("snapshot")

    async def advance(self) -> None:
        await self._call("advance")

    async def close(self) -> None:
        self._writer.close()
        with contextlib.suppress(OSError):
            await self._writer.wait_closed()

    async def _call(self, method: str) -> dict[str, Any]:
        async with self._lock:
            self._id += 1
            msg_id = self._id
            payload = json.dumps(
                {"jsonrpc": "2.0", "id": msg_id, "method": method, "params": {}},
                separators=(",", ":"),
            )
            self._writer.write(payload.encode("utf-8") + b"\n")
            await self._writer.drain()
            line = await self._reader.readline()
            if not line:
                raise ConnectionError(f"filetracking: connection closed during {method!r}")
            reply: dict[str, Any] = json.loads(line)
            if "error" in reply:
                err = reply["error"]
                raise RuntimeError(f"filetracking {method!r} error: {err.get('message')}")
            result = reply.get("result")
            if not isinstance(result, dict):
                raise RuntimeError(f"filetracking {method!r}: result was not an object")
            return result


@asynccontextmanager
async def file_tracking_observer(client: HudClient) -> AsyncIterator[None]:
    """Stream workspace diffs to telemetry for the duration of the ``with`` block.

    A no-op unless telemetry is enabled and the manifest has a ``filetracking``
    binding. The binding's presence is the authoritative opt-in: it is published
    iff the workspace was served with ``track_files=True`` (which itself defaults
    to ``HUD_FILE_TRACKING_ENABLED``), so honoring it here means an explicit
    ``track_files=True`` streams even when the global setting is off.
    """
    from hud.settings import settings

    if not settings.telemetry_enabled:
        yield
        return
    try:
        cap = client.binding("filetracking")
    except (KeyError, RuntimeError):
        yield
        return

    # Open the capability, re-baseline past scenario setup (so the first emitted
    # diff is the agent's, not setup churn), and emit the post-setup manifest as
    # the reconstruction anchor (paths + hashes, no content). Tracking is
    # observation-only, so any setup failure — a refused tunnel, a failed
    # re-baseline (which would misattribute setup edits to the agent), or a
    # missing anchor — skips tracking rather than breaking the agent loop.
    ft: FileTrackingClient | None = None
    try:
        ft = await FileTrackingClient.connect(cap)
        await ft.advance()
        emit_file_snapshot(await ft.snapshot(), started_at=now_iso())
    except Exception as exc:
        if ft is not None:
            with contextlib.suppress(Exception):
                await ft.close()
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
        with contextlib.suppress(Exception):
            await ft.close()


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
