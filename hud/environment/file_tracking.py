"""Serving layer for :class:`~hud.environment.file_tracker.FileTracker`.

Exposes one tracker over the ``filetracking/1`` wire: a framed-JSON request
loop handling ``diff`` / ``snapshot`` / ``advance``. Scans run in a thread
executor (CPU-bound directory walks must not block the event loop) and are
serialized by a lock, since the tracker's baseline is mutable state.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

from .utils import error, read_frame, reply, send_frame

if TYPE_CHECKING:
    from .file_tracker import FileTracker

LOGGER = logging.getLogger("hud.environment.file_tracking")


class _FileTrackingHandler:
    """Per-server dispatcher; one tracker shared across connections under a lock."""

    def __init__(self, tracker: FileTracker) -> None:
        self._tracker = tracker
        self._lock = asyncio.Lock()

    async def handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            while (msg := await read_frame(reader)) is not None:
                msg_id = msg.get("id")
                method = msg.get("method", "")
                try:
                    result = await self._dispatch(method)
                except Exception as exc:  # never tear the connection down on one bad call
                    LOGGER.debug("filetracking %s failed: %s", method, exc)
                    if isinstance(msg_id, int):
                        await send_frame(writer, error(msg_id, -32000, str(exc)))
                    continue
                if isinstance(msg_id, int):
                    await send_frame(writer, reply(msg_id, result))
        except (ConnectionError, OSError):
            pass
        finally:
            writer.close()

    async def _dispatch(self, method: str) -> dict[str, Any]:
        loop = asyncio.get_running_loop()
        async with self._lock:
            if method == "diff":
                diff = await loop.run_in_executor(None, self._tracker.take_snapshot)
                return diff.to_dict()
            if method == "snapshot":
                manifest = self._tracker.current_manifest()
                return {"files": manifest, "files_scanned": len(manifest)}
            if method == "advance":
                await loop.run_in_executor(None, self._tracker.advance_baseline)
                return {"advanced": True}
            raise ValueError(f"unknown filetracking method: {method!r}")


async def serve_file_tracking(
    tracker: FileTracker, host: str = "127.0.0.1", port: int = 0
) -> asyncio.Server:
    """Bind a ``filetracking/1`` server for ``tracker``. Caller drives the port."""
    handler = _FileTrackingHandler(tracker)
    server = await asyncio.start_server(handler.handle, host, port)
    sock = server.sockets[0].getsockname()
    LOGGER.info("filetracking bound on %s:%s", sock[0], sock[1])
    return server


__all__ = ["serve_file_tracking"]
