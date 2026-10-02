"""Workspace diffs from an env's filetracking capability, streamed as spans."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any, Literal, Self, TypeAlias
from urllib.parse import urlsplit

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

    from hud.capabilities import Capability
    from hud.clients.client import HudClient

logger = logging.getLogger("hud.eval.file_tracking")

_DRAIN_TIMEOUT = 10.0
# flush can carry a 50 MiB diff plus base64 capture and JSON escaping overhead.
_FRAME_LIMIT_BYTES = 160 * 1024 * 1024
_FILETRACKING_SCHEMA = "hud.filetracking.v1"
_FileTrackingSpanName: TypeAlias = Literal[
    "filetracking.capture",
    "filetracking.diff",
    "filetracking.setup",
    "filetracking.snapshot",
]


class FileTrackingClient:
    """Live ``filetracking/1`` connection."""

    def __init__(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        self._reader = reader
        self._writer = writer
        self._id = 0
        self._lock = asyncio.Lock()

    @classmethod
    async def connect(cls, cap: Capability) -> Self:
        parts = urlsplit(cap.url)
        if parts.hostname is None or parts.port is None:
            raise ValueError(f"filetracking capability missing host or port: {cap.url!r}")
        reader, writer = await asyncio.open_connection(
            parts.hostname,
            parts.port,
            limit=_FRAME_LIMIT_BYTES,
        )
        return cls(reader, writer)

    async def close(self) -> None:
        self._writer.close()
        with contextlib.suppress(OSError):
            await self._writer.wait_closed()

    async def call(self, method: str) -> dict[str, Any]:
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
    """Stream workspace diffs and final artifacts during the block.

    The env's ``filetracking`` binding is the opt-in, so ``track_files=True``
    streams even when ``HUD_FILE_TRACKING_ENABLED`` is off.
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

    # Older servers can only advance past setup; both set the agent-edit baseline.
    ft: FileTrackingClient | None = None
    try:
        ft = await FileTrackingClient.connect(cap)
        if cap.params.get("setup_diff") is True:
            setup = await ft.call("setup")
            _emit_file_tracking(
                "filetracking.setup",
                setup,
                started_at=now_iso(),
            )
        else:
            await ft.call("advance")
        _emit_file_tracking(
            "filetracking.snapshot",
            await ft.call("snapshot"),
            started_at=now_iso(),
        )
    except Exception as exc:
        if ft is not None:
            with contextlib.suppress(Exception):
                await ft.close()
        logger.warning("file tracking setup failed; not tracking this rollout: %s", exc)
        yield
        return

    stop = asyncio.Event()

    async def poll_diffs() -> None:
        while not stop.is_set():
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(stop.wait(), timeout=settings.file_tracking_interval)
            if stop.is_set():
                return
            started_at = now_iso()
            try:
                diff = await ft.call("diff")
            except Exception as exc:
                logger.debug("file tracking diff failed: %s", exc)
                continue
            if diff.get("files_changed"):
                _emit_file_tracking("filetracking.diff", diff, started_at=started_at)

    task = asyncio.create_task(poll_diffs())
    try:
        yield
    finally:
        stop.set()
        # Cancelling mid-request would desync the connection: let the poll finish first.
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(asyncio.shield(task), _DRAIN_TIMEOUT)
        if not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        # Bounded: a cancel above may have desynced the connection.
        try:
            flush = await asyncio.wait_for(ft.call("flush"), _DRAIN_TIMEOUT)
        except TimeoutError:
            pass
        except Exception as exc:
            logger.debug("file tracking flush failed: %s", exc)
        else:
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
        with contextlib.suppress(Exception):
            await ft.close()


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
