"""Workspace diffs from an env's filetracking capability, streamed as spans."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any, Self
from urllib.parse import urlsplit

from hud.environment.utils import FILE_TRACKING_FRAME_LIMIT_BYTES
from hud.settings import settings
from hud.telemetry.exporter import export
from hud.utils.time import now_iso

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from hud.capabilities.base import Capability
    from hud.clients.client import HudClient

logger = logging.getLogger("hud.eval.file_tracking")

_DRAIN_TIMEOUT = 10.0
_FILETRACKING_SCHEMA = "hud.filetracking.v1"


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
            limit=FILE_TRACKING_FRAME_LIMIT_BYTES,
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
async def file_tracking_observer(client: HudClient, trace_id: str) -> AsyncGenerator[None]:
    """Stream workspace diffs and final artifacts as spans of ``trace_id`` during the block.

    A no-op unless the trace is recorded somewhere (a local file or an upload) and the env
    binds ``filetracking``; the binding alone opts in, whatever ``HUD_FILE_TRACKING_ENABLED``
    says. A failed tracking setup logs a warning and the block runs untracked.
    """
    if settings.trace_dir is None and not settings.reporting_enabled:
        yield
        return
    try:
        cap = client.binding("filetracking")
    except (KeyError, RuntimeError):
        yield
        return

    # Env images whose tracker lacks ``setup_diff`` can only ``advance`` past setup; both
    # calls set the baseline agent edits are diffed against.
    ft: FileTrackingClient | None = None
    try:
        ft = await FileTrackingClient.connect(cap)
        if cap.params.get("setup_diff") is True:
            setup = await ft.call("setup")
            export(trace_id, name="filetracking.setup", schema=_FILETRACKING_SCHEMA, payload=setup)
        else:
            await ft.call("advance")
        snapshot = await ft.call("snapshot")
        export(
            trace_id, name="filetracking.snapshot", schema=_FILETRACKING_SCHEMA, payload=snapshot
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
                export(
                    trace_id,
                    name="filetracking.diff",
                    schema=_FILETRACKING_SCHEMA,
                    payload=diff,
                    started_at=started_at,
                )

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
        with contextlib.suppress(Exception):
            await ft.close()
