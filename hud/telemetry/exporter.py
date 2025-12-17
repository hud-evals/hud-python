"""High-performance span exporter for HUD telemetry backend.

This module provides a lightweight span exporter that sends spans to the HUD
telemetry API immediately, using a thread pool to avoid blocking async code.

No OpenTelemetry dependency required.
"""

from __future__ import annotations

import atexit
import concurrent.futures as cf
import contextlib
import logging
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from hud.shared import make_request_sync

logger = logging.getLogger(__name__)

# Global singleton thread pool for span exports
_export_executor: ThreadPoolExecutor | None = None

# Pending futures for shutdown coordination
_pending_futures: list[cf.Future[bool]] = []

# Spans waiting to be flushed at context exit (per task_run_id)
_pending_spans: dict[str, list[dict[str, Any]]] = defaultdict(list)


def _get_export_executor() -> ThreadPoolExecutor:
    """Get or create the global thread pool for span exports."""
    global _export_executor
    if _export_executor is None:
        _export_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="span-export")

        def cleanup() -> None:
            if _export_executor is not None:
                _export_executor.shutdown(wait=True)

        atexit.register(cleanup)
    return _export_executor


def _do_upload(
    task_run_id: str,
    spans: list[dict[str, Any]],
    telemetry_url: str,
    api_key: str,
) -> bool:
    """Upload spans to HUD API (sync, runs in thread pool)."""
    try:
        url = f"{telemetry_url}/trace/{task_run_id}/telemetry-upload"
        payload: dict[str, Any] = {"telemetry": spans}

        logger.debug("Uploading %d spans to %s", len(spans), url)
        make_request_sync(
            method="POST",
            url=url,
            json=payload,
            api_key=api_key,
        )
        return True
    except Exception as e:
        logger.debug("Failed to upload spans for task %s: %s", task_run_id, e)
        return False


def _get_api_key() -> str | None:
    """Get the API key - prefer context override, fallback to settings."""
    from hud.eval.context import get_current_api_key
    from hud.settings import settings

    return get_current_api_key() or settings.api_key


def queue_span(span: dict[str, Any]) -> None:
    """Queue a span and immediately upload it (non-blocking).

    Uses thread pool to upload without blocking the event loop.
    """
    from hud.settings import settings

    api_key = _get_api_key()
    if not api_key or not settings.telemetry_enabled:
        return

    task_run_id = span.get("attributes", {}).get("task_run_id")
    if not task_run_id:
        return

    # Store for potential re-flush at context exit
    _pending_spans[task_run_id].append(span)

    # Capture api_key for upload closure (context may change)
    upload_api_key = api_key

    # Upload immediately via thread pool
    import asyncio

    try:
        loop = asyncio.get_running_loop()
        # In async context - use thread pool
        executor = _get_export_executor()

        def _upload() -> bool:
            return _do_upload(task_run_id, [span], settings.hud_telemetry_url, upload_api_key)

        future = loop.run_in_executor(executor, _upload)
        _pending_futures.append(future)  # type: ignore[arg-type]

        def _cleanup_done(f: cf.Future[bool]) -> None:
            with contextlib.suppress(Exception):
                _ = f.exception()
            with contextlib.suppress(ValueError):
                _pending_futures.remove(f)
            # Remove from pending spans on success
            if not f.exception():
                with contextlib.suppress(Exception):
                    if task_run_id in _pending_spans and span in _pending_spans[task_run_id]:
                        _pending_spans[task_run_id].remove(span)

        future.add_done_callback(_cleanup_done)  # type: ignore[arg-type]

    except RuntimeError:
        # No event loop - upload synchronously
        if _do_upload(task_run_id, [span], settings.hud_telemetry_url, upload_api_key):
            with contextlib.suppress(Exception):
                if task_run_id in _pending_spans and span in _pending_spans[task_run_id]:
                    _pending_spans[task_run_id].remove(span)


def flush(task_run_id: str | None = None) -> None:
    """Flush any pending spans (called at context exit).

    This ensures any spans that failed to upload are retried.

    Args:
        task_run_id: Optional task run ID to flush. If None, flushes all.
    """
    from hud.settings import settings

    api_key = _get_api_key()
    if not api_key or not settings.telemetry_enabled:
        _pending_spans.clear()
        return

    if task_run_id:
        # Flush specific task
        spans = _pending_spans.pop(task_run_id, [])
        if spans:
            _do_upload(task_run_id, spans, settings.hud_telemetry_url, api_key)
    else:
        # Flush all
        for tid, spans in list(_pending_spans.items()):
            if spans:
                _do_upload(tid, spans, settings.hud_telemetry_url, api_key)
        _pending_spans.clear()


def shutdown(timeout: float = 10.0) -> bool:
    """Shutdown and wait for pending exports.

    Args:
        timeout: Maximum time to wait in seconds

    Returns:
        True if all exports completed, False if timed out
    """
    # Wait for pending async exports
    if _pending_futures:
        try:
            done, not_done = cf.wait(_pending_futures, timeout=timeout)
            for f in done:
                with contextlib.suppress(Exception):
                    _ = f.exception()
            _pending_futures.clear()

            # Flush any remaining spans synchronously
            flush()

            return len(not_done) == 0
        except Exception:
            return False

    # Flush any remaining spans
    flush()
    return True


# Register shutdown handler
atexit.register(lambda: shutdown(timeout=5.0))


__all__ = [
    "flush",
    "queue_span",
    "shutdown",
]
