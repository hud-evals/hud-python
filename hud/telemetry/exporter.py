"""Batching span exporter for the HUD telemetry backend.

``queue_span`` hands each span to one background daemon worker that batches by
trace and uploads. The worker owns all batching state; ``flush`` drains it and is
the only lifecycle primitive (it also runs at interpreter exit).
"""

from __future__ import annotations

import atexit
import logging
import queue
import threading
from collections import defaultdict
from typing import Any

from hud.telemetry.span import TASK_RUN_ID_ATTRIBUTE
from hud.utils import make_request_sync

logger = logging.getLogger(__name__)

_MAX_BATCH_SIZE = 100
_FLUSH_INTERVAL_SECONDS = 1.0

# A queued ``Event`` is a flush marker: the worker uploads the current batch and
# sets it. Spans carry their own ``hud.task_run_id`` (under ``attributes``), so
# the worker groups them without any extra per-span bookkeeping. The worker is a
# daemon and runs for the life of the process.
_export_queue: queue.Queue[dict[str, Any] | threading.Event] = queue.Queue()
_worker: threading.Thread | None = None
_worker_lock = threading.Lock()


def _do_upload(
    task_run_id: str,
    spans: list[dict[str, Any]],
    telemetry_url: str,
    api_key: str,
) -> None:
    try:
        url = f"{telemetry_url}/trace/{task_run_id}/telemetry-upload"
        logger.debug("Uploading %d spans to %s", len(spans), url)
        make_request_sync(method="POST", url=url, json={"telemetry": spans}, api_key=api_key)
    except Exception as exc:
        logger.debug("Failed to upload spans for task %s: %s", task_run_id, exc)


def queue_span(span: dict[str, Any]) -> None:
    """Queue a span for batched background export."""
    from hud.settings import settings

    if not settings.telemetry_enabled or not settings.api_key:
        return
    if not span.get("attributes", {}).get(TASK_RUN_ID_ATTRIBUTE):
        return

    _ensure_worker()
    _export_queue.put(span)


def flush(timeout: float = 10.0) -> bool:
    """Wait until spans queued before this call have been uploaded.

    Returns False if the worker did not drain within ``timeout``.
    """
    with _worker_lock:
        worker = _worker
    if worker is None or not worker.is_alive():
        return True

    drained = threading.Event()
    _export_queue.put(drained)
    return drained.wait(timeout)


def _ensure_worker() -> None:
    global _worker
    with _worker_lock:
        if _worker is None or not _worker.is_alive():
            _worker = threading.Thread(target=_run, name="hud-telemetry-export", daemon=True)
            _worker.start()


def _run() -> None:
    batch: list[dict[str, Any]] = []
    while True:
        try:
            item = _export_queue.get(timeout=_FLUSH_INTERVAL_SECONDS)
        except queue.Empty:
            batch = _upload(batch)
            continue
        if isinstance(item, threading.Event):
            batch = _upload(batch)
            item.set()
        else:
            batch.append(item)
            if len(batch) >= _MAX_BATCH_SIZE:
                batch = _upload(batch)


def _upload(batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not batch:
        return []
    from hud.settings import settings

    api_key = settings.api_key
    if not api_key:
        return []
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for span in batch:
        grouped[span["attributes"][TASK_RUN_ID_ATTRIBUTE]].append(span)
    for task_run_id, spans in grouped.items():
        _do_upload(task_run_id, spans, settings.hud_telemetry_url, api_key)
    return []


atexit.register(lambda: flush(timeout=5.0))


__all__ = ["flush", "queue_span"]
