"""Batching span exporter for the HUD telemetry backend."""

from __future__ import annotations

import atexit
import json
import logging
import queue
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import httpx2

from hud.settings import settings
from hud.telemetry.span import TASK_RUN_ID_ATTRIBUTE
from hud.utils.requests import make_request_sync

logger = logging.getLogger(__name__)

# Upload parallelism and batch caps sized for image-heavy rollouts without oversized POSTs.
_UPLOAD_WORKERS = 8
_MAX_BATCH_SPANS = 100
_MAX_BATCH_BYTES = 4 * 1024 * 1024
_FLUSH_INTERVAL = 1.0
_UPLOAD_RETRIES = 2
_UPLOAD_RETRY_DELAY = 0.5
_HTTP_TIMEOUT = httpx2.Timeout(connect=10.0, read=60.0, write=60.0, pool=10.0)


_Upload = tuple[str, list[dict[str, Any]], str, str]

# The intake worker owns all batching state; it and the upload workers are
# daemons that run for the process's life. ``_lock`` guards the thread/client
# handles and ``_pending``, the count of uploads queued but not yet finished.
_queue: queue.Queue[dict[str, Any] | threading.Event] = queue.Queue()
_uploads: queue.Queue[_Upload] = queue.Queue()
_pending = 0
_worker: threading.Thread | None = None
_uploaders: list[threading.Thread] = []
_client: httpx2.Client | None = None
_lock = threading.Lock()
_idle = threading.Condition(_lock)

_local_lock = threading.Lock()


def _export_local(span: dict[str, Any], local_dir: str | None) -> None:
    """Best-effort append of one span to ``<local_dir>/<trace_id>.jsonl``."""
    if not local_dir:
        return
    # Only I/O is best-effort: a config-shape error (e.g. a mock or wrong type
    # reaching settings) must fail loudly, not write to a repr-named path.
    if not isinstance(local_dir, str):
        raise TypeError(f"telemetry_local_dir must be a str path, got {type(local_dir).__name__}")
    try:
        path = Path(local_dir)
        path.mkdir(parents=True, exist_ok=True)
        trace_id = span.get("trace_id") or "unknown"
        line = json.dumps(span, ensure_ascii=False)
        with _local_lock, (path / f"{trace_id}.jsonl").open("a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        logger.debug("local span export failed", exc_info=True)


def queue_span(span: dict[str, Any]) -> None:
    """Write a span to the local span dir (if set) and queue it for the HUD backend.

    Spans without a ``hud.task_run_id`` attribute are dropped. The backend upload needs
    telemetry enabled and an API key.
    """
    if not span.get("attributes", {}).get(TASK_RUN_ID_ATTRIBUTE):
        return
    _export_local(span, settings.span_dir)
    if not settings.reporting_enabled:
        return
    _ensure_worker()
    _queue.put(span)


def flush(timeout: float = 10.0) -> bool:
    """Drain queued spans and wait for their uploads to finish.

    Returns ``False`` if it did not fully drain within ``timeout``.
    """
    with _lock:
        worker = _worker
    if worker is None or not worker.is_alive():
        return True

    deadline = time.monotonic() + timeout
    marker = threading.Event()
    _queue.put(marker)
    if not marker.wait(max(0.0, deadline - time.monotonic())):
        return False
    with _idle:
        return _idle.wait_for(lambda: _pending == 0, max(0.0, deadline - time.monotonic()))


def _ensure_worker() -> None:
    global _worker, _client
    with _lock:
        if _worker is not None and _worker.is_alive():
            return
        _client = httpx2.Client(
            timeout=_HTTP_TIMEOUT,
            limits=httpx2.Limits(
                max_connections=_UPLOAD_WORKERS * 2,
                max_keepalive_connections=_UPLOAD_WORKERS * 2,
                keepalive_expiry=30.0,
            ),
        )
        # Plain daemon threads, not a ThreadPoolExecutor: executors shut down before
        # atexit handlers run, which would leave the exit flush nothing to upload with.
        _uploaders[:] = [
            threading.Thread(target=_upload_loop, name=f"hud-telemetry-upload-{i}", daemon=True)
            for i in range(_UPLOAD_WORKERS)
        ]
        for uploader in _uploaders:
            uploader.start()
        _worker = threading.Thread(target=_run, name="hud-telemetry-export", daemon=True)
        _worker.start()


def _run() -> None:
    batch: list[dict[str, Any]] = []
    nbytes = 0
    while True:
        try:
            item = _queue.get(timeout=_FLUSH_INTERVAL)
        except queue.Empty:
            batch, nbytes = _dispatch(batch)
            continue
        if isinstance(item, threading.Event):
            batch, nbytes = _dispatch(batch)
            item.set()
            continue
        batch.append(item)
        nbytes += _span_bytes(item)
        if len(batch) >= _MAX_BATCH_SPANS or nbytes >= _MAX_BATCH_BYTES:
            batch, nbytes = _dispatch(batch)


def _dispatch(batch: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    """Queue one upload per trace in the batch; return an empty batch."""
    global _pending
    api_key = settings.api_key
    if not batch or not api_key:
        return [], 0
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for span in batch:
        grouped[span["attributes"][TASK_RUN_ID_ATTRIBUTE]].append(span)
    for task_run_id, spans in grouped.items():
        with _lock:
            _pending += 1
        _uploads.put((task_run_id, spans, settings.hud_telemetry_url, api_key))
    return [], 0


def _upload_loop() -> None:
    global _pending
    while True:
        upload = _uploads.get()
        try:
            _do_upload(*upload)
        finally:
            with _idle:
                _pending -= 1
                _idle.notify_all()


def _do_upload(
    task_run_id: str,
    spans: list[dict[str, Any]],
    telemetry_url: str,
    api_key: str,
) -> None:
    url = f"{telemetry_url}/trace/{task_run_id}/telemetry-upload"
    try:
        make_request_sync(
            method="POST",
            url=url,
            json={"telemetry": spans},
            api_key=api_key,
            max_retries=_UPLOAD_RETRIES,
            retry_delay=_UPLOAD_RETRY_DELAY,
            client=_client,
        )
    except Exception as exc:
        logger.warning(
            "telemetry upload failed for trace %s (%d spans): %s", task_run_id, len(spans), exc
        )


def _span_bytes(span: dict[str, Any]) -> int:
    try:
        return len(json.dumps(span, default=str))
    except (TypeError, ValueError):
        return 0


atexit.register(lambda: flush(timeout=30.0))
