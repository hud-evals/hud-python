"""Batching span exporter for the HUD telemetry backend.

``queue_span`` hands each span to one background intake worker that batches by
count *and* serialized byte-size, then dispatches each per-trace batch to a small
pool of upload workers over a pooled HTTP connection — so the large image frames
a robot rollout emits every tick upload in parallel instead of serially behind
one connection. ``flush`` drains the queue and waits for the in-flight uploads to
*finish* (not a fixed sleep); it also runs at interpreter exit.

The upload workers are this module's own daemon threads rather than a
``ThreadPoolExecutor``: the standard library shuts executors down before
``atexit`` handlers run, which would leave the exit flush nothing to upload with.
"""

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

import httpx

from hud.telemetry.span import TASK_RUN_ID_ATTRIBUTE
from hud.utils import make_request_sync

logger = logging.getLogger(__name__)

# 8 parallel uploads with 4 MiB / 100-span batches drains a rollout's image
# frames fastest without oversized POSTs.
_UPLOAD_WORKERS = 8
_MAX_BATCH_SPANS = 100
_MAX_BATCH_BYTES = 4 * 1024 * 1024
_FLUSH_INTERVAL = 1.0
_UPLOAD_RETRIES = 2
_UPLOAD_RETRY_DELAY = 0.5
_HTTP_TIMEOUT = httpx.Timeout(connect=10.0, read=60.0, write=60.0, pool=10.0)


class _Marker(threading.Event):
    """An in-band flush (or stop) marker the intake worker honors in queue order."""

    def __init__(self, *, stop: bool = False) -> None:
        super().__init__()
        self.stop = stop


_Upload = tuple[str, list[dict[str, Any]], str, str]

# The intake worker owns all batching state; it and the upload workers are
# daemons that run for the process's life. ``_lock`` guards the thread/client
# handles and ``_pending``, the count of uploads queued but not yet finished.
_queue: queue.Queue[dict[str, Any] | _Marker] = queue.Queue()
_uploads: queue.Queue[_Upload | None] = queue.Queue()
_pending = 0
_worker: threading.Thread | None = None
_uploaders: list[threading.Thread] = []
_client: httpx.Client | None = None
_lock = threading.Lock()
_idle = threading.Condition(_lock)

# Local file exporter — the second export target, independent of the backend.
_local_lock = threading.Lock()


def _export_local(span: dict[str, Any], local_dir: str | None) -> None:
    """Append one span as a JSON line to ``<local_dir>/<trace_id>.jsonl``.

    Runs regardless of ``telemetry_enabled`` / ``api_key``. Set
    ``HUD_TELEMETRY_LOCAL_DIR`` to choose a directory, or leave it unset while
    uploads are off to write under ``~/.hud/spans``. Best-effort.
    """
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
    """Export a span: to the local file exporter (if set) and the HUD backend."""
    from hud.settings import settings

    if not span.get("attributes", {}).get(TASK_RUN_ID_ATTRIBUTE):
        return
    _export_local(span, settings.span_dir)
    if not settings.telemetry_enabled or not settings.api_key:
        return
    _ensure_worker()
    _queue.put(span)


def flush(timeout: float = 10.0) -> bool:
    """Drain queued spans and wait for their uploads to finish.

    Puts a marker behind everything queued so far, waits for the worker to reach
    it, then waits for the dispatched uploads to complete. Returns ``False`` if it
    did not fully drain within ``timeout``.
    """
    with _lock:
        worker = _worker
    if worker is None or not worker.is_alive():
        return True

    deadline = time.monotonic() + timeout
    marker = _Marker()
    _queue.put(marker)
    if not marker.wait(max(0.0, deadline - time.monotonic())):
        return False
    with _idle:
        return _idle.wait_for(lambda: _pending == 0, max(0.0, deadline - time.monotonic()))


def reset(timeout: float = 30.0) -> None:
    """Flush, stop the workers, and close the HTTP client (tests/benchmarks)."""
    global _worker, _client, _pending
    with _lock:
        worker, uploaders, client = _worker, list(_uploaders), _client
    if worker is not None and worker.is_alive():
        flush(timeout)
        stop = _Marker(stop=True)
        _queue.put(stop)
        stop.wait(timeout)
        worker.join(timeout)
    for _ in uploaders:
        _uploads.put(None)
    for uploader in uploaders:
        uploader.join(timeout)
    if client is not None:
        client.close()
    with _lock:
        _worker = _client = None
        _uploaders.clear()
        _pending = 0


def _ensure_worker() -> None:
    global _worker, _client
    with _lock:
        if _worker is not None and _worker.is_alive():
            return
        _client = httpx.Client(
            timeout=_HTTP_TIMEOUT,
            limits=httpx.Limits(
                max_connections=_UPLOAD_WORKERS * 2,
                max_keepalive_connections=_UPLOAD_WORKERS * 2,
                keepalive_expiry=30.0,
            ),
        )
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
        if isinstance(item, _Marker):
            batch, nbytes = _dispatch(batch)
            item.set()
            if item.stop:
                return
            continue
        batch.append(item)
        nbytes += _span_bytes(item)
        if len(batch) >= _MAX_BATCH_SPANS or nbytes >= _MAX_BATCH_BYTES:
            batch, nbytes = _dispatch(batch)


def _dispatch(batch: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    """Queue one upload per trace in the batch; return an empty batch."""
    global _pending
    from hud.settings import settings

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
    while (upload := _uploads.get()) is not None:
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


__all__ = ["flush", "queue_span", "reset"]
