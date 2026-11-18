"""Batching span exporter for the HUD telemetry backend."""

from __future__ import annotations

import atexit
import json
import logging
import queue
import threading
import time
from collections import defaultdict
from typing import TYPE_CHECKING, Any, cast

import httpx2

from hud.settings import settings
from hud.telemetry.span import TASK_RUN_ID_ATTRIBUTE, Span, record_span
from hud.utils.requests import make_request_sync
from hud.utils.time import now_iso

if TYPE_CHECKING:
    from pathlib import Path

    from hud.types import Step
    from hud.utils.serialization import JsonObject

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
# Local directories already reported unwritable, so each is warned about once.
_unwritable: set[Path] = set()


def write_local(span: Span) -> None:
    """Append ``span`` to its trace's local file, when local trace files are on.

    A directory that can't be written is warned about once and doesn't fail the run.
    """
    directory = settings.local_trace_dir
    if directory is None:
        return
    try:
        directory.mkdir(parents=True, exist_ok=True)
        line = span.model_dump_json()
        with _local_lock, (directory / f"{span.trace_id}.jsonl").open("a", encoding="utf-8") as f:
            f.write(line + "\n")
    except OSError as exc:
        with _local_lock:
            first = directory not in _unwritable
            _unwritable.add(directory)
        if first:
            logger.warning("local trace files can't be written to %s: %s", directory, exc)


def export_span(span: Span) -> None:
    """Write ``span`` to its local trace file and queue it for the HUD platform.

    The upload needs telemetry on and an API key.
    """
    write_local(span)
    if not settings.reporting_enabled:
        return
    _ensure_worker()
    _queue.put(span.model_dump(mode="json"))


def export_step(trace_id: str, step: Step[Any]) -> None:
    """Export ``step`` as one span of the trace ``trace_id``."""
    now = now_iso()
    export_span(
        record_span(
            trace_id,
            name=f"step.{step.source}",
            schema=step.schema_tag,
            payload=cast("JsonObject", step.model_dump(mode="json", exclude_none=True)),
            started_at=step.started_at or now,
            ended_at=step.ended_at or now,
            error=step.error,
        )
    )


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
