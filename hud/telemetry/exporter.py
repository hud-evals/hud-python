"""Batching span exporter for the HUD telemetry backend.

``queue_span`` hands each span to one background intake worker that batches by
count *and* serialized byte-size, then dispatches each per-trace batch to a small
pool of upload workers over a pooled HTTP connection — so the large image frames
a robot rollout emits every tick upload in parallel instead of serially behind
one connection. ``flush`` drains the queue and waits for the in-flight uploads to
*finish* (not a fixed sleep); it also runs at interpreter exit.
"""

from __future__ import annotations

import atexit
import json
import logging
import queue
import threading
import time
from collections import defaultdict
from concurrent.futures import Future, ThreadPoolExecutor, wait
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


# The worker owns all batching state; it is a daemon and runs for the process's
# life. ``_lock`` guards the worker/pool/client handles and the in-flight set.
_queue: queue.Queue[dict[str, Any] | _Marker] = queue.Queue()
_inflight: set[Future[None]] = set()
_worker: threading.Thread | None = None
_pool: ThreadPoolExecutor | None = None
_client: httpx.Client | None = None
_lock = threading.Lock()


def queue_span(span: dict[str, Any]) -> None:
    """Queue a span for batched, parallel background export."""
    from hud.settings import settings

    if not settings.telemetry_enabled or not settings.api_key:
        return
    if not span.get("attributes", {}).get(TASK_RUN_ID_ATTRIBUTE):
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
    with _lock:
        pending = set(_inflight)
    if not pending:
        return True
    _done, not_done = wait(pending, timeout=max(0.0, deadline - time.monotonic()))
    return not not_done


def reset(timeout: float = 30.0) -> None:
    """Flush, stop the worker, and tear down the pool/client (tests/benchmarks)."""
    global _worker, _pool, _client
    with _lock:
        worker, pool, client = _worker, _pool, _client
    if worker is not None and worker.is_alive():
        flush(timeout)
        stop = _Marker(stop=True)
        _queue.put(stop)
        stop.wait(timeout)
        worker.join(timeout)
    if pool is not None:
        pool.shutdown(wait=True)
    if client is not None:
        client.close()
    with _lock:
        _worker = _pool = _client = None
        _inflight.clear()


def _ensure_worker() -> None:
    global _worker, _pool, _client
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
        _pool = ThreadPoolExecutor(_UPLOAD_WORKERS, thread_name_prefix="hud-telemetry-upload")
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
    """Submit one upload per trace in the batch to the pool; return an empty batch."""
    from hud.settings import settings

    pool, api_key = _pool, settings.api_key
    if not batch or pool is None or not api_key:
        return [], 0
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for span in batch:
        grouped[span["attributes"][TASK_RUN_ID_ATTRIBUTE]].append(span)
    for task_run_id, spans in grouped.items():
        future = pool.submit(_do_upload, task_run_id, spans, settings.hud_telemetry_url, api_key)
        with _lock:
            _inflight.add(future)
        future.add_done_callback(_retire)
    return [], 0


def _retire(future: Future[None]) -> None:
    with _lock:
        _inflight.discard(future)


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
