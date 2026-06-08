"""Trace context: the per-rollout ``Trace-Id`` contextvar.

Standalone (no env/eval dependency) so any layer — the new ``Run``/``Taskset``
flow, ``@instrument``, the exporter, or the legacy eval context — can set and
read the active trace without importing the environment stack.
"""

from __future__ import annotations

import contextvars
from contextlib import contextmanager
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Generator

# Current trace headers (for span attribution via @instrument).
_current_trace_headers: contextvars.ContextVar[dict[str, str] | None] = contextvars.ContextVar(
    "current_trace_headers", default=None
)


def get_current_trace_id() -> str | None:
    """Get the current trace ID (task_run_id) from context, or None.

    Used by ``@instrument`` to know where to send telemetry.
    """
    headers = _current_trace_headers.get()
    if headers:
        return headers.get("Trace-Id")
    return None


@contextmanager
def set_trace_context(trace_id: str) -> Generator[None, None, None]:
    """Temporarily bind ``trace_id`` as the active trace (for span attribution)."""
    token = _current_trace_headers.set({"Trace-Id": trace_id})
    try:
        yield
    finally:
        _current_trace_headers.reset(token)


__all__ = [
    "get_current_trace_id",
    "set_trace_context",
]
