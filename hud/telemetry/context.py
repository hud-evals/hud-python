"""Trace context: the per-rollout ``Trace-Id`` contextvar."""

from __future__ import annotations

import contextvars
from contextlib import contextmanager
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Generator

_current_trace_headers: contextvars.ContextVar[dict[str, str] | None] = contextvars.ContextVar(
    "current_trace_headers", default=None
)


def get_current_trace_id() -> str | None:
    """Get the current trace ID (task_run_id) from context, or None."""
    headers = _current_trace_headers.get()
    return headers.get("Trace-Id") if headers else None


def get_trace_headers() -> dict[str, str]:
    """A copy of the active trace headers, or ``{}`` outside a trace."""
    headers = _current_trace_headers.get()
    if headers is not None:
        return dict(headers)
    trace_id = get_current_trace_id()
    return {"Trace-Id": trace_id} if trace_id is not None else {}


@contextmanager
def set_trace_context(
    trace_id: str,
    *,
    parent_trace_id: str | None = None,
) -> Generator[None, None, None]:
    """Temporarily bind an active trace and its immediate parent."""
    headers = {"Trace-Id": trace_id}
    if parent_trace_id is not None:
        headers["X-HUD-Parent-Trace-Id"] = parent_trace_id
    token = _current_trace_headers.set(headers)
    try:
        yield
    finally:
        _current_trace_headers.reset(token)
