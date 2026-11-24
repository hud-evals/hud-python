"""Trace propagation: the ``Trace-Id`` a rollout sends, and the one an MCP request carries."""

from __future__ import annotations

import contextvars
from contextlib import contextmanager
from typing import TYPE_CHECKING

from fastmcp.server.dependencies import get_context, get_http_headers

if TYPE_CHECKING:
    from collections.abc import Generator

# MCP request meta key carrying the caller's trace id.
TRACE_ID_META_KEY = "hud/trace-id"

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
    return dict(headers) if headers is not None else {}


def get_mcp_trace_id() -> str | None:
    """The trace id on the active inbound MCP request, or ``None``.

    The ``hud/trace-id`` request meta key wins over the ``trace-id`` HTTP header.
    """
    try:
        context = get_context()
    except RuntimeError:
        return None
    request_context = context.request_context
    meta = (request_context.meta if request_context else None) or {}
    trace_id = meta.get(TRACE_ID_META_KEY)
    if trace_id is None:
        trace_id = get_http_headers().get("trace-id")
    if trace_id is not None and not isinstance(trace_id, str):
        raise TypeError(f"{TRACE_ID_META_KEY} must be a string")
    return trace_id


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
