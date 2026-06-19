"""Emit file-tracking observations as ``hud.filetracking.v1`` telemetry spans.

A standalone span schema — *not* attached to tool-call results. The rollout
observer pulls a diff from the workspace's ``filetracking/1`` capability and
calls :func:`emit_file_diff`, which builds an OTel-shaped span under the active
trace context and hands it to the exporter. Each span is self-timestamped so
the viewer correlates file changes to the step timeline by time.
"""

from __future__ import annotations

from typing import Any

from hud.telemetry.context import get_current_trace_id
from hud.telemetry.exporter import queue_span
from hud.telemetry.span import (
    PAYLOAD_ATTRIBUTE,
    SCHEMA_ATTRIBUTE,
    TASK_RUN_ID_ATTRIBUTE,
    Span,
    new_span_id,
    normalize_trace_id,
)
from hud.utils.time import now_iso

#: Schema tag the platform projector dispatches on to build ``file_change`` /
#: ``file_snapshot`` events.
FILETRACKING_SCHEMA = "hud.filetracking.v1"

DIFF_SPAN_NAME = "filetracking.diff"
SNAPSHOT_SPAN_NAME = "filetracking.snapshot"


def _emit(payload: dict[str, Any], name: str, started_at: str, ended_at: str | None) -> bool:
    """Build and queue one file-tracking span. No-ops outside a rollout."""
    task_run_id = get_current_trace_id()
    if task_run_id is None:
        return False
    span = Span(
        name=name,
        trace_id=normalize_trace_id(task_run_id),
        span_id=new_span_id(),
        start_time=started_at,
        end_time=ended_at or now_iso(),
        status_code="OK",
        attributes={
            SCHEMA_ATTRIBUTE: FILETRACKING_SCHEMA,
            TASK_RUN_ID_ATTRIBUTE: task_run_id,
            PAYLOAD_ATTRIBUTE: payload,
        },
    )
    queue_span(span.model_dump(mode="json"))
    return True


def emit_file_diff(
    payload: dict[str, Any], *, started_at: str, ended_at: str | None = None
) -> bool:
    """Emit a per-scan diff (``DiffResult.to_dict()`` shape); ``False`` outside a rollout."""
    return _emit(payload, DIFF_SPAN_NAME, started_at, ended_at)


def emit_file_snapshot(
    payload: dict[str, Any], *, started_at: str, ended_at: str | None = None
) -> bool:
    """Emit the baseline manifest (``{files, files_scanned}``). The reconstruction anchor."""
    return _emit(payload, SNAPSHOT_SPAN_NAME, started_at, ended_at)


__all__ = [
    "DIFF_SPAN_NAME",
    "FILETRACKING_SCHEMA",
    "SNAPSHOT_SPAN_NAME",
    "emit_file_diff",
    "emit_file_snapshot",
]
