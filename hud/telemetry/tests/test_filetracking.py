"""The hud.filetracking.v1 span emitter."""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

from hud.telemetry.context import set_trace_context
from hud.telemetry.filetracking import (
    DIFF_SPAN_NAME,
    FILETRACKING_SCHEMA,
    SNAPSHOT_SPAN_NAME,
    emit_file_diff,
    emit_file_snapshot,
)

_PAYLOAD = {"files_changed": 1, "patches": [{"path": "a.txt", "status": "modified"}]}


def test_emit_noops_without_a_trace_context() -> None:
    with patch("hud.telemetry.filetracking.queue_span") as queue_span:
        emitted = emit_file_diff(_PAYLOAD, started_at="2026-06-18T22:00:00Z")
    assert emitted is False
    queue_span.assert_not_called()


def test_emit_builds_a_schema_tagged_span() -> None:
    captured: list[dict[str, Any]] = []
    with (
        patch("hud.telemetry.filetracking.queue_span", side_effect=captured.append),
        set_trace_context("run-abc-123"),
    ):
        emitted = emit_file_diff(_PAYLOAD, started_at="2026-06-18T22:00:00Z")

    assert emitted is True
    assert len(captured) == 1
    span = captured[0]
    assert span["name"] == DIFF_SPAN_NAME
    attributes = span["attributes"]
    assert attributes["hud.schema"] == FILETRACKING_SCHEMA
    assert attributes["hud.task_run_id"] == "run-abc-123"
    assert attributes["hud.payload"]["files_changed"] == 1
    # trace_id is the normalized 32-hex telemetry id, not the raw run id.
    assert len(span["trace_id"]) == 32


def test_emit_snapshot_uses_the_snapshot_span_name() -> None:
    captured: list[dict[str, Any]] = []
    manifest = {"files_scanned": 2, "files": [{"path": "a", "size": 1, "content_hash": "h"}]}
    with (
        patch("hud.telemetry.filetracking.queue_span", side_effect=captured.append),
        set_trace_context("run-abc-123"),
    ):
        emitted = emit_file_snapshot(manifest, started_at="2026-06-18T22:00:00Z")

    assert emitted is True
    span = captured[0]
    assert span["name"] == SNAPSHOT_SPAN_NAME
    assert span["attributes"]["hud.schema"] == FILETRACKING_SCHEMA
    assert span["attributes"]["hud.payload"]["files_scanned"] == 2
