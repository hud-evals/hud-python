"""Core trajectory contract tests: ``Trace`` invariants + step span emission."""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

from mcp import types as mcp_types

from hud.telemetry.context import set_trace_context
from hud.types import Step, Trace


def test_trace_final_returns_newest_non_none_answer():
    """final() asks newest-first; None means "no answer", falsy answers win."""
    trace = Trace()
    trace.record(Step(source="agent", extra={"note": "first"}))
    trace.record(Step(source="agent", extra={"note": ""}))
    trace.record(Step(source="tool"))

    assert trace.final(lambda s: s.extra.get("note")) == ""
    assert trace.final(lambda s: s.error) is None


def test_trace_collect_gathers_answers_in_step_order():
    """collect() keeps step order and skips steps that answer None."""
    trace = Trace()
    trace.record(Step(source="agent", extra={"n": 1}))
    trace.record(Step(source="tool"))
    trace.record(Step(source="agent", extra={"n": 2}))

    assert trace.collect(lambda s: s.extra.get("n")) == [1, 2]


def test_trace_append_numbers_steps():
    """Trace.append assigns sequential 1-based step ids."""
    trace = Trace()
    trace.record(Step(source="user"))
    trace.record(Step(source="agent"))
    assert len(trace) == 2
    assert [step.step_id for step in trace.steps] == [1, 2]


def test_trace_validator_numbers_preloaded_steps():
    """Steps passed to the constructor are renumbered on validation."""
    trace = Trace(steps=[Step(source="user"), Step(source="agent"), Step(source="tool")])
    assert [step.step_id for step in trace.steps] == [1, 2, 3]


def test_trace_error_surfaces_last_step_error():
    """Trace.error reads the most recent step error; is_error follows status."""
    trace = Trace()
    assert trace.error is None
    assert trace.is_error is False

    trace.record(Step(source="tool", error="first"))
    trace.record(Step(source="agent"))
    trace.record(Step(source="system", error="second"))
    trace.status = "error"

    assert trace.error == "second"
    assert trace.is_error is True


def test_step_emit_wraps_step_in_schema_tagged_span():
    captured: list[dict[str, Any]] = []
    step = Step(
        source="user",
        messages=[
            mcp_types.PromptMessage(
                role="user",
                content=mcp_types.TextContent(type="text", text="do the thing"),
            ),
        ],
    )

    with (
        patch("hud.types.queue_span", side_effect=captured.append),
        set_trace_context("run-1"),
    ):
        step.emit()

    (span,) = captured
    assert span["name"] == "step.user"
    assert span["attributes"]["hud.schema"] == "hud.step.v1"
    assert span["attributes"]["hud.task_run_id"] == "run-1"
    payload = span["attributes"]["hud.payload"]
    assert payload["source"] == "user"
    assert payload["messages"][0]["content"]["text"] == "do the thing"
    assert span["status_code"] == "OK"


def test_step_emit_marks_error_status():
    captured: list[dict[str, Any]] = []

    with (
        patch("hud.types.queue_span", side_effect=captured.append),
        set_trace_context("run-1"),
    ):
        Step(source="system", error="boom").emit()

    (span,) = captured
    assert span["status_code"] == "ERROR"
    assert span["status_message"] == "boom"


def test_step_emit_without_context_is_noop():
    captured: list[dict[str, Any]] = []

    with patch("hud.types.queue_span", side_effect=captured.append):
        Step(source="system", error="boom").emit()

    assert captured == []


def test_trace_record_emits_and_stamps_end():
    """record = number + stamp end + append + emit, in one call."""
    captured: list[dict[str, Any]] = []
    trace = Trace()

    with (
        patch("hud.types.queue_span", side_effect=captured.append),
        set_trace_context("run-1"),
    ):
        trace.record(Step(source="user"))
        trace.record(Step(source="agent", ended_at="2026-05-14T20:00:05Z"))

    assert [span["attributes"]["hud.payload"]["step_id"] for span in captured] == [1, 2]
    assert trace.steps[0].ended_at is not None  # stamped at record time
    assert trace.steps[1].ended_at == "2026-05-14T20:00:05Z"  # explicit timing kept
    assert captured[1]["end_time"] == "2026-05-14T20:00:05Z"
