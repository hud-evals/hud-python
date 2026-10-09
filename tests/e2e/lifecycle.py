"""What a cleanly completed task lifecycle looks like in a run's step spans."""

from __future__ import annotations

from typing import Any

# Tool steps may fail without failing the run: a real model can call a tool badly.
LIFECYCLE_SOURCES = {"task", "agent", "system"}


def task_calls(steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The ``task_call`` of every task step, in order."""
    return [step["task_call"] for step in steps if step["source"] == "task"]


def assert_clean_lifecycle(steps: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Assert the run set up its task, graded it, and errored nowhere; return both results."""
    errors = [
        step["error"] for step in steps if step["source"] in LIFECYCLE_SOURCES and step.get("error")
    ]
    assert not errors, errors
    calls = task_calls(steps)
    assert [call["phase"] for call in calls] == ["setup", "evaluate"], calls
    setup, grade = (call["result"] for call in calls)
    assert setup.get("prompt"), f"task setup returned no prompt: {setup}"
    assert not grade.get("isError"), grade
    assert isinstance(grade["score"], int | float), grade
    assert 0 <= grade["score"] <= 1, grade
    return setup, grade
