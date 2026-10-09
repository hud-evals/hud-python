"""Capped rollouts: the scripted answers they get and what a clean lifecycle leaves in the spans."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from tests.harness import say

if TYPE_CHECKING:
    from tests.harness import ModelRequest, Turn
    from tests.harness.cli import Result

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


def scripted(request: ModelRequest) -> Turn:
    """The agent answers at once; the LLM judge finds every criterion met."""
    if request.protocol == "chat":
        return say(json.dumps({"criterion_status": "MET", "explanation": "scripted judge"}))
    return say("done")


def evaluated(result: Result) -> dict[str, Any]:
    assert result.exit_code == 0, result
    assert result.json["error_count"] == 0, result
    (run,) = result.json["runs"]
    return run
