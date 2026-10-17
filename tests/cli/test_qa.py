"""``hud qa``: list QA checks, run them on traces or tasks, and read their verdicts."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from tests.support import Hud, Platform

TRACE_ID = "00000000-0000-4000-a000-000000000002"
TASK_ID = "00000000-0000-4000-a000-000000000005"
CHECK = "failure_analysis"
OTHER_CHECK = "reward_hacking"
TASK_CHECK = "task_reward_exploit"
CHECKS = [
    {
        "key": CHECK,
        "subject_type": "trace",
        "title": "Failure Analysis",
        "question": "Why did the agent fail?",
    },
    {
        "key": OTHER_CHECK,
        "subject_type": "trace",
        "title": "Reward Hacking",
        "question": "Did the agent game the grader?",
    },
    {
        "key": TASK_CHECK,
        "subject_type": "task",
        "title": "Task Reward Exploit",
        "question": "Can the reward be earned without solving the task?",
    },
]


@pytest.fixture(autouse=True)
def catalog(platform: Platform) -> None:
    """The platform's QA check catalog, which ``hud qa`` reads to place checks by level."""
    platform.qa_checks = CHECKS


def row(status: str = "completed", **fields: Any) -> dict[str, Any]:
    """One ``/v2/qa/results`` row as the platform returns it."""
    return {
        "id": "00000000-0000-4000-a000-000000000003",
        "check_key": CHECK,
        "subject_type": "trace",
        "subject_trace_id": TRACE_ID,
        "source": "analysis",
        "status": status,
        "verdict": None,
        "result": None,
        "legacy_result": None,
        "note": None,
        "error": None,
        **fields,
    }


def verdict(tag: str, *, check_key: str = CHECK) -> dict[str, Any]:
    """A completed ``qa_agent_result.v1`` row."""
    result = {"schema_version": "qa_agent_result.v1", "verdict": tag, "summary": "", "findings": []}
    return row(check_key=check_key, verdict=tag, result=result)


def started(platform: Platform) -> list[tuple[str, str]]:
    """The (check, subject) pairs QA runs were started for."""
    return [
        (r["check_key"], r.get("subject_trace_id") or r["subject_task_id"])
        for r in platform.qa_results
    ]


@pytest.mark.parametrize(
    ("argv", "keys"),
    [
        (["qa", "--quiet"], [CHECK, OTHER_CHECK, TASK_CHECK]),
        (["qa", "list", "--level", "task", "--quiet"], [TASK_CHECK]),
    ],
)
def test_qa_lists_the_checks_at_a_level(hud: Hud, argv: list[str], keys: list[str]) -> None:
    result = hud(*argv)

    assert (result.exit_code, result.stdout.split()) == (0, keys), result


def test_qa_run_refuses_trace_and_task_checks_together(hud: Hud, platform: Platform) -> None:
    result = hud("qa", "run", f"{CHECK},{TASK_CHECK}", TRACE_ID, "--yes", "--json")

    assert result.exit_code == 2, result
    assert platform.qa_results == []


@pytest.mark.parametrize(
    ("argv", "runs"),
    [
        (
            [f"{CHECK}, {OTHER_CHECK},{CHECK}", TRACE_ID.upper()],
            [(CHECK, TRACE_ID), (OTHER_CHECK, TRACE_ID)],
        ),
        ([TASK_CHECK, TASK_ID], [(TASK_CHECK, TASK_ID)]),
    ],
    ids=["deduplicated-and-normalised", "task-level"],
)
def test_qa_run_without_waiting_starts_each_check_once(
    hud: Hud, platform: Platform, argv: list[str], runs: list[tuple[str, str]]
) -> None:
    result = hud("qa", "run", *argv, "--no-wait", "--yes", "--json")

    assert result.exit_code == 0, result
    assert started(platform) == runs


@pytest.mark.parametrize(
    ("outcome", "exit_code", "verdicts"),
    [
        pytest.param(verdict("passed"), 0, ["passed"], id="passed"),
        pytest.param(verdict("failed"), 1, ["failed"], id="failed"),
        pytest.param(row("error", error="invalid result"), 1, [None], id="errored"),
    ],
)
def test_qa_run_waits_for_every_check_and_fails_unless_all_pass(
    hud: Hud,
    platform: Platform,
    outcome: dict[str, Any],
    exit_code: int,
    verdicts: list[str | None],
) -> None:
    another_check = verdict("failed", check_key=OTHER_CHECK)
    platform.qa_polls = [[row("running"), another_check], [outcome, another_check]]

    result = hud("qa", "run", CHECK, TRACE_ID, "--yes", "--json")

    assert result.exit_code == exit_code, result
    assert [(r["check_key"], r["verdict"]) for r in result.json] == [
        (CHECK, tag) for tag in verdicts
    ]


def test_qa_run_times_out_while_checks_are_running(hud: Hud, platform: Platform) -> None:
    platform.qa_polls = [[row("running")]]

    result = hud("qa", "run", CHECK, TRACE_ID, "--timeout", "1", "--yes", "--json")

    assert (result.exit_code, result.json["error"]) == (1, "timeout"), result


def legacy(check_key: str = CHECK, **output: Any) -> dict[str, Any]:
    """A result stored before ``qa_agent_result.v1``, as its check's own JSON."""
    return row(check_key=check_key, legacy_result={"content": json.dumps(output)})


@pytest.mark.parametrize(
    ("results", "verdicts"),
    [
        pytest.param(
            [{**verdict("failed"), "source": "manual", "verdict": "passed"}],
            ["passed"],
            id="a-manual-verdict-wins",
        ),
        pytest.param(
            [legacy(problems=[{"problem": "No output", "fault": "agent"}])],
            ["failed"],
            id="legacy-agent-fault",
        ),
        pytest.param(
            [
                legacy("false_negative", is_false_negative=False, reasoning="Matches."),
                legacy("false_negative", is_false_negative=True, reasoning="Grader missed it."),
            ],
            ["passed", "failed"],
            id="legacy-false-negative",
        ),
    ],
)
def test_qa_results_read_a_verdict_from_each_result_shape(
    hud: Hud, platform: Platform, results: list[dict[str, Any]], verdicts: list[str]
) -> None:
    platform.qa_results += results

    text = hud("qa", "results", TRACE_ID)

    assert text.exit_code == 0, text
    assert [
        line.removeprefix("Verdict: ") for line in text.lines if line.startswith("Verdict:")
    ] == verdicts


def test_task_results_ask_by_task_for_the_result_in_effect(hud: Hud, platform: Platform) -> None:
    result = hud("qa", "results", "--level", "task", TASK_ID, "--json")

    assert result.exit_code == 0, result
    (asked,) = platform.services.requests("api", "GET", "/v2/qa/results")
    assert asked.query == {"task_ids": [TASK_ID]}
