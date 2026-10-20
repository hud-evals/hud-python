"""``hud qa``: list QA checks, run them on traces or tasks, and read their verdicts."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest

from tests.fixtures.services import Reply

if TYPE_CHECKING:
    from tests.conftest import Hud
    from tests.fixtures.services import FakeServices

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
def catalog(signed_in: FakeServices) -> FakeServices:
    """The platform's QA check catalog, which ``hud qa`` reads to place checks by level."""
    signed_in.route("api", "GET", "/v2/qa/checks", json={"checks": CHECKS})
    signed_in.route("api", "POST", "/v2/qa/runs", json={"results": []})
    return signed_in


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


@pytest.mark.parametrize(
    ("argv", "started"),
    [
        (
            [f"{CHECK}, {OTHER_CHECK},{CHECK}", TRACE_ID.upper()],
            {"check_keys": [CHECK, OTHER_CHECK], "trace_ids": [TRACE_ID], "overwrite": False},
        ),
        (
            [TASK_CHECK, TASK_ID],
            {"check_keys": [TASK_CHECK], "task_ids": [TASK_ID], "overwrite": False},
        ),
    ],
    ids=["deduplicated-and-normalised", "task-level"],
)
def test_qa_run_without_waiting_starts_each_check_once(
    hud: Hud, catalog: FakeServices, argv: list[str], started: dict[str, Any]
) -> None:
    result = hud("qa", "run", *argv, "--no-wait", "--yes", "--json")

    assert result.exit_code == 0, result
    assert catalog.bodies("api", "POST", "/v2/qa/runs") == [started]


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
    catalog: FakeServices,
    outcome: dict[str, Any],
    exit_code: int,
    verdicts: list[str | None],
) -> None:
    another_check = verdict("failed", check_key=OTHER_CHECK)
    catalog.route(
        "api",
        "GET",
        "/v2/qa/results",
        Reply(json={"results": [row("running"), another_check]}),
        Reply(json={"results": [outcome, another_check]}),
    )

    result = hud("qa", "run", CHECK, TRACE_ID, "--yes", "--json")

    assert result.exit_code == exit_code, result
    assert [(r["check_key"], r["verdict"]) for r in result.json] == [
        (CHECK, tag) for tag in verdicts
    ]


def test_qa_run_times_out_while_checks_are_running(hud: Hud, catalog: FakeServices) -> None:
    catalog.route("api", "GET", "/v2/qa/results", json={"results": [row("running")]})

    result = hud("qa", "run", CHECK, TRACE_ID, "--timeout", "1", "--yes", "--json")

    assert (result.exit_code, result.json["error"]) == (1, "timeout"), result


def legacy(**output: Any) -> dict[str, Any]:
    """A result stored before ``qa_agent_result.v1``, as its check's own JSON."""
    return row(legacy_result={"content": json.dumps(output)})


@pytest.mark.parametrize(
    ("outcome", "exit_code"),
    [
        pytest.param(
            {**verdict("failed"), "source": "manual", "verdict": "passed"},
            0,
            id="a-manual-verdict-wins",
        ),
        pytest.param(
            legacy(problems=[{"problem": "No output", "fault": "agent"}]),
            1,
            id="legacy-agent-fault",
        ),
        pytest.param(
            legacy(is_false_negative=False, reasoning="Matches."),
            0,
            id="legacy-false-negative-cleared",
        ),
        pytest.param(
            legacy(is_false_negative=True, reasoning="Grader missed it."),
            1,
            id="legacy-false-negative-found",
        ),
    ],
)
def test_qa_run_reads_a_verdict_from_each_result_shape(
    hud: Hud, catalog: FakeServices, outcome: dict[str, Any], exit_code: int
) -> None:
    catalog.route("api", "GET", "/v2/qa/results", json={"results": [outcome]})

    result = hud("qa", "run", CHECK, TRACE_ID, "--yes", "--json")

    assert result.exit_code == exit_code, result
