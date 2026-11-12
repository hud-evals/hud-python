"""``hud qa run``: run QA checks and exit on their verdicts."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from tests.fixtures.services import Reply

if TYPE_CHECKING:
    from tests.conftest import Hud
    from tests.fixtures.services import FakeServices

TRACE_ID = "00000000-0000-4000-a000-000000000002"
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
        "note": None,
        "error": None,
        **fields,
    }


def verdict(tag: str, *, check_key: str = CHECK) -> dict[str, Any]:
    """A completed ``qa_agent_result.v1`` row."""
    result = {"schema_version": "qa_agent_result.v1", "verdict": tag, "summary": "", "findings": []}
    return row(check_key=check_key, verdict=tag, result=result)


@pytest.mark.parametrize(
    ("outcome", "exit_code", "verdicts"),
    [
        pytest.param(verdict("passed"), 0, ["passed"], id="passed"),
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


def test_a_manual_verdict_wins_over_the_check_result(hud: Hud, catalog: FakeServices) -> None:
    outcome = {**verdict("failed"), "source": "manual", "verdict": "passed"}
    catalog.route("api", "GET", "/v2/qa/results", json={"results": [outcome]})

    result = hud("qa", "run", CHECK, TRACE_ID, "--yes", "--json")

    assert result.exit_code == 0, result
