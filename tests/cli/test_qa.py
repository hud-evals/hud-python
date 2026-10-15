"""``hud qa``: list QA checks, run them on traces or tasks, and read their verdicts."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest

from tests.harness import Reply

if TYPE_CHECKING:
    from tests.harness import FakeServices, Hud

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
def catalog(platform: FakeServices) -> None:
    """The platform's QA check catalog, which ``hud qa`` reads to place checks by level."""
    platform.route("api", "GET", "/v2/qa/checks", json={"checks": CHECKS})


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
    ("argv", "keys"),
    [
        (["qa", "--quiet"], [CHECK, OTHER_CHECK, TASK_CHECK]),
        (["qa", "list", "--quiet"], [CHECK, OTHER_CHECK, TASK_CHECK]),
        (["qa", "list", "--level", "task", "--quiet"], [TASK_CHECK]),
        (["qa", "list", "--level", "TRACE", "--quiet"], [CHECK, OTHER_CHECK]),
    ],
)
def test_qa_lists_the_checks_at_a_level(hud: Hud, argv: list[str], keys: list[str]) -> None:
    result = hud(*argv)

    assert (result.exit_code, result.stdout.split()) == (0, keys), result


def test_qa_list_json_is_the_catalog(hud: Hud) -> None:
    result = hud("qa", "list", "--json")

    assert (result.exit_code, result.json) == (0, {"checks": CHECKS}), result


@pytest.mark.parametrize(
    "argv",
    [
        pytest.param([CHECK], id="no-ids"),
        pytest.param([" , ", TRACE_ID], id="no-checks"),
        pytest.param([CHECK, "not-a-trace"], id="an-id-that-is-not-a-uuid"),
        pytest.param(["missing_check", TRACE_ID], id="an-unknown-check"),
        pytest.param([f"{CHECK},{TASK_CHECK}", TRACE_ID], id="trace-and-task-checks-together"),
    ],
)
def test_qa_run_refuses_before_starting_anything(
    hud: Hud, platform: FakeServices, argv: list[str]
) -> None:
    result = hud("qa", "run", *argv, "--json")

    assert (result.exit_code, result.json["error"]) == (2, "usage"), result
    assert platform.requests("api", "POST") == []


@pytest.mark.parametrize(
    ("argv", "body"),
    [
        (
            [f"{CHECK}, {OTHER_CHECK},{CHECK}", TRACE_ID.upper()],
            {"check_keys": [CHECK, OTHER_CHECK], "trace_ids": [TRACE_ID], "overwrite": False},
        ),
        (
            [CHECK, TRACE_ID, "--overwrite"],
            {"check_keys": [CHECK], "trace_ids": [TRACE_ID], "overwrite": True},
        ),
        (
            [TASK_CHECK, TASK_ID],
            {"check_keys": [TASK_CHECK], "task_ids": [TASK_ID], "overwrite": False},
        ),
    ],
    ids=["deduplicated-and-normalised", "overwrite", "task-level"],
)
def test_qa_run_without_waiting_starts_one_run(
    hud: Hud, platform: FakeServices, argv: list[str], body: dict[str, Any]
) -> None:
    queued = [row("queued", check_key=key) for key in body["check_keys"]]
    platform.route("api", "POST", "/v2/qa/runs", json={"results": queued})

    result = hud("qa", "run", *argv, "--no-wait", "--json")

    assert (result.exit_code, result.json) == (0, queued), result
    assert platform.bodies("api", "POST", "/v2/qa/runs") == [body]
    assert platform.requests("api", "GET", "/v2/qa/results") == []


@pytest.mark.parametrize(
    ("argv", "level", "ids"),
    [
        ([f"{CHECK},{OTHER_CHECK}", TRACE_ID], "trace", {"trace_ids": [TRACE_ID]}),
        ([TASK_CHECK, TASK_ID], "task", {"task_ids": [TASK_ID]}),
    ],
)
def test_a_dry_run_plans_the_run_at_its_level_and_starts_nothing(
    hud: Hud, platform: FakeServices, argv: list[str], level: str, ids: dict[str, list[str]]
) -> None:
    result = hud("qa", "run", *argv, "--dry-run", "--json")

    assert result.exit_code == 0, result
    assert result.json["level"] == level
    assert {key: result.json[key] for key in ids} == ids
    assert platform.requests("api", "POST") == []


@pytest.mark.parametrize(
    ("posted", "polled", "exit_code", "verdicts"),
    [
        pytest.param([verdict("passed")], [], 0, ["passed"], id="settled"),
        pytest.param(
            [row("queued")],
            [[row("running")], [verdict("failed")]],
            1,
            ["failed"],
            id="polls-until-done",
        ),
        pytest.param(
            [row("queued")],
            [[verdict("failed", check_key=OTHER_CHECK), verdict("passed")]],
            0,
            ["passed"],
            id="ignores-other-checks",
        ),
        pytest.param([row("error", error="invalid result")], [], 1, [None], id="errored"),
    ],
)
def test_qa_run_waits_for_every_check_and_fails_unless_all_pass(
    hud: Hud,
    platform: FakeServices,
    posted: list[dict[str, Any]],
    polled: list[list[dict[str, Any]]],
    exit_code: int,
    verdicts: list[str | None],
) -> None:
    platform.route("api", "POST", "/v2/qa/runs", json={"results": posted})
    platform.route(
        "api", "GET", "/v2/qa/results", *[Reply(json={"results": rows}) for rows in polled]
    )

    result = hud("qa", "run", CHECK, TRACE_ID, "--json")

    assert result.exit_code == exit_code, result
    assert [(r["check_key"], r["verdict"]) for r in result.json] == [
        (CHECK, tag) for tag in verdicts
    ]
    assert len(platform.requests("api", "GET", "/v2/qa/results")) == len(polled)


def test_qa_run_times_out_while_checks_are_running(hud: Hud, platform: FakeServices) -> None:
    platform.route("api", "POST", "/v2/qa/runs", json={"results": [row("queued")]})
    platform.route("api", "GET", "/v2/qa/results", json={"results": [row("running")]})

    result = hud("qa", "run", CHECK, TRACE_ID, "--timeout", "1", "--json")

    assert (result.exit_code, result.json["error"]) == (1, "timeout"), result


def legacy(check_key: str = CHECK, **output: Any) -> dict[str, Any]:
    """A result stored before ``qa_agent_result.v1``, as its check's own JSON."""
    return row(check_key=check_key, legacy_result={"content": json.dumps(output)})


@pytest.mark.parametrize(
    ("results", "verdicts"),
    [
        pytest.param([verdict("failed")], ["failed"], id="a-verdict"),
        pytest.param(
            [row(verdict="passed", source="skipped", note="Full reward.")], ["passed"], id="skipped"
        ),
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
        pytest.param([legacy(summary="Clean.", problems=[])], ["passed"], id="legacy-no-problems"),
        pytest.param(
            [
                legacy(
                    problems=[
                        {"problem": "Bad regex", "fault": "agent"},
                        {"problem": "Cut off", "fault": "unclear"},
                    ]
                )
            ],
            ["failed"],
            id="legacy-mixed-faults",
        ),
        pytest.param(
            [
                legacy("false_negative", is_false_negative=False, reasoning="Matches."),
                legacy("false_negative", is_false_negative=True, reasoning="Grader missed it."),
            ],
            ["passed", "failed"],
            id="legacy-false-negative",
        ),
        pytest.param(
            [row("queued"), row("error", error="Analysis crashed.")],
            ["failed"],
            id="pending-and-errored",
        ),
    ],
)
def test_qa_results_read_a_verdict_from_each_result_shape(
    hud: Hud, platform: FakeServices, results: list[dict[str, Any]], verdicts: list[str]
) -> None:
    platform.route("api", "GET", "/v2/qa/results", json={"results": results})

    text = hud("qa", "results", TRACE_ID)
    document = hud("qa", "results", TRACE_ID, "--json")

    assert (text.exit_code, document.exit_code) == (0, 0), text
    assert document.json == results
    assert [
        line.removeprefix("Verdict: ") for line in text.lines if line.startswith("Verdict:")
    ] == verdicts
    assert [sent.query for sent in platform.requests("api", "GET", "/v2/qa/results")] == [
        {"trace_ids": [TRACE_ID]}
    ] * 2


def test_task_results_with_history_ask_for_the_task_history(
    hud: Hud, platform: FakeServices
) -> None:
    platform.route("api", "GET", "/v2/qa/results", json={"results": []})

    result = hud("qa", "results", "--level", "task", TASK_ID, "--history", "--json")

    assert (result.exit_code, result.json) == (0, []), result
    (request,) = platform.requests("api", "GET", "/v2/qa/results")
    assert request.query == {"task_ids": [TASK_ID], "include_history": ["True"]}
