"""``hud qa``: list QA checks, run them on traces, and read their results."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest
from inline_snapshot import snapshot

from tests.harness import Reply

if TYPE_CHECKING:
    from tests.harness import FakeServices, Hud

TRACE_ID = "00000000-0000-4000-a000-000000000002"
ANALYSIS_TRACE_ID = "00000000-0000-4000-a000-000000000004"
CHECK = "failure_analysis"
OTHER_CHECK = "reward_hacking"
CHECKS = [
    {
        "key": CHECK,
        "title": "Failure Analysis",
        "question": "Why did the agent fail?",
        "description": "Attributes each failure to the agent, the evaluation, or the platform.",
    },
    {
        "key": OTHER_CHECK,
        "title": "Reward Hacking",
        "question": "Did the agent game the grader?",
        "description": "Looks for answers that satisfy the grader without solving the task.",
    },
]


def row(status: str = "completed", **fields: Any) -> dict[str, Any]:
    """One ``/v2/qa/results`` row as the platform returns it."""
    return {
        "id": "00000000-0000-4000-a000-000000000003",
        "check_key": CHECK,
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


def verdict(tag: str, *, check_key: str = CHECK, summary: str | None = None) -> dict[str, Any]:
    """A completed ``qa_agent_result.v1`` row."""
    return row(
        check_key=check_key,
        verdict=tag,
        result={
            "schema_version": "qa_agent_result.v1",
            "verdict": tag,
            "summary": summary or ("Looks good." if tag == "passed" else "A gap was found."),
            "findings": [],
            "metadata": {},
        },
    )


@pytest.mark.parametrize("argv", [["qa"], ["qa", "list"]])
def test_qa_lists_the_checks(hud: Hud, platform: FakeServices, argv: list[str]) -> None:
    platform.route("api", "GET", "/v2/qa/checks", json={"checks": CHECKS})

    result = hud(*argv)

    assert result.exit_code == 0, result
    assert result.stdout == snapshot("""\
╭───────────╮
│ QA Checks │
╰───────────╯
┏━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Key              ┃ Title            ┃ Question                       ┃
┡━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ failure_analysis │ Failure Analysis │ Why did the agent fail?        │
│ reward_hacking   │ Reward Hacking   │ Did the agent game the grader? │
└──────────────────┴──────────────────┴────────────────────────────────┘

Tip: hud qa run <check>[,<check>...] <trace-id>... to run checks
""")


@pytest.mark.parametrize(
    ("argv", "checks", "stdout"),
    [
        (["qa", "list", "--quiet"], CHECKS, f"{CHECK}\n{OTHER_CHECK}\n"),
        (["qa", "--quiet"], CHECKS, f"{CHECK}\n{OTHER_CHECK}\n"),
        (["qa", "list", "--json"], CHECKS, json.dumps({"checks": CHECKS}, indent=2) + "\n"),
        (["qa", "list"], [], "No QA checks are available.\n"),
    ],
)
def test_qa_list_modes(
    hud: Hud, platform: FakeServices, argv: list[str], checks: list[Any], stdout: str
) -> None:
    platform.route("api", "GET", "/v2/qa/checks", json={"checks": checks})

    result = hud(*argv)

    assert (result.exit_code, result.stdout) == (0, stdout)


@pytest.mark.parametrize(
    ("argv", "document"),
    [
        (
            ["qa", "run", CHECK],
            snapshot({"error": "usage", "message": "Missing parameter: trace_ids"}),
        ),
        (
            ["qa", "run", " , ", TRACE_ID],
            snapshot(
                {
                    "error": "usage",
                    "message": "Name at least one QA check; `hud qa list` shows them.",
                    "input": {"checks": " , "},
                }
            ),
        ),
        (
            ["qa", "run", CHECK, "not-a-trace"],
            snapshot({"error": "usage", "message": "'not-a-trace' is not a valid UUID."}),
        ),
    ],
)
def test_qa_run_needs_checks_and_trace_ids(
    hud: Hud, platform: FakeServices, argv: list[str], document: dict[str, Any]
) -> None:
    result = hud(*argv, "--json")

    assert result.exit_code == 2, result
    assert result.json == document
    assert platform.requests("api", "POST") == []


@pytest.mark.parametrize(
    ("argv", "body"),
    [
        (
            ["qa", "run", f"{CHECK}, {OTHER_CHECK},{CHECK}", TRACE_ID.upper(), "--no-wait"],
            {"check_keys": [CHECK, OTHER_CHECK], "trace_ids": [TRACE_ID], "overwrite": False},
        ),
        (
            ["qa", "run", CHECK, TRACE_ID, "--no-wait", "--overwrite"],
            {"check_keys": [CHECK], "trace_ids": [TRACE_ID], "overwrite": True},
        ),
    ],
)
def test_qa_run_without_waiting_posts_one_run(
    hud: Hud, platform: FakeServices, argv: list[str], body: dict[str, Any]
) -> None:
    queued = [row("queued", check_key=key) for key in body["check_keys"]]
    platform.route("api", "POST", "/v2/qa/runs", json={"results": queued})

    result = hud(*argv, "--json")

    assert result.exit_code == 0, result
    assert result.json == queued
    assert platform.bodies("api", "POST", "/v2/qa/runs") == [body]
    assert platform.requests("api", "GET") == []


def test_qa_run_dry_run_posts_nothing(hud: Hud, platform: FakeServices) -> None:
    result = hud("qa", "run", f"{CHECK},{OTHER_CHECK}", TRACE_ID, "--dry-run", "--json")

    assert result.exit_code == 0, result
    assert result.json == snapshot(
        {
            "dry_run": True,
            "action": "qa_run",
            "check_keys": ["failure_analysis", "reward_hacking"],
            "trace_ids": ["00000000-0000-4000-a000-000000000002"],
            "overwrite": False,
            "wait": True,
        }
    )
    assert platform.requests("api") == []


@pytest.mark.parametrize(
    ("posted", "polled", "exit_code", "stdout"),
    [
        (
            [verdict("passed")],
            [],
            0,
            snapshot("""\
╭────────────╮
│ QA Results │
╰────────────╯
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Trace                                ┃ Check            ┃ Verdict ┃ Summary                  ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ 00000000-0000-4000-a000-000000000002 │ failure_analysis │ passed  │ Looks good.              │
└──────────────────────────────────────┴──────────────────┴─────────┴──────────────────────────┘

Tip: hud qa results <trace-id> for summaries and findings
"""),
        ),
        (
            [row("queued")],
            [[row("running")], [verdict("failed")]],
            1,
            snapshot("""\
╭────────────╮
│ QA Results │
╰────────────╯
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Trace                                ┃ Check            ┃ Verdict ┃ Summary                  ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ 00000000-0000-4000-a000-000000000002 │ failure_analysis │ failed  │ A gap was found.         │
└──────────────────────────────────────┴──────────────────┴─────────┴──────────────────────────┘

Tip: hud qa results <trace-id> for summaries and findings
"""),
        ),
        (
            [row("queued")],
            [[verdict("failed", check_key=OTHER_CHECK), verdict("passed")]],
            0,
            snapshot("""\
╭────────────╮
│ QA Results │
╰────────────╯
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Trace                                ┃ Check            ┃ Verdict ┃ Summary                  ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ 00000000-0000-4000-a000-000000000002 │ failure_analysis │ passed  │ Looks good.              │
└──────────────────────────────────────┴──────────────────┴─────────┴──────────────────────────┘

Tip: hud qa results <trace-id> for summaries and findings
"""),
        ),
        (
            [row("error", error="The QA check produced an invalid result.")],
            [],
            1,
            snapshot("""\
╭────────────╮
│ QA Results │
╰────────────╯
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Trace                                ┃ Check            ┃ Verdict ┃ Summary                  ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ 00000000-0000-4000-a000-000000000002 │ failure_analysis │ failed  │ The QA check produced an │
│                                      │                  │         │ invalid result.          │
└──────────────────────────────────────┴──────────────────┴─────────┴──────────────────────────┘

Tip: hud qa results <trace-id> for summaries and findings
"""),
        ),
        (
            [
                verdict(
                    "failed",
                    summary="**Verdict:** The grader missed it.\n\n- The answer was right.",
                )
            ],
            [],
            1,
            snapshot("""\
╭────────────╮
│ QA Results │
╰────────────╯
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Trace                                ┃ Check            ┃ Verdict ┃ Summary                  ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ 00000000-0000-4000-a000-000000000002 │ failure_analysis │ failed  │ Verdict: The grader      │
│                                      │                  │         │ missed it.               │
└──────────────────────────────────────┴──────────────────┴─────────┴──────────────────────────┘

Tip: hud qa results <trace-id> for summaries and findings
"""),
        ),
    ],
)
def test_qa_run_waits_for_every_check_and_fails_unless_all_pass(
    hud: Hud,
    platform: FakeServices,
    posted: list[dict[str, Any]],
    polled: list[list[dict[str, Any]]],
    exit_code: int,
    stdout: str,
) -> None:
    platform.route("api", "POST", "/v2/qa/runs", json={"results": posted})
    platform.route(
        "api", "GET", "/v2/qa/results", *[Reply(json={"results": rows}) for rows in polled]
    )

    result = hud("qa", "run", CHECK, TRACE_ID)

    assert result.exit_code == exit_code, result
    assert result.stdout == stdout
    assert [request.query for request in platform.requests("api", "GET")] == [
        {"trace_ids": [TRACE_ID]}
    ] * len(polled)


def test_qa_run_times_out_while_checks_are_running(hud: Hud, platform: FakeServices) -> None:
    platform.route("api", "POST", "/v2/qa/runs", json={"results": [row("queued")]})
    platform.route("api", "GET", "/v2/qa/results", json={"results": [row("running")]})

    result = hud("qa", "run", CHECK, TRACE_ID, "--timeout", "1", "--json")

    assert result.exit_code == 1, result
    assert result.json == snapshot(
        {
            "error": "timeout",
            "message": "Timed out after 1s waiting for QA checks.",
            "suggestion": "Retry; the failure may be transient. Increase --timeout if set.",
        }
    )


@pytest.mark.parametrize(
    "results",
    [
        pytest.param(
            [
                row(
                    verdict="failed",
                    analysis_trace_id=ANALYSIS_TRACE_ID,
                    result={
                        "schema_version": "qa_agent_result.v1",
                        "verdict": "failed",
                        "summary": "**Verdict:** The agent never wrote /app/[regex].txt.",
                        "findings": [
                            {
                                "finding_type": "missing_output",
                                "severity": "error",
                                "summary": "Required [/output] file was never created",
                                "recommended_action": "Write the regex to /app/regex.txt.",
                                "evidence_refs": ["trajectory.json"],
                            }
                        ],
                        "metadata": {"confidence": 0.9},
                    },
                )
            ],
            id="findings",
        ),
        pytest.param(
            [
                row(
                    verdict="passed",
                    source="skipped",
                    note="A full reward passes this check without analysis.",
                )
            ],
            id="skipped",
        ),
        pytest.param(
            [{**verdict("failed"), "source": "manual", "verdict": "passed", "note": "Reviewed."}],
            id="manual-verdict-wins",
        ),
        pytest.param(
            [
                row(
                    legacy_result={
                        "content": json.dumps(
                            {
                                "summary": "The agent never wrote /app/regex.txt.",
                                "confidence": "high",
                                "problems": [
                                    {
                                        "problem": "Required output file was never created",
                                        "description": "The agent did not save any regex.",
                                        "fault": "agent",
                                    }
                                ],
                            }
                        ),
                        "reward": 0.0,
                    }
                )
            ],
            id="legacy-failure-analysis",
        ),
        pytest.param(
            [row(legacy_result={"summary": "Clean.", "problems": [], "confidence": "high"})],
            id="legacy-no-problems",
        ),
        pytest.param(
            [
                row(
                    legacy_result={
                        "problems": [
                            {"problem": "Bad regex", "fault": "agent"},
                            {"problem": "Cut off", "fault": "unclear"},
                        ]
                    }
                )
            ],
            id="legacy-mixed-faults",
        ),
        pytest.param(
            [
                row(
                    check_key="false_negative",
                    legacy_result={
                        "content": json.dumps(
                            {
                                "is_false_negative": False,
                                "reasoning": "The zero reward matches the missing file.",
                                "confidence": "high",
                            }
                        )
                    },
                ),
                row(
                    check_key="false_negative",
                    legacy_result={
                        "content": '{"is_false_negative": true, "reasoning": "Grader missed it."}'
                    },
                ),
            ],
            id="legacy-boolean",
        ),
        pytest.param(
            [row("queued"), row("error", error="Analysis crashed.")],
            id="pending-and-errored",
        ),
    ],
)
def test_qa_results_render_each_result_shape(
    hud: Hud, platform: FakeServices, request: pytest.FixtureRequest, results: list[dict[str, Any]]
) -> None:
    platform.route("api", "GET", "/v2/qa/results", json={"results": results})

    text = hud("qa", "results", TRACE_ID)
    document = hud("qa", "results", TRACE_ID, "--json")

    assert text.exit_code == document.exit_code == 0
    assert document.json == results
    assert text.stdout == RENDERED[request.node.callspec.id]
    assert [sent.query for sent in platform.requests("api", "GET")] == [
        {"trace_ids": [TRACE_ID]}
    ] * 2


RENDERED = snapshot(
    {
        "findings": """\
╭───────────────────────────────────────────────────────╮
│ failure_analysis 00000000-0000-4000-a000-000000000002 │
╰───────────────────────────────────────────────────────╯
Verdict: failed
Confidence: 90%
╭─ Summary ────────────────────────────────────────────────────────────────────────────────────╮
│ Verdict: The agent never wrote /app/[regex].txt.                                             │
╰──────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ 1. Required [/output] file was never created ───────────────────────────────────────────────╮
│ Action: Write the regex to /app/regex.txt.                                                   │
│ severity: error                                                                              │
│ evidence: trajectory.json                                                                    │
╰──────────────────────────────────────────────────────────────────────────────────────────────╯

View: https://hud.example/trace/00000000-0000-4000-a000-000000000002
Analysis: https://hud.example/trace/00000000-0000-4000-a000-000000000004
""",
        "skipped": """\
╭───────────────────────────────────────────────────────╮
│ failure_analysis 00000000-0000-4000-a000-000000000002 │
╰───────────────────────────────────────────────────────╯
Verdict: passed
Source: skipped
Note: A full reward passes this check without analysis.

View: https://hud.example/trace/00000000-0000-4000-a000-000000000002
""",
        "manual-verdict-wins": """\
╭───────────────────────────────────────────────────────╮
│ failure_analysis 00000000-0000-4000-a000-000000000002 │
╰───────────────────────────────────────────────────────╯
Verdict: passed
Source: manual
Note: Reviewed.
╭─ Summary ────────────────────────────────────────────────────────────────────────────────────╮
│ A gap was found.                                                                             │
╰──────────────────────────────────────────────────────────────────────────────────────────────╯

View: https://hud.example/trace/00000000-0000-4000-a000-000000000002
""",
        "legacy-failure-analysis": """\
╭───────────────────────────────────────────────────────╮
│ failure_analysis 00000000-0000-4000-a000-000000000002 │
╰───────────────────────────────────────────────────────╯
Verdict: failed
Cause: Agent failure
Confidence: high
╭─ Summary ────────────────────────────────────────────────────────────────────────────────────╮
│ The agent never wrote /app/regex.txt.                                                        │
╰──────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ 1. Required output file was never created ──────────────────────────────────────────────────╮
│ The agent did not save any regex.                                                            │
│ fault: agent                                                                                 │
╰──────────────────────────────────────────────────────────────────────────────────────────────╯

View: https://hud.example/trace/00000000-0000-4000-a000-000000000002
""",
        "legacy-no-problems": """\
╭───────────────────────────────────────────────────────╮
│ failure_analysis 00000000-0000-4000-a000-000000000002 │
╰───────────────────────────────────────────────────────╯
Verdict: passed
Cause: No failure
Confidence: high
╭─ Summary ────────────────────────────────────────────────────────────────────────────────────╮
│ Clean.                                                                                       │
╰──────────────────────────────────────────────────────────────────────────────────────────────╯

View: https://hud.example/trace/00000000-0000-4000-a000-000000000002
""",
        "legacy-mixed-faults": """\
╭───────────────────────────────────────────────────────╮
│ failure_analysis 00000000-0000-4000-a000-000000000002 │
╰───────────────────────────────────────────────────────╯
Verdict: failed
Cause: Mixed failure
╭─ 1. Bad regex ───────────────────────────────────────────────────────────────────────────────╮
│ fault: agent                                                                                 │
╰──────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ 2. Cut off ─────────────────────────────────────────────────────────────────────────────────╮
│ fault: unclear                                                                               │
╰──────────────────────────────────────────────────────────────────────────────────────────────╯

View: https://hud.example/trace/00000000-0000-4000-a000-000000000002
""",
        "legacy-boolean": """\
╭─────────────────────────────────────────────────────╮
│ false_negative 00000000-0000-4000-a000-000000000002 │
╰─────────────────────────────────────────────────────╯
Verdict: passed
False Negative: no
Confidence: high
╭─ Summary ────────────────────────────────────────────────────────────────────────────────────╮
│ The zero reward matches the missing file.                                                    │
╰──────────────────────────────────────────────────────────────────────────────────────────────╯

View: https://hud.example/trace/00000000-0000-4000-a000-000000000002

╭─────────────────────────────────────────────────────╮
│ false_negative 00000000-0000-4000-a000-000000000002 │
╰─────────────────────────────────────────────────────╯
Verdict: failed
False Negative: yes
╭─ Summary ────────────────────────────────────────────────────────────────────────────────────╮
│ Grader missed it.                                                                            │
╰──────────────────────────────────────────────────────────────────────────────────────────────╯

View: https://hud.example/trace/00000000-0000-4000-a000-000000000002
""",
        "pending-and-errored": """\
╭───────────────────────────────────────────────────────╮
│ failure_analysis 00000000-0000-4000-a000-000000000002 │
╰───────────────────────────────────────────────────────╯
Status: queued

View: https://hud.example/trace/00000000-0000-4000-a000-000000000002

╭───────────────────────────────────────────────────────╮
│ failure_analysis 00000000-0000-4000-a000-000000000002 │
╰───────────────────────────────────────────────────────╯
Verdict: failed
╭─ Summary ────────────────────────────────────────────────────────────────────────────────────╮
│ Analysis crashed.                                                                            │
╰──────────────────────────────────────────────────────────────────────────────────────────────╯

View: https://hud.example/trace/00000000-0000-4000-a000-000000000002
""",
    }
)


def test_qa_results_without_results_says_so(hud: Hud, platform: FakeServices) -> None:
    platform.route("api", "GET", "/v2/qa/results", json={"results": []})

    result = hud("qa", "results", TRACE_ID)

    assert (result.exit_code, result.stdout) == (0, "No QA results found.\n")
