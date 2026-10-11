"""CLI behavior for HUD's QA checks on evaluation traces and tasks."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, call, patch

import pytest
from typer.testing import CliRunner

from hud.cli.__main__ import app
from hud.cli.qa import presentation_for_result

runner = CliRunner()

_TRACE_ID = "00000000-0000-4000-a000-000000000002"
_RESULT_ID = "00000000-0000-4000-a000-000000000003"
_ANALYSIS_TRACE_ID = "00000000-0000-4000-a000-000000000004"
_TASK_ID = "00000000-0000-4000-a000-000000000005"
_TASK_VERSION_ID = "00000000-0000-4000-a000-000000000006"
_CHECK = "failure_analysis"
_OTHER_CHECK = "reward_hacking"
_TASK_CHECK = "task_reward_exploit"


def _check(
    key: str = _CHECK, title: str = "Failure Analysis", level: str = "trace"
) -> dict[str, object]:
    return {
        "key": key,
        "subject_type": level,
        "title": title,
        "question": "Why did the agent fail?",
        "description": "Attributes each failure to the agent, the evaluation, or the platform.",
    }


_CATALOG = [
    _check(),
    _check(_OTHER_CHECK, "Reward Hacking"),
    _check(_TASK_CHECK, "Task Reward Exploit", "task"),
]


def _platform(*polls: list[dict[str, Any]]) -> MagicMock:
    """A platform serving the check catalog and one results reply per poll (the last repeats)."""
    platform = MagicMock()
    replies = list(polls)

    def get(path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        if path == "/qa/checks":
            return {"checks": _CATALOG}
        return {"results": replies.pop(0) if len(replies) > 1 else replies[0]}

    platform.get.side_effect = get
    return platform


def _row(
    status: str = "completed",
    *,
    check_key: str = _CHECK,
    verdict: str | None = None,
    result: dict[str, object] | None = None,
    legacy_result: dict[str, object] | None = None,
    **fields: object,
) -> dict[str, Any]:
    return {
        "id": _RESULT_ID,
        "check_key": check_key,
        "subject_type": "trace",
        "subject_trace_id": _TRACE_ID,
        "source": "analysis",
        "status": status,
        "verdict": verdict,
        "result": result,
        "legacy_result": legacy_result,
        "note": None,
        "error": None,
        "created_at": "2026-10-02T00:00:00Z",
        "completed_at": None,
        "run_id": None,
        **fields,
    }


def _result(verdict: str = "passed", *, check_key: str = _CHECK) -> dict[str, Any]:
    return _row(
        check_key=check_key,
        verdict=verdict,
        result={
            "schema_version": "qa_agent_result.v1",
            "verdict": verdict,
            "summary": "Looks good." if verdict == "passed" else "A gap was found.",
            "findings": [],
            "metadata": {},
        },
    )


def _task_result(verdict: str = "passed") -> dict[str, Any]:
    return {
        **_result(verdict, check_key=_TASK_CHECK),
        "subject_type": "task",
        "subject_trace_id": None,
        "subject_task_id": _TASK_ID,
        "subject_task_version_id": _TASK_VERSION_ID,
    }


def _invoke(platform: MagicMock, args: list[str]):
    with patch("hud.cli.qa.PlatformClient.from_settings", return_value=platform):
        return runner.invoke(app, args, env={"TERM": "xterm", "COLUMNS": "200"})


def _table_row(output: str, *cells: str) -> str:
    """The table line holding every given cell."""
    return next(line for line in output.splitlines() if all(cell in line for cell in cells))


@pytest.mark.parametrize("args", [["qa"], ["qa", "list"]])
def test_qa_lists_checks_as_a_table(args: list[str]) -> None:
    platform = MagicMock()
    platform.get.return_value = {"checks": [_check()]}

    result = _invoke(platform, args)

    assert result.exit_code == 0
    assert "QA Checks" in result.output
    assert _table_row(result.output, _CHECK, "trace", "Failure Analysis", "Why did the agent fail?")
    platform.get.assert_called_once_with("/qa/checks")


def test_qa_list_quiet_prints_keys() -> None:
    platform = MagicMock()
    platform.get.return_value = {"checks": [_check(), _check(_OTHER_CHECK, "Reward Hacking")]}

    result = _invoke(platform, ["qa", "list", "--quiet"])

    assert result.exit_code == 0
    assert result.output.split() == [_CHECK, _OTHER_CHECK]


@pytest.mark.parametrize(
    ("level", "keys"), [("task", [_TASK_CHECK]), ("TRACE", [_CHECK, _OTHER_CHECK])]
)
def test_qa_list_filters_by_level(level: str, keys: list[str]) -> None:
    result = _invoke(_platform(), ["qa", "list", "--level", level, "--quiet"])

    assert result.exit_code == 0
    assert result.output.split() == keys


@pytest.mark.parametrize("args", [[_CHECK], [" , ", _TRACE_ID]])
def test_qa_run_requires_checks_and_traces(args: list[str]) -> None:
    platform = MagicMock()

    result = _invoke(platform, ["qa", "run", *args])

    assert result.exit_code == 2
    platform.post.assert_not_called()


def test_qa_run_rejects_non_uuid_traces() -> None:
    platform = MagicMock()

    result = _invoke(platform, ["qa", "run", _CHECK, "not-a-trace"])

    assert result.exit_code == 2
    platform.post.assert_not_called()


def test_qa_run_no_wait_posts_comma_separated_checks_as_one_run() -> None:
    platform = _platform()
    platform.post.return_value = {
        "results": [_row("queued"), _row("queued", check_key=_OTHER_CHECK)]
    }

    result = _invoke(
        platform,
        [
            "qa",
            "run",
            f"{_CHECK}, {_OTHER_CHECK},{_CHECK}",
            _TRACE_ID.upper(),
            "--no-wait",
            "--json",
        ],
    )

    assert result.exit_code == 0
    assert [row["status"] for row in json.loads(result.output)] == ["queued", "queued"]
    platform.post.assert_called_once_with(
        "/qa/runs",
        json={"check_keys": [_CHECK, _OTHER_CHECK], "trace_ids": [_TRACE_ID], "overwrite": False},
    )
    assert platform.get.call_args_list == [call("/qa/checks")]


def test_qa_run_task_checks_take_task_ids_and_wait_on_tasks() -> None:
    platform = _platform([_task_result("failed")])
    platform.post.return_value = {"results": [{**_task_result(), "status": "queued"}]}

    with patch("hud.cli.qa.time.sleep"):
        result = _invoke(platform, ["qa", "run", _TASK_CHECK, _TASK_ID])

    assert result.exit_code == 1
    platform.post.assert_called_once_with(
        "/qa/runs",
        json={"check_keys": [_TASK_CHECK], "task_ids": [_TASK_ID], "overwrite": False},
    )
    platform.get.assert_called_with("/qa/results", params={"task_ids": [_TASK_ID]})
    assert "Task " in result.output
    assert _table_row(result.output, _TASK_ID, _TASK_CHECK, "failed")
    assert "hud qa results --level task <task-id>" in result.output


@pytest.mark.parametrize(
    ("checks", "message"),
    [
        ("missing_check", "Unknown QA check: missing_check"),
        (f"{_CHECK},{_TASK_CHECK}", "Run trace-level and task-level checks separately."),
    ],
)
def test_qa_run_rejects_unknown_or_mixed_level_checks(checks: str, message: str) -> None:
    platform = _platform()

    result = _invoke(platform, ["qa", "run", checks, _TRACE_ID])

    assert result.exit_code == 2
    assert message in result.output
    platform.post.assert_not_called()


def test_qa_run_dry_run_reports_the_level_without_launching() -> None:
    platform = _platform()

    result = _invoke(platform, ["qa", "run", _TASK_CHECK, _TASK_ID, "--dry-run", "--json"])

    assert result.exit_code == 0
    planned = json.loads(result.stdout)
    assert planned["level"] == "task"
    assert planned["task_ids"] == [_TASK_ID]
    platform.post.assert_not_called()


def test_qa_run_forwards_overwrite() -> None:
    platform = _platform()
    platform.post.return_value = {"results": [_result()]}

    result = _invoke(platform, ["qa", "run", _CHECK, _TRACE_ID, "--overwrite"])

    assert result.exit_code == 0
    assert platform.post.call_args.kwargs["json"]["overwrite"] is True


@pytest.mark.parametrize(("verdict", "exit_code"), [("failed", 1), ("passed", 0)])
def test_qa_run_waits_and_scores(verdict: str, exit_code: int) -> None:
    platform = _platform([_row("running")], [_result(verdict)])
    platform.post.return_value = {"results": [_row("queued")]}

    with patch("hud.cli.qa.time.sleep"):
        result = _invoke(platform, ["qa", "run", _CHECK, _TRACE_ID])

    assert result.exit_code == exit_code
    assert _table_row(result.output, _TRACE_ID, _CHECK, verdict)
    platform.get.assert_called_with("/qa/results", params={"trace_ids": [_TRACE_ID]})


def test_qa_run_does_not_poll_settled_results() -> None:
    platform = _platform()
    platform.post.return_value = {"results": [_result("passed")]}

    result = _invoke(platform, ["qa", "run", _CHECK, _TRACE_ID])

    assert result.exit_code == 0
    assert platform.get.call_args_list == [call("/qa/checks")]


def test_qa_run_table_shows_the_summary_headline_as_markdown() -> None:
    row = _result("failed")
    row["result"] = {
        **row["result"],
        "summary": "**Verdict:** The grader missed it.\n\n- The answer was right.",
    }
    platform = _platform()
    platform.post.return_value = {"results": [row]}

    result = _invoke(platform, ["qa", "run", _CHECK, _TRACE_ID])

    assert result.exit_code == 1
    assert _table_row(result.output, _TRACE_ID, _CHECK, "failed", "Verdict: The grader missed it.")
    assert "**" not in result.output
    assert "The answer was right." not in result.output


def test_qa_run_wait_ignores_unrequested_checks() -> None:
    platform = _platform([_result("failed", check_key=_OTHER_CHECK), _result("passed")])
    platform.post.return_value = {"results": [_row("queued")]}

    with patch("hud.cli.qa.time.sleep"):
        result = _invoke(platform, ["qa", "run", _CHECK, _TRACE_ID])

    assert result.exit_code == 0
    assert _OTHER_CHECK not in result.output


def test_qa_run_errored_check_fails() -> None:
    platform = _platform()
    platform.post.return_value = {
        "results": [_row("error", error="The QA check produced an invalid result.")]
    }

    result = _invoke(platform, ["qa", "run", _CHECK, _TRACE_ID])

    assert result.exit_code == 1
    assert "The QA check produced an invalid result." in result.output


def test_qa_run_times_out() -> None:
    platform = _platform([_row("running")])
    platform.post.return_value = {"results": [_row("queued")]}
    clock = [0.0]

    def sleep(seconds: float) -> None:
        clock[0] += seconds

    with (
        patch("hud.cli.qa.time.sleep", side_effect=sleep),
        patch("hud.cli.qa.time.monotonic", side_effect=lambda: clock[0]),
    ):
        result = _invoke(platform, ["qa", "run", _CHECK, _TRACE_ID, "--timeout", "1"])

    assert result.exit_code != 0
    assert "Timed out" in result.output


def test_qa_results_queries_traces() -> None:
    platform = MagicMock()
    platform.get.return_value = {"results": [_result()]}

    result = _invoke(platform, ["qa", "results", _TRACE_ID, "--json"])

    assert result.exit_code == 0
    assert json.loads(result.output)[0]["verdict"] == "passed"
    platform.get.assert_called_once_with("/qa/results", params={"trace_ids": [_TRACE_ID]})


def test_qa_results_reads_task_history_without_a_trace_link() -> None:
    platform = MagicMock()
    platform.get.return_value = {
        "results": [{**_task_result("failed"), "analysis_trace_id": _ANALYSIS_TRACE_ID}]
    }

    result = _invoke(platform, ["qa", "results", "--level", "task", _TASK_ID, "--history"])

    assert result.exit_code == 0
    platform.get.assert_called_once_with(
        "/qa/results", params={"task_ids": [_TASK_ID], "include_history": True}
    )
    assert _TASK_ID in result.output
    assert f"Task version: {_TASK_VERSION_ID}" in result.output
    assert "Verdict: failed" in result.output
    assert "View:" not in result.output
    assert f"Analysis: https://hud.ai/trace/{_ANALYSIS_TRACE_ID}" in result.output


def test_qa_results_tui_renders_findings_and_trace_link() -> None:
    platform = MagicMock()
    platform.get.return_value = {
        "results": [
            _row(
                verdict="failed",
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
        ]
    }

    result = _invoke(platform, ["qa", "results", _TRACE_ID])

    assert result.exit_code == 0
    assert _CHECK in result.output
    assert "Verdict: failed" in result.output
    assert "Confidence: 90%" in result.output
    assert "Verdict: The agent never wrote /app/[regex].txt." in result.output
    assert "**" not in result.output
    assert "1. Required [/output] file was never created" in result.output
    assert "Action: Write the regex to /app/regex.txt." in result.output
    assert "severity: error" in result.output
    assert "evidence: trajectory.json" in result.output
    assert f"View: https://hud.ai/trace/{_TRACE_ID}" in result.output
    assert "Analysis:" not in result.output


def test_qa_results_links_the_analysis_trace_when_returned() -> None:
    platform = MagicMock()
    platform.get.return_value = {
        "results": [{**_result(), "analysis_trace_id": _ANALYSIS_TRACE_ID}]
    }

    result = _invoke(platform, ["qa", "results", _TRACE_ID])

    assert result.exit_code == 0
    assert f"https://hud.ai/trace/{_ANALYSIS_TRACE_ID}" in result.output


def test_qa_results_skipped_check_shows_source_and_note() -> None:
    platform = MagicMock()
    platform.get.return_value = {
        "results": [
            _row(
                verdict="passed",
                source="skipped",
                note="A full reward passes this check without analysis.",
            )
        ]
    }

    result = _invoke(platform, ["qa", "results", _TRACE_ID])

    assert result.exit_code == 0
    assert "Verdict: passed" in result.output
    assert "Source: skipped" in result.output
    assert "A full reward passes this check without analysis." in result.output


def test_qa_results_legacy_failure_analysis() -> None:
    platform = MagicMock()
    platform.get.return_value = {
        "results": [
            _row(
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
        ]
    }

    result = _invoke(platform, ["qa", "results", _TRACE_ID])

    assert result.exit_code == 0
    assert "Verdict: failed" in result.output
    assert "Cause: Agent failure" in result.output
    assert "Required output file was never created" in result.output
    assert "The agent did not save any regex." in result.output


def test_qa_results_legacy_boolean_omits_findings() -> None:
    platform = MagicMock()
    platform.get.return_value = {
        "results": [
            _row(
                check_key="false_negative",
                legacy_result={
                    "content": json.dumps(
                        {
                            "is_false_negative": False,
                            "reasoning": "The zero reward matches the missing file.",
                            "confidence": "high",
                        }
                    ),
                },
            )
        ]
    }

    result = _invoke(platform, ["qa", "results", _TRACE_ID])

    assert result.exit_code == 0
    assert "Verdict: passed" in result.output
    assert "False Negative: no" in result.output
    assert "1. " not in result.output
    assert "The zero reward matches the missing file." in result.output


def test_row_verdict_overrides_the_output_verdict() -> None:
    view = presentation_for_result(
        {
            **_result("failed"),
            "source": "manual",
            "verdict": "passed",
            "note": "Reviewed.",
        }
    )

    assert view.tag == "passed"
    assert view.summary == "A gap was found."


def test_legacy_failure_analysis_problems_are_a_failed_agent_finding() -> None:
    view = presentation_for_result(
        _row(
            legacy_result={
                "content": (
                    '{"summary": "Missing file.", "problems": ['
                    '{"problem": "No regex", "fault": "agent", "description": "Never wrote it."}'
                    '], "confidence": "high"}'
                )
            }
        )
    )

    assert view.kind == "problems"
    assert view.tag == "failed"
    assert view.answer == "Agent failure"
    assert view.findings[0].title == "No regex"
    assert view.findings[0].fault == "agent"


def test_legacy_failure_analysis_empty_problems_is_passed() -> None:
    view = presentation_for_result(
        _row(legacy_result={"summary": "Clean.", "problems": [], "confidence": "high"})
    )

    assert view.tag == "passed"
    assert view.answer == "No failure"
    assert view.findings == ()


def test_legacy_mixed_faults_are_labeled_mixed_failure() -> None:
    view = presentation_for_result(
        _row(
            legacy_result={
                "problems": [
                    {"problem": "Bad regex", "fault": "agent"},
                    {"problem": "Cut off", "fault": "unclear"},
                ]
            }
        )
    )

    assert view.tag == "failed"
    assert view.answer == "Mixed failure"


def test_legacy_false_negative_yes_is_failed_without_findings() -> None:
    view = presentation_for_result(
        _row(
            legacy_result={
                "content": '{"is_false_negative": true, "reasoning": "Grader missed it."}'
            }
        )
    )

    assert view.kind == "boolean"
    assert view.tag == "failed"
    assert view.label == "False Negative"
    assert view.answer == "yes"
    assert view.findings == ()
    assert view.summary == "Grader missed it."


def test_queued_results_are_pending_not_passed() -> None:
    view = presentation_for_result(_row("queued"))

    assert view.kind == "pending"
    assert view.tag == "unknown"
    assert view.label == "queued"
