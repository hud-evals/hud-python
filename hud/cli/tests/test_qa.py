"""CLI behavior for HUD's QA checks on evaluation traces."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from hud.cli.__main__ import app
from hud.cli.qa import presentation_for_result

runner = CliRunner()

_TRACE_ID = "00000000-0000-4000-a000-000000000002"
_RESULT_ID = "00000000-0000-4000-a000-000000000003"
_ANALYSIS_TRACE_ID = "00000000-0000-4000-a000-000000000004"
_CHECK = "failure_analysis"
_OTHER_CHECK = "reward_hacking"


def _check(key: str = _CHECK, title: str = "Failure Analysis") -> dict[str, object]:
    return {
        "key": key,
        "title": title,
        "question": "Why did the agent fail?",
        "description": "Attributes each failure to the agent, the evaluation, or the platform.",
    }


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


def _invoke(platform: MagicMock, args: list[str]):
    with patch("hud.cli.qa.PlatformClient.from_settings", return_value=platform):
        return runner.invoke(app, args)


@pytest.mark.parametrize("args", [["qa"], ["qa", "list"]])
def test_qa_lists_check_title_and_key(args: list[str]) -> None:
    platform = MagicMock()
    platform.get.return_value = {"checks": [_check()]}

    result = _invoke(platform, args)

    assert result.exit_code == 0
    assert result.output.strip() == f"Failure Analysis\t{_CHECK}"
    platform.get.assert_called_once_with("/qa/checks")


def test_qa_list_quiet_prints_keys() -> None:
    platform = MagicMock()
    platform.get.return_value = {"checks": [_check(), _check(_OTHER_CHECK, "Reward Hacking")]}

    result = _invoke(platform, ["qa", "list", "--quiet"])

    assert result.exit_code == 0
    assert result.output.split() == [_CHECK, _OTHER_CHECK]


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
    platform = MagicMock()
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
    platform.get.assert_not_called()


def test_qa_run_forwards_overwrite() -> None:
    platform = MagicMock()
    platform.post.return_value = {"results": [_result()]}

    result = _invoke(platform, ["qa", "run", _CHECK, _TRACE_ID, "--overwrite"])

    assert result.exit_code == 0
    assert platform.post.call_args.kwargs["json"]["overwrite"] is True


@pytest.mark.parametrize(("verdict", "exit_code"), [("failed", 1), ("passed", 0)])
def test_qa_run_waits_and_scores(verdict: str, exit_code: int) -> None:
    platform = MagicMock()
    platform.post.return_value = {"results": [_row("queued")]}
    platform.get.side_effect = [
        {"results": [_row("running")]},
        {"results": [_result(verdict)]},
    ]

    with patch("hud.cli.qa.time.sleep"):
        result = _invoke(platform, ["qa", "run", _CHECK, _TRACE_ID])

    assert result.exit_code == exit_code
    assert f"{_TRACE_ID}\t{_CHECK}\t{verdict}" in result.output
    platform.get.assert_called_with("/qa/results", params={"trace_ids": [_TRACE_ID]})


def test_qa_run_does_not_poll_settled_results() -> None:
    platform = MagicMock()
    platform.post.return_value = {"results": [_result("passed")]}

    result = _invoke(platform, ["qa", "run", _CHECK, _TRACE_ID])

    assert result.exit_code == 0
    platform.get.assert_not_called()


def test_qa_run_wait_ignores_unrequested_checks() -> None:
    platform = MagicMock()
    platform.post.return_value = {"results": [_row("queued")]}
    platform.get.return_value = {
        "results": [_result("failed", check_key=_OTHER_CHECK), _result("passed")]
    }

    with patch("hud.cli.qa.time.sleep"):
        result = _invoke(platform, ["qa", "run", _CHECK, _TRACE_ID])

    assert result.exit_code == 0
    assert _OTHER_CHECK not in result.output


def test_qa_run_errored_check_fails() -> None:
    platform = MagicMock()
    platform.post.return_value = {
        "results": [_row("error", error="The QA check produced an invalid result.")]
    }

    result = _invoke(platform, ["qa", "run", _CHECK, _TRACE_ID])

    assert result.exit_code == 1
    assert "The QA check produced an invalid result." in result.output


def test_qa_run_times_out() -> None:
    platform = MagicMock()
    platform.post.return_value = {"results": [_row("queued")]}
    platform.get.return_value = {"results": [_row("running")]}
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


def test_qa_results_tui_renders_findings_and_trace_link() -> None:
    platform = MagicMock()
    platform.get.return_value = {
        "results": [
            _row(
                verdict="failed",
                result={
                    "schema_version": "qa_agent_result.v1",
                    "verdict": "failed",
                    "summary": "The agent never wrote /app/[regex].txt.",
                    "findings": [{"summary": "Required [/output] file was never created"}],
                    "metadata": {},
                },
            )
        ]
    }

    result = _invoke(platform, ["qa", "results", _TRACE_ID])

    assert result.exit_code == 0
    assert _CHECK in result.output
    assert "verdict: failed" in result.output
    assert "The agent never wrote /app/[regex].txt." in result.output
    assert "Required [/output] file was never created" in result.output
    assert f"https://hud.ai/trace/{_TRACE_ID}" in result.output
    assert "analysis:" not in result.output


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
    assert "verdict: passed" in result.output
    assert "skipped" in result.output
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
    assert "verdict: failed" in result.output
    assert "Agent failure" in result.output
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
    assert "verdict: passed" in result.output
    assert "false negative no" in result.output.lower()
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
