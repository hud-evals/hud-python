"""CLI behavior for trace-level platform QA agents."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from hud.cli import app

runner = CliRunner()

_AGENT_ID = "00000000-0000-4000-a000-000000000001"
_TRACE_ID = "00000000-0000-4000-a000-000000000002"
_RESULT_ID = "00000000-0000-4000-a000-000000000003"
_OTHER_AGENT_ID = "00000000-0000-4000-a000-000000000004"


def _agent(*, subject_type: str = "trace") -> dict[str, object]:
    return {
        "id": _AGENT_ID,
        "name": "Failure Analysis",
        "subject_type": subject_type,
        "model_name": "claude-sonnet",
    }


def _run(status: str = "queued") -> dict[str, object]:
    return {
        "id": _RESULT_ID,
        "qa_agent_id": _AGENT_ID,
        "subject_type": "trace",
        "subject_id": _TRACE_ID,
        "subject_trace_id": _TRACE_ID,
        "status": status,
    }


def _result(verdict: str = "passed") -> dict[str, object]:
    return {
        **_run(status="completed"),
        "agent_name": "Failure Analysis",
        "canonical_result": {
            "schema_version": "qa_agent_result.v1",
            "verdict": verdict,
            "summary": "Looks good." if verdict == "passed" else "A gap was found.",
            "findings": [],
            "metadata": {},
        },
        "error": None,
        "stale": False,
    }


def _invoke(platform: MagicMock, args: list[str]):
    with (
        patch("hud.cli.qa.require_api_key", return_value="api-key"),
        patch("hud.cli.qa.PlatformClient.from_settings", return_value=platform),
    ):
        return runner.invoke(app, args)


def test_qa_lists_agent_name_and_uuid() -> None:
    platform = MagicMock()
    platform.get.return_value = {
        "items": [_agent()],
        "total": 1,
        "limit": 50,
        "offset": 0,
    }

    result = _invoke(platform, ["qa"])

    assert result.exit_code == 0
    assert result.output.strip() == f"Failure Analysis\t{_AGENT_ID}"
    platform.get.assert_called_once_with(
        "/qa-agents",
        params={"subject_type": "trace", "limit": 50, "offset": 0},
    )


def test_qa_run_rejects_resource_agents() -> None:
    platform = MagicMock()
    platform.get.return_value = _agent(subject_type="environment")

    result = _invoke(platform, ["qa", "run", _AGENT_ID, _TRACE_ID, "--no-wait"])

    assert result.exit_code == 1
    assert "trace agents only" in result.output
    platform.post.assert_not_called()


def test_qa_run_no_wait_uses_trace_endpoint() -> None:
    platform = MagicMock()
    platform.get.return_value = _agent()
    platform.post.return_value = [
        {
            **_run(status="completed"),
            "result": {
                "schema_version": "qa_agent_result.v1",
                "verdict": "failed",
                "summary": "A reused failure.",
            },
        },
    ]

    result = _invoke(platform, ["qa", "run", _AGENT_ID, _TRACE_ID, "--no-wait", "--json"])

    assert result.exit_code == 0
    assert json.loads(result.output)[0]["result"]["verdict"] == "failed"
    platform.post.assert_called_once_with(
        f"/qa-agents/{_AGENT_ID}/run",
        json={"trace_ids": [_TRACE_ID], "overwrite": False},
    )


@pytest.mark.parametrize(("verdict", "exit_code"), [("failed", 1), ("passed", 0)])
def test_qa_run_waits_and_scores(verdict: str, exit_code: int) -> None:
    platform = MagicMock()
    platform.post.return_value = [_run()]
    platform.get.side_effect = [_agent(), [], [_result(verdict)]]

    with patch("hud.cli.qa.time.sleep"):
        result = _invoke(platform, ["qa", "run", _AGENT_ID, _TRACE_ID])

    assert result.exit_code == exit_code
    assert verdict in result.output
    platform.get.assert_called_with(
        "/qa-agents/results",
        params={"subject_trace_ids": [_TRACE_ID]},
    )


def test_qa_run_wait_scores_launched_ids_not_older_rows() -> None:
    older = {**_result("failed"), "id": "00000000-0000-4000-a000-000000000099"}
    launched = _run()
    completed = _result("passed")
    platform = MagicMock()
    platform.post.return_value = [launched]
    platform.get.side_effect = [_agent(), [older], [older, completed]]

    with patch("hud.cli.qa.time.sleep"):
        result = _invoke(platform, ["qa", "run", _AGENT_ID, _TRACE_ID])

    assert result.exit_code == 0
    assert "passed" in result.output
    assert "failed" not in result.output


def test_qa_run_wait_reuses_latest_when_launch_returns_empty() -> None:
    older = {**_result("failed"), "id": "00000000-0000-4000-a000-000000000099"}
    newer = _result("passed")
    platform = MagicMock()
    platform.post.return_value = []
    platform.get.side_effect = [_agent(), [older, newer]]

    result = _invoke(platform, ["qa", "run", _AGENT_ID, _TRACE_ID])

    assert result.exit_code == 0
    assert "passed" in result.output


def test_qa_run_waits_for_trace_results_and_ignores_other_agents() -> None:
    platform = MagicMock()
    platform.post.return_value = []
    platform.get.side_effect = [
        _agent(),
        [{**_result("failed"), "qa_agent_id": _OTHER_AGENT_ID}, _result("passed")],
    ]

    result = _invoke(platform, ["qa", "run", _AGENT_ID, _TRACE_ID])

    assert result.exit_code == 0
    assert "passed" in result.output


def test_qa_results_queries_traces() -> None:
    platform = MagicMock()
    platform.get.return_value = [_result()]

    result = _invoke(platform, ["qa", "results", _TRACE_ID, "--json"])

    assert result.exit_code == 0
    assert json.loads(result.output)[0]["canonical_result"]["verdict"] == "passed"
    platform.get.assert_called_once_with(
        "/qa-agents/results",
        params={"subject_trace_ids": [_TRACE_ID]},
    )


def test_qa_results_default_tui_hides_trajectory() -> None:
    platform = MagicMock()
    platform.get.return_value = [
        {
            **_run(status="completed"),
            "agent_name": "Failure Analysis",
            "result": {
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
            },
        }
    ]

    result = _invoke(platform, ["qa", "results", _TRACE_ID])

    assert result.exit_code == 0
    assert "verdict: failed" in result.output
    assert "verdict: completed" not in result.output
    assert "Agent failure" in result.output
    assert "The agent never wrote /app/regex.txt." in result.output
    assert "Required output file was never created" in result.output
    assert "The agent did not save any regex." in result.output
    assert "pass --rollout to show" in result.output
    assert "shell" not in result.output
    assert "ls /workspace" not in result.output
    assert f"https://hud.ai/trace/{_TRACE_ID}" in result.output
    platform.get.assert_called_once_with(
        "/qa-agents/results",
        params={"subject_trace_ids": [_TRACE_ID]},
    )


def test_qa_results_tui_preserves_bracketed_titles() -> None:
    platform = MagicMock()
    platform.get.side_effect = [
        [
            {
                **_run(status="completed"),
                "agent_name": "Failure Analysis",
                "result": {
                    "content": json.dumps(
                        {
                            "summary": "The agent never wrote /app/regex.txt.",
                            "problems": [
                                {
                                    "problem": "Required [/output] file was never created",
                                    "description": "Save the regex at /app/[regex].txt.",
                                }
                            ],
                        }
                    ),
                },
            }
        ],
        {
            "events": [
                {
                    "kind": "tool_call",
                    "tool_name": "shell[cmd]",
                    "arguments": {"commands": ["ls /workspace"]},
                },
                {
                    "kind": "subagent",
                    "agent_name": "checker[v1]",
                    "arguments": {},
                },
            ],
            "has_more": False,
            "next_seq": 2,
            "status": "completed",
        },
    ]

    result = _invoke(platform, ["qa", "results", _TRACE_ID, "--rollout"])

    assert result.exit_code == 0
    assert "Required [/output] file was never created" in result.output
    assert "Save the regex at /app/[regex].txt." in result.output
    assert "shell[cmd]" in result.output
    assert "checker[v1]" in result.output


def test_qa_results_renders_sanitized_rollout() -> None:
    platform = MagicMock()
    platform.get.side_effect = [
        [_result()],
        {
            "events": [
                {"kind": "agent_message", "text": "Checking the workspace."},
                {"kind": "agent_message", "text": ""},
                {
                    "kind": "agent_message",
                    "text": json.dumps({"summary": "Looks good.", "problems": []}),
                },
                {
                    "kind": "tool_call",
                    "tool_name": "shell",
                    "arguments": {"commands": ["ls /workspace"]},
                    "result_text": "task.py",
                },
            ],
            "has_more": False,
            "next_seq": 2,
            "latest_seq": 1,
            "status": "completed",
        },
    ]

    result = _invoke(platform, ["qa", "results", _TRACE_ID, "--rollout"])

    assert result.exit_code == 0
    assert "Failure Analysis" in result.output
    assert "Looks good." in result.output
    assert "Checking the workspace." in result.output
    assert "shell" in result.output
    assert "ls /workspace" in result.output
    assert "task.py" in result.output
    assert result.output.count("Turn 1") == 1
    assert "pass --rollout to show" not in result.output
    assert f"https://hud.ai/trace/{_TRACE_ID}" in result.output
    assert platform.get.call_args_list[0].args[0] == "/qa-agents/results"
    assert platform.get.call_args_list[1].args == (f"/qa-agents/results/{_RESULT_ID}/rollout",)
    assert platform.get.call_args_list[1].kwargs["params"] == {
        "since_seq": -1,
        "limit": 100,
    }


def test_qa_results_pages_sanitized_rollout() -> None:
    platform = MagicMock()
    platform.get.side_effect = [
        [_result()],
        {
            "events": [{"kind": "tool_call", "tool_name": "shell", "arguments": {}}],
            "has_more": True,
            "next_seq": 10,
            "status": "completed",
        },
        {
            "events": [
                {
                    "kind": "tool_call",
                    "tool_name": "verify_failure_claims",
                    "arguments": {"claims": "The file was missing."},
                }
            ],
            "has_more": False,
            "next_seq": 11,
            "status": "completed",
        },
    ]

    result = _invoke(platform, ["qa", "results", _TRACE_ID, "--rollout"])

    assert result.exit_code == 0
    assert "shell" in result.output
    assert "verify_failure_claims" in result.output
    assert "The file was missing." in result.output
    assert platform.get.call_count == 3


def test_qa_results_empty_problems_is_passed() -> None:
    platform = MagicMock()
    platform.get.return_value = [
        {
            **_run(status="completed"),
            "agent_name": "Failure Analysis",
            "result": {
                "content": json.dumps(
                    {
                        "summary": "The agent completed the task.",
                        "problems": [],
                        "confidence": "high",
                    }
                ),
            },
        }
    ]

    result = _invoke(platform, ["qa", "results", _TRACE_ID])

    assert result.exit_code == 0
    assert "verdict: passed" in result.output
    assert "No failure" in result.output
    assert "1. " not in result.output


def test_qa_results_boolean_agent_omits_findings() -> None:
    platform = MagicMock()
    platform.get.return_value = [
        {
            **_run(status="completed"),
            "agent_name": "False Negative Analysis",
            "result": {
                "content": json.dumps(
                    {
                        "is_false_negative": False,
                        "reasoning": "The zero reward matches the missing file.",
                        "confidence": "high",
                    }
                ),
                "reward": 1.0,
            },
        }
    ]

    result = _invoke(platform, ["qa", "results", _TRACE_ID])

    assert result.exit_code == 0
    assert "False Negative Analysis" in result.output
    assert "verdict: passed" in result.output
    assert "false negative no" in result.output.lower()
    assert "verdict: completed" not in result.output
    assert "1. " not in result.output
    assert "fault:" not in result.output
    assert "The zero reward matches the missing file." in result.output


def test_qa_results_json_normalizes_boolean_blob_to_v1() -> None:
    platform = MagicMock()
    platform.get.return_value = [
        {
            **_run(status="completed"),
            "agent_name": "False Negative Analysis",
            "result": {
                "is_false_negative": True,
                "reasoning": "The grader rejected a valid answer.",
            },
        }
    ]

    result = _invoke(platform, ["qa", "results", _TRACE_ID, "--json"])

    assert result.exit_code == 0
    payload = json.loads(result.output)[0]
    assert payload["result"]["is_false_negative"] is True
    assert payload["canonical_result"] == {
        "schema_version": "qa_agent_result.v1",
        "verdict": "failed",
        "summary": "The grader rejected a valid answer.",
        "findings": [],
        "metadata": {
            "kind": "boolean",
            "label": "False Negative",
            "is_false_negative": True,
            "answer": "yes",
        },
    }


def test_qa_results_false_negative_yes_is_failed() -> None:
    platform = MagicMock()
    platform.get.return_value = [
        {
            **_run(status="completed"),
            "agent_name": "False Negative Analysis",
            "result": {
                "is_false_negative": True,
                "reasoning": "The grader rejected a valid answer.",
            },
        }
    ]

    result = _invoke(platform, ["qa", "results", _TRACE_ID])

    assert result.exit_code == 0
    assert "verdict: failed" in result.output
    assert "false negative yes" in result.output.lower()
    assert "1. " not in result.output
    assert "The grader rejected a valid answer." in result.output


def test_qa_results_rollout_skips_boolean_result_json() -> None:
    blob = json.dumps(
        {"is_false_negative": False, "reasoning": "The zero reward matches the missing file."}
    )
    platform = MagicMock()
    platform.get.side_effect = [
        [
            {
                **_run(status="completed"),
                "agent_name": "False Negative Analysis",
                "result": {"content": blob},
            }
        ],
        {
            "events": [
                {"kind": "agent_message", "text": "Checking the grader."},
                {"kind": "agent_message", "text": blob},
            ],
            "has_more": False,
            "next_seq": 2,
            "status": "completed",
        },
    ]

    result = _invoke(platform, ["qa", "results", _TRACE_ID, "--rollout"])

    assert result.exit_code == 0
    assert "Checking the grader." in result.output
    assert result.output.count("Turn 1") == 1
    assert '"is_false_negative"' not in result.output
