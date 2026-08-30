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
