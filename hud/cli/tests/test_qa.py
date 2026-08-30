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
_SCENARIO_ID = "00000000-0000-4000-a000-000000000005"
_MODEL_ID = "00000000-0000-4000-a000-000000000006"


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


def test_qa_agents_lists_trace_scope() -> None:
    platform = MagicMock()
    platform.get.return_value = {
        "items": [_agent()],
        "total": 1,
        "limit": 50,
        "offset": 0,
    }

    result = _invoke(platform, ["qa", "agents"])

    assert result.exit_code == 0
    assert _AGENT_ID in result.output
    platform.get.assert_called_once_with(
        "/qa-agents",
        params={"subject_type": "trace", "limit": 50, "offset": 0},
    )


def test_qa_scenarios_lists_trace_candidates() -> None:
    platform = MagicMock()
    platform.get.return_value = [
        {"id": _SCENARIO_ID, "name": "qa:failure-analysis", "registry_display_name": "Trace QA"},
    ]

    result = _invoke(platform, ["qa", "scenarios"])

    assert result.exit_code == 0
    assert _SCENARIO_ID in result.output
    platform.get.assert_called_once_with(
        "/qa-agents/scenarios",
        params={"subject_type": "trace", "limit": 500},
    )


def test_qa_create_posts_a_trace_agent() -> None:
    platform = MagicMock()
    platform.post.return_value = _agent()

    result = _invoke(
        platform,
        [
            "qa",
            "create",
            "--name",
            "Failure Analysis",
            "--scenario",
            _SCENARIO_ID,
            "--model",
            _MODEL_ID,
            "--args",
            '{"threshold": 0.5}',
            "--json",
        ],
    )

    assert result.exit_code == 0
    assert json.loads(result.output)["id"] == _AGENT_ID
    platform.post.assert_called_once_with(
        "/qa-agents",
        json={
            "name": "Failure Analysis",
            "scenario_id": _SCENARIO_ID,
            "model_id": _MODEL_ID,
            "subject_type": "trace",
            "partial_args": {"threshold": 0.5},
        },
    )


def test_qa_create_rejects_invalid_args() -> None:
    platform = MagicMock()

    result = _invoke(
        platform,
        [
            "qa",
            "create",
            "--name",
            "Failure Analysis",
            "--scenario",
            _SCENARIO_ID,
            "--model",
            _MODEL_ID,
            "--args",
            "not-json",
        ],
    )

    assert result.exit_code == 1
    assert "--args must be a JSON object" in result.output
    platform.post.assert_not_called()


def test_qa_update_patches_trace_agent() -> None:
    platform = MagicMock()
    platform.get.return_value = _agent()
    platform.patch.return_value = {**_agent(), "name": "Renamed"}

    result = _invoke(platform, ["qa", "update", _AGENT_ID, "--name", "Renamed", "--private"])

    assert result.exit_code == 0
    assert "Renamed" in result.output
    platform.patch.assert_called_once_with(
        f"/qa-agents/{_AGENT_ID}",
        json={"name": "Renamed", "public": False},
    )


@pytest.mark.parametrize(
    ("args", "method"),
    [
        (["qa", "update", _AGENT_ID, "--name", "Nope"], "patch"),
        (["qa", "delete", _AGENT_ID], "delete"),
        (["qa", "run", _AGENT_ID, _TRACE_ID, "--no-wait"], "post"),
    ],
)
def test_qa_commands_reject_resource_agents(args: list[str], method: str) -> None:
    platform = MagicMock()
    platform.get.return_value = _agent(subject_type="environment")

    result = _invoke(platform, args)

    assert result.exit_code == 1
    assert "trace agents only" in result.output
    getattr(platform, method).assert_not_called()


def test_qa_delete_removes_trace_agent() -> None:
    platform = MagicMock()
    platform.get.return_value = _agent()
    platform.delete.return_value = {}

    result = _invoke(platform, ["qa", "delete", _AGENT_ID])

    assert result.exit_code == 0
    assert f"Deleted {_AGENT_ID}" in result.output
    platform.delete.assert_called_once_with(f"/qa-agents/{_AGENT_ID}")


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
