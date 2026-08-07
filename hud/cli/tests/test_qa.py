"""CLI behavior for resource-scoped platform QA agents."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from hud.cli import app

runner = CliRunner()

_AGENT_ID = "00000000-0000-4000-a000-000000000001"
_SUBJECT_ID = "00000000-0000-4000-a000-000000000002"
_RESULT_ID = "00000000-0000-4000-a000-000000000003"


def _run(status: str = "queued") -> dict[str, object]:
    return {
        "id": _RESULT_ID,
        "qa_agent_id": _AGENT_ID,
        "subject_type": "task",
        "subject_id": _SUBJECT_ID,
        "status": status,
    }


def _result(verdict: str = "passed") -> dict[str, object]:
    return {
        **_run(status="completed"),
        "agent_name": "Task Contract Auditor",
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


def test_qa_agents_lists_requested_resource_scope() -> None:
    platform = MagicMock()
    platform.get.return_value = {
        "items": [
            {
                "id": _AGENT_ID,
                "name": "Task Contract Auditor",
                "subject_type": "task",
                "model_name": "claude-sonnet",
            },
        ],
        "total": 1,
        "limit": 50,
        "offset": 0,
    }

    with (
        patch("hud.cli.qa.require_api_key", return_value="api-key"),
        patch("hud.cli.qa.PlatformClient.from_settings", return_value=platform),
    ):
        result = runner.invoke(app, ["qa", "agents", "--subject-type", "task"])

    assert result.exit_code == 0
    assert _AGENT_ID in result.output
    assert "Task Contract Auditor" in result.output
    platform.get.assert_called_once_with(
        "/qa-agents",
        params={"subject_type": "task", "limit": 50, "offset": 0},
    )


def test_qa_subject_type_uses_typer_validation() -> None:
    with patch("hud.cli.qa.require_api_key") as require_api_key:
        result = runner.invoke(app, ["qa", "agents", "--subject-type", "trace"])

    assert result.exit_code == 2
    assert "Invalid value" in result.output
    require_api_key.assert_not_called()


def test_qa_run_no_wait_returns_the_launch_response_without_scoring_it() -> None:
    platform = MagicMock()
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

    with (
        patch("hud.cli.qa.require_api_key", return_value="api-key"),
        patch("hud.cli.qa.PlatformClient.from_settings", return_value=platform),
    ):
        result = runner.invoke(
            app,
            ["qa", "run", _AGENT_ID, _SUBJECT_ID, "--no-wait", "--json"],
        )

    assert result.exit_code == 0
    assert json.loads(result.output)[0]["result"]["verdict"] == "failed"
    platform.post.assert_called_once_with(
        f"/qa-agents/{_AGENT_ID}/run-resources",
        json={"subject_ids": [_SUBJECT_ID], "overwrite": False},
    )
    platform.get.assert_not_called()


def test_qa_run_waits_for_exact_result_ids_and_returns_quality_exit() -> None:
    platform = MagicMock()
    platform.post.return_value = [_run()]
    platform.get.side_effect = [[], [_result("failed")]]

    with (
        patch("hud.cli.qa.require_api_key", return_value="api-key"),
        patch("hud.cli.qa.PlatformClient.from_settings", return_value=platform),
        patch("hud.cli.qa.time.sleep"),
    ):
        result = runner.invoke(app, ["qa", "run", _AGENT_ID, _SUBJECT_ID])

    assert result.exit_code == 1
    assert "failed" in result.output
    assert "A gap was found." in result.output
    assert platform.get.call_count == 2
    platform.get.assert_called_with(
        "/qa-agents/results/resources",
        params={"result_ids": [_RESULT_ID]},
    )


def test_qa_run_returns_zero_for_passed_results() -> None:
    platform = MagicMock()
    platform.post.return_value = [_run()]
    platform.get.return_value = [_result()]

    with (
        patch("hud.cli.qa.require_api_key", return_value="api-key"),
        patch("hud.cli.qa.PlatformClient.from_settings", return_value=platform),
    ):
        result = runner.invoke(app, ["qa", "run", _AGENT_ID, _SUBJECT_ID])

    assert result.exit_code == 0
    assert "passed" in result.output


def test_qa_results_queries_resource_subjects() -> None:
    platform = MagicMock()
    platform.get.return_value = [_result()]

    with (
        patch("hud.cli.qa.require_api_key", return_value="api-key"),
        patch("hud.cli.qa.PlatformClient.from_settings", return_value=platform),
    ):
        result = runner.invoke(
            app,
            ["qa", "results", "task", _SUBJECT_ID, "--json"],
        )

    assert result.exit_code == 0
    assert json.loads(result.output)[0]["canonical_result"]["verdict"] == "passed"
    platform.get.assert_called_once_with(
        "/qa-agents/results/resources",
        params={"subject_type": "task", "subject_ids": [_SUBJECT_ID]},
    )
