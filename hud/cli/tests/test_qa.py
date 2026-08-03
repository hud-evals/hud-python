"""CLI behavior for resource-scoped platform QA agents."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from hud.cli import app
from hud.utils.exceptions import HudNetworkError, HudRequestError, HudTimeoutError

runner = CliRunner()

_AGENT_ID = "00000000-0000-4000-a000-000000000001"
_SUBJECT_ID = "00000000-0000-4000-a000-000000000002"
_SECOND_SUBJECT_ID = "00000000-0000-4000-a000-000000000005"
_TRACE_ID = "00000000-0000-4000-a000-000000000003"
_SECOND_TRACE_ID = "00000000-0000-4000-a000-000000000006"


def _agent() -> dict[str, object]:
    return {
        "id": _AGENT_ID,
        "name": "Benchmark Coverage",
        "subject_type": "taskset",
        "scenario_name": "trace-explorer:taskset_benchmark_coverage",
        "model_name": "claude-sonnet",
        "public": False,
    }


def _run(
    status: str = "queued",
    *,
    subject_id: str = _SUBJECT_ID,
    analysis_trace_id: str = _TRACE_ID,
) -> dict[str, object]:
    return {
        "id": "00000000-0000-4000-a000-000000000004",
        "qa_agent_id": _AGENT_ID,
        "subject_type": "taskset",
        "subject_id": subject_id,
        "analysis_trace_id": analysis_trace_id,
        "status": status,
        "attempt": 1,
    }


def _result(verdict: str = "passed") -> dict[str, object]:
    return {
        "qa_agent_id": _AGENT_ID,
        "subject_type": "taskset",
        "subject_id": _SUBJECT_ID,
        "agent_name": "Benchmark Coverage",
        "analysis_trace_id": _TRACE_ID,
        "status": "completed",
        "canonical_result": {
            "schema_version": "qa_agent_result.v1",
            "verdict": verdict,
            "summary": "Coverage is sufficient." if verdict == "passed" else "A gap was found.",
            "findings": [],
            "metadata": {},
        },
        "error": None,
        "stale": False,
        "attempt": 1,
    }


def test_qa_agents_lists_resource_agents() -> None:
    """Agent discovery forwards subject scope and renders stable identifiers."""
    platform = MagicMock()
    platform.get.return_value = {
        "items": [_agent()],
        "total": 1,
        "limit": 50,
        "offset": 0,
    }

    with (
        patch("hud.cli.qa.require_api_key", return_value="api-key"),
        patch("hud.cli.qa.PlatformClient.from_settings", return_value=platform),
    ):
        result = runner.invoke(app, ["qa", "agents", "--subject-type", "taskset"])

    assert result.exit_code == 0
    assert "Benchmark Coverage" in result.output
    assert _AGENT_ID in result.output
    platform.get.assert_called_once_with(
        "/qa-agents",
        params={"subject_type": "taskset", "limit": 50, "offset": 0},
    )


def test_qa_agents_json_preserves_platform_payload() -> None:
    """Machine output retains pagination and agent fields without reshaping."""
    payload = {"items": [_agent()], "total": 1, "limit": 50, "offset": 0}
    platform = MagicMock()
    platform.get.return_value = payload

    with (
        patch("hud.cli.qa.require_api_key", return_value="api-key"),
        patch("hud.cli.qa.PlatformClient.from_settings", return_value=platform),
    ):
        result = runner.invoke(
            app,
            ["qa", "agents", "--subject-type", "taskset", "--json"],
        )

    assert result.exit_code == 0
    assert json.loads(result.output) == payload


def test_qa_run_reuses_evidence_by_default_and_can_skip_waiting() -> None:
    """The default preserves evidence, while --no-wait returns after selection."""
    platform = MagicMock()
    platform.get.return_value = _agent()
    platform.post.return_value = [_run()]

    with (
        patch("hud.cli.qa.require_api_key", return_value="api-key"),
        patch("hud.cli.qa.PlatformClient.from_settings", return_value=platform),
    ):
        result = runner.invoke(
            app,
            ["qa", "run", _AGENT_ID, _SUBJECT_ID, "--no-wait"],
        )

    assert result.exit_code == 0
    assert "queued" in result.output
    platform.post.assert_called_once_with(
        f"/qa-agents/{_AGENT_ID}/run-resources",
        json={"subject_ids": [_SUBJECT_ID], "overwrite": False},
    )
    platform.get.assert_called_once_with(f"/qa-agents/{_AGENT_ID}")


def test_qa_run_no_wait_renders_reused_result_verdict() -> None:
    """A terminal selection reports its stored verdict without an extra result request."""
    platform = MagicMock()
    platform.get.return_value = _agent()
    platform.post.return_value = [
        {
            **_run(status="completed"),
            "result": {
                "schema_version": "qa_agent_result.v1",
                "verdict": "failed",
                "summary": "A gap was found.",
            },
        },
    ]

    with (
        patch("hud.cli.qa.require_api_key", return_value="api-key"),
        patch("hud.cli.qa.PlatformClient.from_settings", return_value=platform),
    ):
        result = runner.invoke(
            app,
            ["qa", "run", _AGENT_ID, _SUBJECT_ID, "--no-wait"],
        )

    assert result.exit_code == 0
    assert "failed" in result.output
    assert "A gap was found." in result.output
    platform.get.assert_called_once_with(f"/qa-agents/{_AGENT_ID}")


def test_qa_run_waits_for_terminal_result_and_returns_quality_exit() -> None:
    """Waiting returns one for a completed quality failure, not an execution error."""
    platform = MagicMock()
    platform.post.return_value = [_run()]
    platform.get.side_effect = [
        _agent(),
        [{**_run(), "status": "queued"}],
        [_result("failed")],
    ]

    with (
        patch("hud.cli.qa.require_api_key", return_value="api-key"),
        patch("hud.cli.qa.PlatformClient.from_settings", return_value=platform),
        patch("hud.cli.qa.time.sleep"),
    ):
        result = runner.invoke(
            app,
            ["qa", "run", _AGENT_ID, _SUBJECT_ID, "--wait"],
        )

    assert result.exit_code == 1
    assert "failed" in result.output.lower()
    assert "A gap was found." in result.output
    assert platform.get.call_count == 3
    platform.get.assert_called_with(
        "/qa-agents/results/resources",
        params={"subject_type": "taskset", "subject_ids": [_SUBJECT_ID]},
    )


def test_qa_run_reused_failure_preserves_quality_exit() -> None:
    """Run-new-only reuse still evaluates the stored result when waiting."""
    platform = MagicMock()
    platform.post.return_value = [_run(status="completed")]
    platform.get.side_effect = [_agent(), [_result("failed")]]

    with (
        patch("hud.cli.qa.require_api_key", return_value="api-key"),
        patch("hud.cli.qa.PlatformClient.from_settings", return_value=platform),
    ):
        result = runner.invoke(app, ["qa", "run", _AGENT_ID, _SUBJECT_ID])

    assert result.exit_code == 1
    assert "failed" in result.output.lower()
    assert "A gap was found." in result.output
    platform.get.assert_called_with(
        "/qa-agents/results/resources",
        params={"subject_type": "taskset", "subject_ids": [_SUBJECT_ID]},
    )


def test_qa_run_pins_exact_reused_attempt_from_launch_response() -> None:
    """A newer nonmatching attempt cannot replace the exact evidence selected by Platform."""
    selected_failure = _result("failed")
    newer_pass = {
        **_result("passed"),
        "analysis_trace_id": _SECOND_TRACE_ID,
        "attempt": 2,
    }
    platform = MagicMock()
    platform.post.return_value = [_run(status="completed")]
    platform.get.side_effect = [_agent(), [selected_failure, newer_pass]]

    with (
        patch("hud.cli.qa.require_api_key", return_value="api-key"),
        patch("hud.cli.qa.PlatformClient.from_settings", return_value=platform),
    ):
        result = runner.invoke(app, ["qa", "run", _AGENT_ID, _SUBJECT_ID])

    assert result.exit_code == 1
    assert "A gap was found." in result.output


def test_qa_run_matches_canonical_results_for_uppercase_uuid_input() -> None:
    """API-normalized UUID casing does not make a completed result disappear."""
    platform = MagicMock()
    platform.post.return_value = [_run(status="completed")]
    platform.get.side_effect = [_agent(), [_result("passed")]]

    with (
        patch("hud.cli.qa.require_api_key", return_value="api-key"),
        patch("hud.cli.qa.PlatformClient.from_settings", return_value=platform),
    ):
        result = runner.invoke(
            app,
            ["qa", "run", _AGENT_ID.upper(), _SUBJECT_ID.upper()],
        )

    assert result.exit_code == 0
    assert "passed" in result.output.lower()


def test_qa_run_partial_reuse_waits_for_new_and_scores_all_subjects() -> None:
    """A reused failure remains visible while another subject runs."""
    reused_failure = {
        **_result("failed"),
        "subject_id": _SECOND_SUBJECT_ID,
        "analysis_trace_id": _SECOND_TRACE_ID,
    }
    platform = MagicMock()
    platform.post.return_value = [
        _run(),
        _run(
            status="completed",
            subject_id=_SECOND_SUBJECT_ID,
            analysis_trace_id=_SECOND_TRACE_ID,
        ),
    ]
    platform.get.side_effect = [
        _agent(),
        [{**_run(), "status": "queued"}, reused_failure],
        [_result("passed"), reused_failure],
    ]

    with (
        patch("hud.cli.qa.require_api_key", return_value="api-key"),
        patch("hud.cli.qa.PlatformClient.from_settings", return_value=platform),
        patch("hud.cli.qa.time.sleep"),
    ):
        result = runner.invoke(
            app,
            ["qa", "run", _AGENT_ID, _SUBJECT_ID, _SECOND_SUBJECT_ID],
        )

    assert result.exit_code == 1
    assert _SUBJECT_ID in result.output
    assert _SECOND_SUBJECT_ID in result.output
    platform.get.assert_called_with(
        "/qa-agents/results/resources",
        params={
            "subject_type": "taskset",
            "subject_ids": [_SUBJECT_ID, _SECOND_SUBJECT_ID],
        },
    )


def test_qa_run_rejects_incomplete_selected_result_response() -> None:
    """Deployment skew fails closed instead of scoring unpinned historical evidence."""
    platform = MagicMock()
    platform.get.return_value = _agent()
    platform.post.return_value = [_run()]

    with (
        patch("hud.cli.qa.require_api_key", return_value="api-key"),
        patch("hud.cli.qa.PlatformClient.from_settings", return_value=platform),
    ):
        result = runner.invoke(
            app,
            ["qa", "run", _AGENT_ID, _SUBJECT_ID, _SECOND_SUBJECT_ID],
        )

    assert result.exit_code == 3
    assert "exactly one QA result per requested resource" in result.output
    assert platform.get.call_count == 1


def test_qa_run_wait_timeout_is_execution_error() -> None:
    """An exhausted local wait budget is not reported as a quality failure."""
    platform = MagicMock()
    platform.get.return_value = _agent()
    platform.post.return_value = [_run()]

    with (
        patch("hud.cli.qa.require_api_key", return_value="api-key"),
        patch("hud.cli.qa.PlatformClient.from_settings", return_value=platform),
        patch("hud.cli.qa.time.monotonic", side_effect=[0, 2]),
    ):
        result = runner.invoke(
            app,
            ["qa", "run", _AGENT_ID, _SUBJECT_ID, "--wait", "--timeout", "1"],
        )

    assert result.exit_code == 3
    assert "Timed out after 1s" in result.output
    platform.get.assert_called_once_with(f"/qa-agents/{_AGENT_ID}")


def test_qa_run_rejects_missing_analysis_trace_contract() -> None:
    """A malformed launch cannot enter a polling loop that never resolves."""
    platform = MagicMock()
    run = _run()
    del run["analysis_trace_id"]
    platform.get.return_value = _agent()
    platform.post.return_value = [run]

    with (
        patch("hud.cli.qa.require_api_key", return_value="api-key"),
        patch("hud.cli.qa.PlatformClient.from_settings", return_value=platform),
    ):
        result = runner.invoke(app, ["qa", "run", _AGENT_ID, _SUBJECT_ID, "--wait"])

    assert result.exit_code == 3
    assert "exactly one QA result per requested resource" in result.output
    platform.get.assert_called_once_with(f"/qa-agents/{_AGENT_ID}")


def test_qa_results_queries_repeated_subject_ids() -> None:
    """Result inspection passes the canonical resource scope and identifiers."""
    platform = MagicMock()
    platform.get.return_value = [_result()]

    with (
        patch("hud.cli.qa.require_api_key", return_value="api-key"),
        patch("hud.cli.qa.PlatformClient.from_settings", return_value=platform),
    ):
        result = runner.invoke(
            app,
            ["qa", "results", "taskset", _SUBJECT_ID, "--json"],
        )

    assert result.exit_code == 0
    assert json.loads(result.output)[0]["canonical_result"]["verdict"] == "passed"
    platform.get.assert_called_once_with(
        "/qa-agents/results/resources",
        params={"subject_type": "taskset", "subject_ids": [_SUBJECT_ID]},
    )


def test_qa_results_rejects_trace_scope_before_request() -> None:
    """The resource CLI does not route trace subjects through the wrong API."""
    platform = MagicMock()

    with (
        patch("hud.cli.qa.require_api_key", return_value="api-key"),
        patch("hud.cli.qa.PlatformClient.from_settings", return_value=platform),
    ):
        result = runner.invoke(app, ["qa", "results", "trace", _SUBJECT_ID])

    assert result.exit_code == 2
    assert "environment, taskset" in result.output
    platform.get.assert_not_called()


def test_qa_request_failure_uses_request_error_exit() -> None:
    """Authentication and platform failures stay distinct from quality verdicts."""
    platform = MagicMock()
    platform.get.side_effect = HudRequestError("access denied", status_code=403)

    with (
        patch("hud.cli.qa.require_api_key", return_value="api-key"),
        patch("hud.cli.qa.PlatformClient.from_settings", return_value=platform),
    ):
        result = runner.invoke(app, ["qa", "agents", "--subject-type", "environment"])

    assert result.exit_code == 2
    assert "access denied" in result.output


def test_qa_network_failure_uses_execution_error_exit() -> None:
    """Connection failures are execution errors, not quality failures."""
    platform = MagicMock()
    platform.get.side_effect = HudNetworkError("connection failed")

    with (
        patch("hud.cli.qa.require_api_key", return_value="api-key"),
        patch("hud.cli.qa.PlatformClient.from_settings", return_value=platform),
    ):
        result = runner.invoke(app, ["qa", "agents"])

    assert result.exit_code == 3
    assert "connection failed" in result.output


def test_qa_poll_timeout_failure_uses_execution_error_exit() -> None:
    """Transport timeouts during polling preserve the execution-error contract."""
    platform = MagicMock()
    platform.get.side_effect = [_agent(), HudTimeoutError("request timed out")]
    platform.post.return_value = [_run()]

    with (
        patch("hud.cli.qa.require_api_key", return_value="api-key"),
        patch("hud.cli.qa.PlatformClient.from_settings", return_value=platform),
    ):
        result = runner.invoke(app, ["qa", "run", _AGENT_ID, _SUBJECT_ID])

    assert result.exit_code == 3
    assert "request timed out" in result.output
