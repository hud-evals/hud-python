"""CLI behavior for HUD environment readiness checks."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from hud.cli import app
from hud.cli.utils.registry import RegistryEnvironment

runner = CliRunner()


def _report(status: str) -> dict[str, object]:
    return {
        "schema_version": "environment_readiness.v1",
        "environment_id": "environment-id",
        "status": status,
        "ready": status == "passed",
        "summary": f"Environment is {status}.",
        "human_report": f"Environment readiness: {status.upper()}",
        "criteria": [],
    }


def test_check_prints_report_and_exits_one_when_not_ready() -> None:
    """A completed advisory failure is distinct from a request error."""
    platform = MagicMock()
    platform.post.return_value = _report("failed")

    with (
        patch("hud.cli.check.require_api_key", return_value="api-key"),
        patch("hud.cli.check.PlatformClient.from_settings", return_value=platform),
        patch(
            "hud.cli.check.resolve_registry_environments",
            return_value=[RegistryEnvironment(id="environment-id", name="browser")],
        ),
    ):
        result = runner.invoke(app, ["check", "browser"])

    assert result.exit_code == 1
    assert "Environment readiness: FAILED" in result.output
    platform.post.assert_called_once_with(
        "/checks/environment-readiness",
        json={"environment_id": "environment-id", "overwrite": False},
    )


def test_check_json_preserves_machine_contract_and_success_exit() -> None:
    """JSON output is stable enough for CI and returns zero only when ready."""
    platform = MagicMock()
    platform.post.return_value = _report("passed")

    with (
        patch("hud.cli.check.require_api_key", return_value="api-key"),
        patch("hud.cli.check.PlatformClient.from_settings", return_value=platform),
        patch(
            "hud.cli.check.resolve_registry_environments",
            return_value=[RegistryEnvironment(id="environment-id", name="browser")],
        ),
    ):
        result = runner.invoke(app, ["check", "browser", "--json", "--overwrite"])

    assert result.exit_code == 0
    assert json.loads(result.output)["status"] == "passed"
    platform.post.assert_called_once_with(
        "/checks/environment-readiness",
        json={"environment_id": "environment-id", "overwrite": True},
    )


def test_check_rejects_ambiguous_environment_without_running() -> None:
    """Name resolution must not silently select the wrong environment."""
    platform = MagicMock()

    with (
        patch("hud.cli.check.require_api_key", return_value="api-key"),
        patch("hud.cli.check.PlatformClient.from_settings", return_value=platform),
        patch(
            "hud.cli.check.resolve_registry_environments",
            return_value=[
                RegistryEnvironment(id="first-id", name="browser"),
                RegistryEnvironment(id="second-id", name="browser-dev"),
            ],
        ),
    ):
        result = runner.invoke(app, ["check", "browser"])

    assert result.exit_code == 2
    assert "matches multiple environments" in result.output
    platform.post.assert_not_called()


def test_check_returns_three_for_platform_execution_error() -> None:
    """A completed platform execution error has its documented exit code."""
    platform = MagicMock()
    platform.post.return_value = _report("error")

    with (
        patch("hud.cli.check.require_api_key", return_value="api-key"),
        patch("hud.cli.check.PlatformClient.from_settings", return_value=platform),
        patch(
            "hud.cli.check.resolve_registry_environments",
            return_value=[RegistryEnvironment(id="environment-id", name="browser")],
        ),
    ):
        result = runner.invoke(app, ["check", "browser"])

    assert result.exit_code == 3


def test_check_active_requests_worker_probe_with_model_override() -> None:
    """The opt-in active mode forwards execution intent without changing static calls."""
    platform = MagicMock()
    platform.post.return_value = _report("unknown")

    with (
        patch("hud.cli.check.require_api_key", return_value="api-key"),
        patch("hud.cli.check.PlatformClient.from_settings", return_value=platform),
        patch(
            "hud.cli.check.resolve_registry_environments",
            return_value=[RegistryEnvironment(id="environment-id", name="browser")],
        ),
    ):
        result = runner.invoke(
            app,
            ["check", "browser", "--active", "--model", "probe-model"],
        )

    assert result.exit_code == 1
    platform.post.assert_called_once_with(
        "/checks/environment-readiness",
        json={
            "environment_id": "environment-id",
            "overwrite": False,
            "run_active_probe": True,
            "probe_model": "probe-model",
        },
    )


def test_check_active_polls_until_probe_is_terminal() -> None:
    """Polling reuses the running attempt even when the initial request overwrites."""
    running = _report("unknown")
    running["criteria"] = [
        {
            "check_key": "environment_probe_task_runs",
            "status": "unknown",
            "evidence": {"execution_trace_id": "trace-id"},
        },
    ]
    completed = _report("unknown")
    completed["criteria"] = [
        {
            "check_key": "environment_probe_task_runs",
            "status": "passed",
            "evidence": {"execution_trace_id": "trace-id"},
        },
    ]
    platform = MagicMock()
    platform.post.side_effect = [running, completed]

    with (
        patch("hud.cli.check.require_api_key", return_value="api-key"),
        patch("hud.cli.check.PlatformClient.from_settings", return_value=platform),
        patch(
            "hud.cli.check.resolve_registry_environments",
            return_value=[RegistryEnvironment(id="environment-id", name="browser")],
        ),
        patch("hud.cli.check.time.sleep") as sleep,
    ):
        result = runner.invoke(
            app,
            ["check", "browser", "--active", "--overwrite"],
        )

    assert result.exit_code == 1
    sleep.assert_called_once()
    assert platform.post.call_count == 2
    assert platform.post.call_args_list[0].kwargs["json"]["overwrite"] is True
    assert platform.post.call_args_list[1].kwargs["json"]["overwrite"] is False


def test_check_active_poll_timeout_is_execution_error() -> None:
    """A locally exhausted wait budget is not a completed quality verdict."""
    running = _report("unknown")
    running["criteria"] = [
        {
            "check_key": "environment_probe_task_runs",
            "status": "unknown",
            "evidence": {"execution_trace_id": "trace-id"},
        },
    ]
    platform = MagicMock()
    platform.post.return_value = running

    with (
        patch("hud.cli.check.require_api_key", return_value="api-key"),
        patch("hud.cli.check.PlatformClient.from_settings", return_value=platform),
        patch(
            "hud.cli.check.resolve_registry_environments",
            return_value=[RegistryEnvironment(id="environment-id", name="browser")],
        ),
        patch("hud.cli.check.time.monotonic", side_effect=[0, 2]),
    ):
        result = runner.invoke(
            app,
            ["check", "browser", "--active", "--timeout", "1"],
        )

    assert result.exit_code == 3
    assert "Timed out after 1s" in result.output
    platform.post.assert_called_once()
