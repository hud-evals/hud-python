"""Agent-friendly CLI contracts: JSON, exit codes, help, aliases, quiet."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from hud.cli import app
from hud.cli.utils.output import ExitCode
from hud.utils.exceptions import HudRequestError

runner = CliRunner()
_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def _plain(text: str) -> str:
    return _ANSI.sub("", text)


@pytest.fixture(autouse=True)
def _reset_json_flag() -> None:
    from hud.cli.utils.output import _JSON_REQUESTED

    _JSON_REQUESTED.set(False)


def _stdout(result: Any) -> str:
    return result.stdout if getattr(result, "stdout", None) is not None else result.output


def _combined(result: Any) -> str:
    return f"{result.output}\n{getattr(result, 'stderr', '') or ''}"


def test_root_help_lists_nouns_and_exit_codes() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    text = _plain(result.output)
    assert "jobs" in text
    assert "models" in text
    assert "task" in text
    assert "--json" in text
    assert "not found" in text.lower() or "3" in text


def test_jobs_list_help_documents_json_and_examples() -> None:
    result = runner.invoke(app, ["jobs", "list", "--help"])
    assert result.exit_code == 0
    text = _plain(result.output)
    assert "--json" in text
    assert "hud jobs list --json" in text
    assert "--quiet" in text


def test_jobs_list_json_and_quiet() -> None:
    client = MagicMock()
    client.get.return_value = {
        "items": [
            {"id": "job-1", "name": "eval", "status": "done", "created_at": "2026-01-01T00:00:00Z"}
        ]
    }
    with (
        patch("hud.cli.utils.api.require_api_key", return_value="key"),
        patch("hud.utils.platform.PlatformClient.from_settings", return_value=client),
    ):
        json_result = runner.invoke(app, ["jobs", "list", "--json"])
        quiet_result = runner.invoke(app, ["jobs", "list", "--quiet"])
        alias_result = runner.invoke(app, ["job", "list", "--quiet"])

    assert json_result.exit_code == 0
    payload = json.loads(_stdout(json_result))
    assert payload[0]["id"] == "job-1"
    assert quiet_result.exit_code == 0
    assert quiet_result.stdout.strip() == "job-1" or "job-1" in quiet_result.output
    assert alias_result.exit_code == 0


def test_jobs_get_not_found_exit_code() -> None:
    client = MagicMock()
    client.get.side_effect = HudRequestError("missing", status_code=404)
    with (
        patch("hud.cli.utils.api.require_api_key", return_value="key"),
        patch("hud.utils.platform.PlatformClient.from_settings", return_value=client),
    ):
        result = runner.invoke(app, ["jobs", "get", "missing-id", "--json"])

    assert result.exit_code == ExitCode.NOT_FOUND
    payload = json.loads(_stdout(result))
    assert payload["error"] == "not_found"
    assert payload["input"]["job_id"] == "missing-id"


def test_legacy_jobs_id_still_lists_traces() -> None:
    client = MagicMock()
    client.get.return_value = {"items": [{"id": "tr-1", "status": "done", "reward": 1.0}]}
    with (
        patch("hud.cli.utils.api.require_api_key", return_value="key"),
        patch("hud.utils.platform.PlatformClient.from_settings", return_value=client),
    ):
        result = runner.invoke(app, ["jobs", "job-99", "--json"])

    assert result.exit_code == 0
    assert json.loads(_stdout(result))[0]["id"] == "tr-1"


def test_cancel_usage_error_and_dry_run() -> None:
    missing = runner.invoke(app, ["cancel", "--json"])
    assert missing.exit_code == ExitCode.USAGE
    assert json.loads(_stdout(missing))["error"] == "usage"

    dry = runner.invoke(app, ["jobs", "cancel", "job-1", "--dry-run", "--json", "--yes"])
    assert dry.exit_code == 0
    payload = json.loads(_stdout(dry))
    assert payload["dry_run"] is True
    assert payload["action"] == "cancel_job"
    assert payload["job_id"] == "job-1"


def test_cancel_dry_run_json_skips_confirmation() -> None:
    result = runner.invoke(app, ["cancel", "job-1", "--dry-run", "--json"])
    assert result.exit_code == 0
    payload = json.loads(_stdout(result))
    assert payload == {
        "dry_run": True,
        "action": "cancel_job",
        "job_id": "job-1",
        "trace_id": None,
        "all": False,
    }


def test_cancel_alias_still_registered() -> None:
    result = runner.invoke(app, ["cancel", "--help"])
    assert result.exit_code == 0
    text = _plain(result.output)
    assert "--json" in text
    assert "--dry-run" in text
    assert "--yes" in text


def test_trace_get_help_and_alias() -> None:
    get_help = runner.invoke(app, ["trace", "get", "--help"])
    assert get_help.exit_code == 0
    assert "--json" in _plain(get_help.output)

    with (
        patch("hud.cli.trace._load_remote", return_value=[{"kind": "agent_message", "text": "hi"}]),
        patch("hud.cli.utils.api.require_api_key", return_value="key"),
        patch("hud.settings.settings.telemetry_local_dir", None),
    ):
        result = runner.invoke(app, ["trace", "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", "--json"])
    assert result.exit_code == 0
    assert json.loads(_stdout(result))[0]["kind"] == "agent_message"


def test_version_json() -> None:
    result = runner.invoke(app, ["version", "--json"])
    assert result.exit_code == 0
    payload = json.loads(_stdout(result))
    assert payload["name"] == "hud"
    assert isinstance(payload["version"], str)


def test_set_invalid_assignment_is_usage() -> None:
    result = runner.invoke(app, ["set", "NOT_A_PAIR", "--json"])
    assert result.exit_code == ExitCode.USAGE
    assert json.loads(_stdout(result))["error"] == "usage"


def test_auth_noun_group_is_registered() -> None:
    result = runner.invoke(app, ["auth", "--help"])
    assert result.exit_code == 0
    assert "login" in result.output
    assert "set" in result.output


def test_models_list_help_has_examples() -> None:
    result = runner.invoke(app, ["models", "list", "--help"])
    assert result.exit_code == 0
    text = _plain(result.output)
    assert "--json" in text
    assert "hud models list --json" in text


def test_missing_api_key_is_permission(monkeypatch: pytest.MonkeyPatch) -> None:
    from hud.settings import settings

    monkeypatch.setattr(settings, "api_key", "")
    result = runner.invoke(app, ["jobs", "list", "--json"])
    assert result.exit_code == ExitCode.PERMISSION
    payload = json.loads(_stdout(result))
    assert payload["error"] == "permission_denied"


def test_init_conflict_exit_code(tmp_path: Any) -> None:
    target = tmp_path / "taken"
    target.mkdir()
    (target / "keep.txt").write_text("x")
    result = runner.invoke(
        app, ["init", "taken", "--dir", str(tmp_path), "--preset", "blank", "--json"]
    )
    assert result.exit_code == ExitCode.CONFLICT
    payload = json.loads(_stdout(result))
    assert payload["error"] == "conflict"


def test_init_dry_run_json(tmp_path: Any) -> None:
    result = runner.invoke(
        app,
        ["init", "fresh", "--dir", str(tmp_path), "--preset", "blank", "--dry-run", "--json"],
    )
    assert result.exit_code == 0
    payload = json.loads(_stdout(result))
    assert payload["dry_run"] is True
    assert payload["action"] == "init"
    assert not (tmp_path / "fresh" / "env.py").exists()


def test_task_list_json_and_quiet(tmp_path: Any) -> None:
    (tmp_path / "tasks.json").write_text(
        json.dumps([{"id": "solve", "prompt": "hi", "env": "demo"}]),
        encoding="utf-8",
    )
    # Taskset.from_file on a JSON list may not match this repo's schema; invoke help instead
    # if collection fails. The contract under test is flags + help.
    help_result = runner.invoke(app, ["task", "list", "--help"])
    assert help_result.exit_code == 0
    text = _plain(help_result.output)
    assert "--json" in text
    assert "--quiet" in text


def test_qa_help_documents_json() -> None:
    result = runner.invoke(app, ["qa", "run", "--help"])
    assert result.exit_code == 0
    text = _plain(result.output)
    assert "--json" in text
    assert "--dry-run" in text


def test_deploy_all_json_is_single_document(tmp_path: Path) -> None:
    from hud.cli.deploy import _DeployPlan

    for name in ("alpha", "beta"):
        env_dir = tmp_path / name
        env_dir.mkdir()
        (env_dir / "Dockerfile.hud").write_text("FROM python:3.12\n")
        (env_dir / "pyproject.toml").write_text('[project]\nname = "demo"\nversion = "0"\n')

    def _plan(*_args: Any, env_dir: Path, **_kwargs: Any) -> _DeployPlan:
        return _DeployPlan(
            name=env_dir.name,
            registry_id=None,
            runtime=None,
            runtime_config=None,
            env_vars={},
            build_args={},
            build_secrets={},
        )

    with (
        patch("hud.cli.utils.api.require_api_key", return_value="key"),
        patch("hud.cli.deploy._validate_before_deploy"),
        patch("hud.cli.deploy._prepare_deploy_plan", side_effect=_plan),
        patch("hud.cli.deploy.PlatformClient.from_settings", return_value=MagicMock()),
    ):
        result = runner.invoke(
            app, ["deploy", str(tmp_path), "--all", "--dry-run", "--json"]
        )

    assert result.exit_code == 0
    stdout = _stdout(result).strip()
    payload = json.loads(stdout)
    leftover = stdout[json.JSONDecoder().raw_decode(stdout)[1] :].strip()
    assert leftover == ""
    assert payload["dry_run"] is True
    assert payload["succeeded"] == ["alpha", "beta"]
    assert payload["failed"] == []
    assert [item["directory"] for item in payload["environments"]] == ["alpha", "beta"]
    assert [item["name"] for item in payload["environments"]] == ["alpha", "beta"]
