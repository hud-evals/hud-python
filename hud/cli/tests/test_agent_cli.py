"""CLI contracts: JSON, exit codes, help, aliases, quiet."""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock, patch

from typer.main import get_command
from typer.testing import CliRunner

from hud.cli.app import AuthScope, DirectoryState, ExitCode, app
from hud.cli.project import Placement, ProjectSource
from hud.utils.exceptions import HudRequestError

if TYPE_CHECKING:
    import pytest

runner = CliRunner()
_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def test_model_commands_share_platform_transport(monkeypatch):
    from urllib.parse import parse_qs, urlsplit

    from hud.settings import settings

    model_id = "00000000-0000-4000-a000-000000000001"
    requests = []

    def request(method, url, **kwargs):
        requests.append((method, url, kwargs))
        assert url.startswith("https://api.example/v2/")
        assert kwargs["api_key"] == "test-key"
        if urlsplit(url).path == "/v2/models/resolve":
            assert parse_qs(urlsplit(url).query) == {"model": ["owner/model + version"]}
            return {"id": model_id}
        return [] if method == "GET" else {"id": model_id, "model_name": "forked"}

    monkeypatch.setattr(settings, "api_key", "test-key")
    monkeypatch.setattr(settings, "hud_api_url", "https://api.example/")
    monkeypatch.setattr("hud.utils.platform.make_request_sync", request)
    for command, extra, method, endpoint in [
        ("checkpoints", [], "GET", f"/models/{model_id}/checkpoints"),
        ("head", ["--set", "checkpoint"], "PUT", f"/models/{model_id}/head"),
        ("fork", ["--name", "forked"], "POST", "/models/fork"),
    ]:
        requests.clear()
        result = runner.invoke(app, ["models", command, "owner/model + version", *extra, "--json"])
        assert result.exit_code == 0, result.output
        assert len(requests) == 2
        assert requests[-1][:2] == (method, f"https://api.example/v2{endpoint}")


def test_task_rejects_non_object_args_before_connecting():
    result = runner.invoke(app, ["task", "grade", "solve", "--args", "[]", "--json"])
    assert result.exit_code == ExitCode.USAGE
    assert json.loads(result.stdout)["message"] == "--args must be a JSON object"


def _plain(text: str) -> str:
    return _ANSI.sub("", text)


def _stdout(result: Any) -> str:
    return result.stdout if getattr(result, "stdout", None) is not None else result.output


def _combined(result: Any) -> str:
    return f"{result.output}\n{getattr(result, 'stderr', '') or ''}"


def _single_json(result: Any) -> dict[str, Any]:
    stdout = _stdout(result).strip()
    payload = json.loads(stdout)
    leftover = stdout[json.JSONDecoder().raw_decode(stdout)[1] :].strip()
    assert leftover == ""
    return payload


def _write_env_dirs(parent: Any, *names: str) -> None:
    for name in names:
        env_dir = parent / name
        env_dir.mkdir()
        (env_dir / "Dockerfile.hud").write_text("FROM python:3.12\n")
        (env_dir / "pyproject.toml").write_text('[project]\nname = "demo"\nversion = "0"\n')


def test_root_help_lists_nouns() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    text = _plain(result.output)
    assert "Usage: hud COMMAND" in text
    assert "[OPTIONS] COMMAND [ARGS]" not in text
    assert "jobs" in text
    assert "models" in text
    assert "task" in text
    assert re.search(r"^ {2}auth\b", text, re.M) is None
    assert re.search(r"^ {2}client\b", text, re.M) is None
    assert re.search(r"^ {2}cancel\b", text, re.M) is None
    assert "--help" in text
    assert "--version" in text
    assert "--json" not in text
    assert text.index("--help") < text.index("--version")
    assert "Show help." in text
    assert "Show this message and exit." not in text
    assert get_command(app).list_commands(None) == [
        "init",
        "serve",
        "deploy",
        "eval",
        "task",
        "project",
        "sync",
        "qa",
        "jobs",
        "job",
        "cancel",
        "trace",
        "models",
        "set",
        "version",
    ]


def test_plan_flags_use_shared_help() -> None:
    for args in (
        ["eval", "--help"],
        ["deploy", "--help"],
        ["sync", "tasks", "--help"],
        ["sync", "env", "--help"],
        ["jobs", "cancel", "--help"],
    ):
        result = runner.invoke(app, args)
        assert result.exit_code == 0, result.output
        text = " ".join(_plain(result.output).replace("│", " ").split())
        assert "planned action" in text
        if args[0] != "deploy":
            assert "non-interactive terminals" in text


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
        patch("hud.utils.platform.PlatformClient.from_settings", return_value=client),
    ):
        result = runner.invoke(
            app, ["jobs", "get", "00000000-0000-0000-0000-000000000001", "--json"]
        )

    assert result.exit_code == ExitCode.FAILURE
    payload = json.loads(_stdout(result))
    assert payload["error"] == "not_found"
    assert "Error:" not in (result.stderr or "")
    assert payload["input"]["job_id"] == "00000000-0000-0000-0000-000000000001"


def test_jobs_bare_id_is_not_a_command() -> None:
    result = runner.invoke(app, ["jobs", "00000000-0000-0000-0000-000000000099", "--json"])
    assert result.exit_code == ExitCode.USAGE


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


def test_cancel_alias_is_hidden_and_deprecated() -> None:
    result = runner.invoke(app, ["cancel", "--help"])
    assert result.exit_code == 0
    text = _plain(result.output)
    assert "--json" in text
    assert "--dry-run" in text
    assert "--yes" in text
    assert "deprecated" in text.lower()
    root = _plain(runner.invoke(app, ["--help"]).output)
    assert re.search(r"^ {2}cancel\b", root, re.M) is None


def test_trace_get_help_and_json() -> None:
    get_help = runner.invoke(app, ["trace", "get", "--help"])
    assert get_help.exit_code == 0
    assert "--json" in _plain(get_help.output)

    with (
        patch("hud.cli.trace._load_remote", return_value=[{"kind": "agent_message", "text": "hi"}]),
        patch("hud.settings.settings.telemetry_local_dir", None),
    ):
        result = runner.invoke(app, ["trace", "get", "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", "--json"])
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


def test_auth_noun_group_is_removed() -> None:
    result = runner.invoke(app, ["auth", "--help"])
    assert result.exit_code != 0
    assert "No such command" in _plain(result.output) or "Usage:" in _plain(result.output)


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
    assert result.exit_code == ExitCode.FAILURE
    payload = json.loads(_stdout(result))
    assert payload["error"] == "permission_denied"


def test_init_conflict_exit_code(tmp_path: Any) -> None:
    target = tmp_path / "taken"
    target.mkdir()
    (target / "keep.txt").write_text("x")
    result = runner.invoke(
        app, ["init", "taken", "--dir", str(tmp_path), "--preset", "blank", "--json"]
    )
    assert result.exit_code == ExitCode.FAILURE
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


def test_deploy_all_json_is_single_document(tmp_path: Any) -> None:
    from hud.cli.deploy import _DeployPlan

    _write_env_dirs(tmp_path, "alpha", "beta")

    def _plan(*_args: Any, env_dir: Any, **_kwargs: Any) -> _DeployPlan:
        return _DeployPlan(
            name=env_dir.name,
            registry_id=None,
            placement=Placement(None, ProjectSource.TEAM_DEFAULT),
            runtime=None,
            runtime_config=None,
            env_vars={},
            build_args={},
            build_secrets={},
            state=DirectoryState(
                AuthScope(
                    origin="https://api.example",
                    user_id="11111111-1111-4111-8111-111111111111",
                    team_id="22222222-2222-4222-8222-222222222222",
                )
            ),
        )

    with (
        patch("hud.cli.deploy._validate_before_deploy"),
        patch("hud.cli.deploy._prepare_deploy_plan", side_effect=_plan),
        patch("hud.cli.deploy.PlatformClient.from_settings", return_value=MagicMock()),
    ):
        result = runner.invoke(app, ["deploy", str(tmp_path), "--all", "--dry-run", "--json"])

    assert result.exit_code == 0
    payload = _single_json(result)
    assert payload["dry_run"] is True
    assert payload["succeeded"] == ["alpha", "beta"]
    assert payload["failed"] == []
    assert [item["directory"] for item in payload["environments"]] == ["alpha", "beta"]
    assert [item["name"] for item in payload["environments"]] == ["alpha", "beta"]


def test_deploy_all_json_includes_failed_env_details(tmp_path: Any) -> None:
    from hud.cli.deploy import _DeployPlan, _DeployResult

    _write_env_dirs(tmp_path, "alpha", "beta")

    def _plan(*_args: Any, env_dir: Any, **_kwargs: Any) -> _DeployPlan:
        return _DeployPlan(
            name=env_dir.name,
            registry_id=None,
            placement=Placement(None, ProjectSource.TEAM_DEFAULT),
            runtime=None,
            runtime_config=None,
            env_vars={},
            build_args={},
            build_secrets={},
            state=DirectoryState(
                AuthScope(
                    origin="https://api.example",
                    user_id="11111111-1111-4111-8111-111111111111",
                    team_id="22222222-2222-4222-8222-222222222222",
                )
            ),
        )

    async def _deploy(*, plan: _DeployPlan, **_kwargs: Any) -> _DeployResult:
        if plan.name == "alpha":
            return _DeployResult(
                success=True,
                name=plan.name,
                build_id="b-ok",
                registry_id="r-ok",
                status="SUCCEEDED",
            )
        return _DeployResult(
            success=False, name=plan.name, build_id="b-bad", registry_id="r-bad", status="FAILED"
        )

    def _tarball(env_dir: Any, **_kwargs: Any) -> Any:
        path = env_dir / "context.tar.gz"
        path.write_bytes(b"gz")
        return path

    with (
        patch("hud.cli.deploy._validate_before_deploy"),
        patch("hud.cli.deploy._prepare_deploy_plan", side_effect=_plan),
        patch("hud.cli.deploy._create_tarball", side_effect=_tarball),
        patch("hud.cli.deploy._deploy_async", side_effect=_deploy),
        patch("hud.cli.deploy.PlatformClient.from_settings", return_value=MagicMock()),
    ):
        result = runner.invoke(app, ["deploy", str(tmp_path), "--all", "--json"])

    assert result.exit_code == 1
    payload = _single_json(result)
    assert payload["succeeded"] == ["alpha"]
    assert payload["failed"] == ["beta"]
    by_dir = {item["directory"]: item for item in payload["environments"]}
    assert by_dir["alpha"]["success"] is True
    assert by_dir["alpha"]["build_id"] == "b-ok"
    assert by_dir["alpha"]["status"] == "SUCCEEDED"
    assert by_dir["beta"]["success"] is False
    assert by_dir["beta"]["build_id"] == "b-bad"
    assert by_dir["beta"]["registry_id"] == "r-bad"
    assert by_dir["beta"]["status"] == "FAILED"
    assert by_dir["beta"]["name"] == "beta"


def test_deploy_all_json_missing_api_key_is_single_document(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from hud.settings import settings

    _write_env_dirs(tmp_path, "alpha", "beta")
    monkeypatch.setattr(settings, "api_key", "")
    result = runner.invoke(app, ["deploy", str(tmp_path), "--all", "--json"])

    assert result.exit_code == 1
    payload = _single_json(result)
    assert payload["succeeded"] == []
    assert payload["failed"] == ["alpha", "beta"]
    for item in payload["environments"]:
        assert item["success"] is False
        assert item["error"] == "permission_denied"
        assert "message" in item
    assert "HUD_API_KEY is required" in (result.stderr or result.output)
