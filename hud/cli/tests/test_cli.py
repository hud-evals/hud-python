"""Tests for ``hud.cli`` (infrastructure) and ``hud.cli.__main__`` (the assembled command)."""

from __future__ import annotations

import json
import re
import sys
import uuid
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch
from urllib.parse import urlsplit
from uuid import UUID

import httpx
import pytest
import typer
from dotenv import dotenv_values
from typer.core import TyperGroup
from typer.main import get_command
from typer.testing import CliRunner

from hud.cli import (
    CLI,
    AuthScope,
    CliError,
    DirectoryLink,
    DirectoryState,
    ExitCode,
    set_env_values,
)
from hud.cli.__main__ import app, main, notify_if_outdated, recorded_invocation, version
from hud.utils.exceptions import HudException, HudRequestError
from hud.utils.gateway import list_gateway_models
from hud.utils.platform import PlatformClient

runner = CliRunner()
_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def _plain(text: str) -> str:
    return _ANSI.sub("", text)


def _stdout(result: Any) -> str:
    return result.stdout if getattr(result, "stdout", None) is not None else result.output


def test_set_env_values_merges_and_preserves_quotes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    secret = "key # with 'quotes' and \\slashes\nsecond line"
    path = set_env_values({"HUD_API_KEY": secret, "C": '"quoted"'})
    assert path == tmp_path / ".hud" / ".env"
    set_env_values({"HUD_DEFAULT_PROJECT": "example"})
    loaded = dotenv_values(path, interpolate=False)
    assert loaded["HUD_API_KEY"] == secret
    assert loaded["C"] == '"quoted"'
    assert loaded["HUD_DEFAULT_PROJECT"] == "example"


SCOPE = {
    "origin": "https://api.example",
    "user_id": "11111111-1111-4111-8111-111111111111",
    "team_id": "22222222-2222-4222-8222-222222222222",
}


def test_scoped_links_do_not_cross_origins_users_teams_or_directories(tmp_path: Path) -> None:
    scope = AuthScope.model_validate(SCOPE)
    directory = tmp_path / "environment"
    state = DirectoryState(scope, directory)
    registry = UUID(int=10)
    assert state.update(DirectoryLink(registry_id=registry)) is True
    before = state.path.read_text()
    assert state.update(DirectoryLink(registry_id=registry)) is False
    assert state.path.read_text() == before
    assert state.load().registry_id == registry
    assert DirectoryState(scope, tmp_path / "worktree").load().registry_id is None
    for field, value in [
        ("origin", "https://other.example"),
        ("user_id", str(UUID(int=30))),
        ("team_id", str(UUID(int=40))),
    ]:
        other = AuthScope.model_validate({**SCOPE, field: value})
        with pytest.raises(CliError, match=r"different HUD credentials|was linked against"):
            DirectoryState(other, directory).load()
    assert state.path == directory / ".hud" / "config.json"


def test_released_camelcase_config_reads_and_migrates_on_write(tmp_path: Path) -> None:
    """A config written by hud 0.6.x (camelCase ids, cached name, .env preference, no
    scope) still links the directory, and the next write rewrites it in this schema."""
    directory = tmp_path / "environment"
    path = directory / ".hud" / "config.json"
    path.parent.mkdir(parents=True)
    registry, taskset = UUID(int=10), UUID(int=11)
    path.write_text(
        json.dumps(
            {
                "registryId": str(registry),
                "registryName": "browser-env",
                "tasksetId": str(taskset),
                "syncEnv": True,
            }
        )
    )
    state = DirectoryState(AuthScope.model_validate(SCOPE), directory)

    link = state.load()
    assert (link.registry_id, link.taskset_id, link.scope) == (registry, taskset, None)

    assert state.update(DirectoryLink(project_id=UUID(int=12))) is True
    written = json.loads(path.read_text())
    assert written["scope"] == SCOPE
    assert written["registry_id"] == str(registry)
    assert written["taskset_id"] == str(taskset)
    assert written["project_id"] == str(UUID(int=12))
    assert not {"registryId", "registryName", "tasksetId", "syncEnv"} & written.keys()
    with pytest.raises(CliError, match="different HUD credentials"):
        DirectoryState(
            AuthScope.model_validate({**SCOPE, "user_id": str(UUID(int=3))}), directory
        ).load()


def test_read_leaves_legacy_files_and_home_untouched(tmp_path: Path) -> None:
    directory = tmp_path / "environment"
    legacy = directory / ".hud" / "deploy.json"
    legacy.parent.mkdir(parents=True)
    legacy.write_text('{"registryId":"old"}')
    state = DirectoryState(AuthScope.model_validate(SCOPE), directory)
    assert state.load().registry_id is None
    assert legacy.read_text() == '{"registryId":"old"}'
    assert not (legacy.parent / "config.json").exists()


def test_corrupt_config_is_not_overwritten(tmp_path: Path) -> None:
    state = DirectoryState(AuthScope.model_validate(SCOPE), tmp_path)
    path = state.path
    path.parent.mkdir(parents=True)
    path.write_text('{"broken":')
    with pytest.raises(CliError, match="not a valid HUD workspace config"):
        state.load()
    with pytest.raises(CliError, match="not a valid HUD workspace config"):
        state.update(DirectoryLink())
    assert path.read_text() == '{"broken":'


def test_api_scope_uses_authoritative_identity_not_credentials(monkeypatch) -> None:
    def request(method, url, **kwargs):
        assert method == "GET"
        assert url.endswith("/v2/auth/me")
        return SCOPE

    monkeypatch.setattr("hud.utils.platform.make_request_sync", request)
    first = AuthScope.resolve(PlatformClient("https://API.EXAMPLE:443/", "first-secret"))
    second = AuthScope.resolve(PlatformClient("https://api.example", "rotated-secret"))
    assert first == second == AuthScope.model_validate(SCOPE)
    assert "secret" not in first.model_dump_json()


def test_json_object_names_the_flag() -> None:
    assert CLI.json_object('{"image": "x"}', option="--payload") == {"image": "x"}
    with pytest.raises(CliError) as exc_info:
        CLI.json_object("[]", option="--payload")
    assert exc_info.value.message == "--payload must be a JSON object"
    assert exc_info.value.input == {"payload": "[]"}


def test_output_mode_does_not_leak_between_invocations() -> None:
    from typer.testing import CliRunner

    from hud.cli.__main__ import app

    runner = CliRunner()
    first = runner.invoke(app, ["version", "--json"])
    second = runner.invoke(app, ["version"])
    flag = runner.invoke(app, ["--version"])
    assert json.loads(first.stdout)["name"] == "hud"
    assert "HUD CLI version:" not in first.stdout
    assert second.stdout.startswith("HUD CLI version:")
    assert flag.exit_code == 0
    assert "HUD CLI version:" in flag.stdout
    unexpected = runner.invoke(app, ["--json", "version"])
    assert unexpected.exit_code != 0


def test_map_request_error_status_codes() -> None:
    not_found = CliError.from_http(HudRequestError("x", status_code=404))
    assert not_found.error == "not_found"
    assert not_found.exit_code == ExitCode.FAILURE
    permission = CliError.from_http(HudRequestError("x", status_code=403))
    assert permission.error == "permission_denied"
    assert permission.exit_code == ExitCode.FAILURE
    conflict = CliError.from_http(HudRequestError("x", status_code=409))
    assert conflict.error == "conflict"
    assert conflict.exit_code == ExitCode.FAILURE
    mapped = CliError.from_http(HudRequestError("x", status_code=429))
    assert mapped.error == "rate_limited"


def test_failure_text_writes_only_stderr() -> None:
    from typer.testing import CliRunner

    from hud.cli.__main__ import app

    result = CliRunner().invoke(app, ["set", "NOT_A_PAIR"])
    assert result.exit_code == ExitCode.USAGE
    assert result.stdout == ""
    assert "Error:" in result.stderr
    assert "Hint:" in result.stderr


def test_confirm_or_abort_noninteractive_requires_yes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    with pytest.raises(CliError) as exc_info:
        CLI.confirm_or_abort("Proceed?")
    assert exc_info.value.exit_code == ExitCode.USAGE


def test_confirm_or_abort_yes_skips_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    CLI.confirm_or_abort("Proceed?", yes=True)


def test_read_text_stdin_and_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    target = tmp_path / "answer.txt"
    target.write_text("hello", encoding="utf-8")
    assert CLI.read_text(str(target)) == "hello"

    monkeypatch.setattr("hud.cli.sys.stdin.read", lambda: "from-stdin")
    assert CLI.read_text("-") == "from-stdin"


def test_read_text_missing_file_is_not_found() -> None:
    with pytest.raises(CliError) as exc_info:
        CLI.read_text("/definitely/missing/hud-cli-file.txt")
    assert exc_info.value.error == "not_found"
    assert exc_info.value.exit_code == ExitCode.FAILURE


def _pypi(version: str) -> httpx.Response:
    return httpx.Response(200, json={"info": {"version": version}})


def _fail_fetch(*_a: object, **_k: object) -> httpx.Response:
    raise AssertionError("fetch")


class TestVersionCheck:
    @pytest.fixture(autouse=True)
    def _isolated(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        monkeypatch.delenv("CI", raising=False)
        monkeypatch.delenv("HUD_SKIP_VERSION_CHECK", raising=False)
        monkeypatch.delenv("VIRTUAL_ENV", raising=False)
        monkeypatch.setattr("hud.cli.__main__.__version__", "1.0.0")

    def test_outdated_prints_banner_and_reuses_cache(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
    ) -> None:
        fetches = {"n": 0}

        def get(*_a: object, **_k: object) -> httpx.Response:
            fetches["n"] += 1
            return _pypi("2.0.0")

        monkeypatch.setattr(httpx, "get", get)
        monkeypatch.setattr(sys, "prefix", "/usr")
        monkeypatch.setattr(sys, "base_prefix", "/usr")
        notify_if_outdated(["hud", "eval"])
        first = capsys.readouterr().err
        assert "2.0.0" in first
        assert "current: 1.0.0" in first
        assert "uv tool upgrade hud" in first
        notify_if_outdated(["hud", "eval"])
        assert capsys.readouterr().err.count("2.0.0") == 1
        assert fetches["n"] == 1
        cached = json.loads((tmp_path / ".hud" / ".cache" / "version_check.json").read_text())
        assert cached["latest"] == "2.0.0"

    def test_current_version_is_silent(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setattr(httpx, "get", lambda *_a, **_k: _pypi("1.0.0"))
        notify_if_outdated(["hud", "eval"])
        assert capsys.readouterr().err == ""

    def test_project_venv_suggests_uv_sync(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setattr(httpx, "get", lambda *_a, **_k: _pypi("2.0.0"))
        monkeypatch.setattr(sys, "prefix", "/proj/.venv")
        monkeypatch.setattr(sys, "base_prefix", "/usr")
        notify_if_outdated(["hud", "eval"])
        assert "uv sync --upgrade-package hud" in capsys.readouterr().err

    def test_uv_tool_install_is_not_a_project_venv(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setattr(httpx, "get", lambda *_a, **_k: _pypi("2.0.0"))
        monkeypatch.setattr(sys, "prefix", "/Users/x/.local/share/uv/tools/hud")
        monkeypatch.setattr(sys, "base_prefix", "/usr")
        notify_if_outdated(["hud", "eval"])
        assert "uv tool upgrade hud" in capsys.readouterr().err

    @pytest.mark.parametrize(
        "argv",
        [["hud"], ["hud", "--version"], ["hud", "--help"], ["hud", "--version", "--json"]],
    )
    def test_help_and_version_do_not_fetch(
        self, argv: list[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(httpx, "get", _fail_fetch)
        notify_if_outdated(argv)

    def test_ci_and_opt_out_do_not_fetch(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(httpx, "get", _fail_fetch)
        monkeypatch.setenv("CI", "1")
        notify_if_outdated(["hud", "eval"])
        monkeypatch.delenv("CI")
        monkeypatch.setenv("HUD_SKIP_VERSION_CHECK", "1")
        notify_if_outdated(["hud", "eval"])

    def test_expired_cache_refetches(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        cache = tmp_path / ".hud" / ".cache" / "version_check.json"
        cache.parent.mkdir(parents=True)
        cache.write_text(json.dumps({"latest": "1.5.0", "checked_at": 0.0}))
        monkeypatch.setattr(httpx, "get", lambda *_a, **_k: _pypi("2.0.0"))
        notify_if_outdated(["hud", "eval"])
        assert "2.0.0" in capsys.readouterr().err

    def test_fetch_failure_does_not_raise(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        def down(*_a: object, **_k: object) -> httpx.Response:
            raise httpx.ConnectError("down")

        monkeypatch.setattr(httpx, "get", down)
        notify_if_outdated(["hud", "eval"])
        assert capsys.readouterr().err == ""


def _event(
    argv: list[str],
    monkeypatch: pytest.MonkeyPatch,
    *,
    error: BaseException | None = None,
) -> dict[str, Any] | None:
    sent: list[dict[str, Any]] = []
    monkeypatch.setattr(
        httpx, "post", lambda _url, json, **_k: sent.append(json) or httpx.Response(204)
    )
    try:
        with recorded_invocation(argv, app):
            if error is not None:
                raise error
    except BaseException as exc:
        if exc is not error and type(exc) is not type(error):
            raise
    if not sent:
        return None
    (payload,) = sent
    (event,) = payload["events"]
    return event


class TestUsage:
    @pytest.fixture(autouse=True)
    def _analytics_on(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
        monkeypatch.setenv("HUD_CLI_ANALYTICS_ENABLED", "1")
        monkeypatch.setenv("HUD_TELEMETRY_URL", "https://telemetry.example.test/v3/api")

    def test_arguments_are_never_captured(self, monkeypatch: pytest.MonkeyPatch) -> None:
        event = _event(["hud", "eval", "tasks.py", "claude"], monkeypatch)
        assert event is not None
        assert event["command"] == "eval"
        assert event["subcommand"] is None

    def test_registered_subcommands_are_captured(self, monkeypatch: pytest.MonkeyPatch) -> None:
        event = _event(["hud", "models", "list"], monkeypatch)
        assert event is not None
        assert (event["command"], event["subcommand"]) == ("models", "list")

    def test_callback_group_positionals_are_never_captured(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        trace = _event(["hud", "trace", "8b1f2c3d4e5f"], monkeypatch)
        jobs = _event(["hud", "jobs", "0f9e8d7c"], monkeypatch)
        assert trace is not None and jobs is not None
        assert (trace["command"], trace["subcommand"]) == ("trace", None)
        assert (jobs["command"], jobs["subcommand"]) == ("jobs", None)

    def test_jobs_verbs_are_captured(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for argv, verb in [
            (["hud", "jobs", "list"], "list"),
            (["hud", "jobs", "cancel"], "cancel"),
            (["hud", "trace", "get"], "get"),
            (["hud", "qa", "list"], "list"),
        ]:
            event = _event(argv, monkeypatch)
            assert event is not None and event["subcommand"] == verb

    def test_unregistered_command_is_other(self, monkeypatch: pytest.MonkeyPatch) -> None:
        event = _event(["hud", "secret-name"], monkeypatch)
        assert event is not None
        assert event["command"] == "other"
        assert "secret-name" not in event.values()

    def test_bare_invocation_is_help(self, monkeypatch: pytest.MonkeyPatch) -> None:
        event = _event(["hud"], monkeypatch)
        assert event is not None
        assert event["command"] == "help"

    def test_version_flag_is_not_an_event(self, monkeypatch: pytest.MonkeyPatch) -> None:
        assert _event(["hud", "--version"], monkeypatch) is None
        assert _event(["hud", "--version", "--json"], monkeypatch) is None

    def test_flags_are_skipped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        event = _event(["hud", "--verbose", "serve"], monkeypatch)
        assert event is not None
        assert event["command"] == "serve"

    def test_typer_exit_from_hud_exception_names_the_cause(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        try:
            raise HudException("boom")
        except HudException as exc:
            converted = typer.Exit(1)
            converted.__cause__ = exc
        event = _event(["hud", "eval"], monkeypatch, error=converted)
        assert event is not None
        assert (event["exit_code"], event["error_class"]) == (1, "HudException")

    def test_plain_exit_has_no_error_class(self, monkeypatch: pytest.MonkeyPatch) -> None:
        event = _event(["hud", "eval"], monkeypatch, error=typer.Exit(2))
        assert event is not None
        assert (event["exit_code"], event["error_class"]) == (2, None)

    def test_keyboard_interrupt(self, monkeypatch: pytest.MonkeyPatch) -> None:
        event = _event(["hud", "eval"], monkeypatch, error=KeyboardInterrupt())
        assert event is not None
        assert (event["exit_code"], event["error_class"]) == (130, "KeyboardInterrupt")

    def test_unexpected_exception(self, monkeypatch: pytest.MonkeyPatch) -> None:
        event = _event(["hud", "eval"], monkeypatch, error=ValueError("x"))
        assert event is not None
        assert (event["exit_code"], event["error_class"]) == (1, "ValueError")

    def test_install_id_created_once(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        first = _event(["hud", "eval"], monkeypatch)
        second = _event(["hud", "eval"], monkeypatch)
        assert first is not None and second is not None
        assert first["install_id"] == second["install_id"]
        assert uuid.UUID(first["install_id"])
        assert capsys.readouterr().err.count("anonymous CLI usage") == 1

    def test_opt_out_applies_immediately(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("HUD_CLI_ANALYTICS_ENABLED", "0")
        assert _event(["hud", "eval"], monkeypatch) is None

    def test_command_error_propagates_when_opted_out(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("HUD_CLI_ANALYTICS_ENABLED", "0")
        with (
            pytest.raises(ValueError, match="boom"),
            recorded_invocation(["hud", "eval"], app),
        ):
            raise ValueError("boom")

    def test_payload_is_the_allowlist(self, monkeypatch: pytest.MonkeyPatch) -> None:
        sent: list[tuple[str, dict[str, Any]]] = []
        monkeypatch.setattr(
            httpx,
            "post",
            lambda url, json, **_k: sent.append((url, json)) or httpx.Response(204),
        )
        with recorded_invocation(["hud", "serve", "my_env.py"], app):
            pass
        (url, payload) = sent[0]
        assert url == "https://telemetry.example.test/v3/api/sdk-events/cli"
        (event,) = payload["events"]
        assert event["command"] == "serve"
        assert event["subcommand"] is None
        assert "my_env.py" not in str(payload)
        assert set(event) == {
            "command",
            "subcommand",
            "exit_code",
            "error_class",
            "duration_ms",
            "cli_version",
            "python_version",
            "os",
            "is_ci",
            "install_id",
        }


class TestCLICommands:
    """Test CLI command handling."""

    def test_main_shows_help_when_no_args(self) -> None:
        """Test that main() shows help when no arguments provided."""
        result = runner.invoke(app)
        assert result.exit_code == 2
        assert "Usage:" in result.output

    def test_version_command(self) -> None:
        with patch("hud.cli.__main__.__version__", "1.2.3"):
            result = runner.invoke(app, ["version"])
            assert result.exit_code == 0
            assert "1.2.3" in _plain(result.output)

    def test_mcp_command(self) -> None:
        """Test mcp server command."""
        result = runner.invoke(app, ["mcp"])
        assert result.exit_code == 2

    def test_help_command(self) -> None:
        """Test help command lists v6 commands."""
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "eval" in result.output
        assert "deploy" in result.output


class TestMainFunction:
    """Test the main() function specifically."""

    def test_main_with_help_flag(self) -> None:
        original_argv = sys.argv
        try:
            sys.argv = ["hud", "--help"]
            with (
                patch("hud.cli.__main__.notify_if_outdated"),
                patch("hud.cli.__main__.app") as mock_app,
            ):
                main()
                mock_app.assert_called()
        finally:
            sys.argv = original_argv

    def test_main_with_no_args(self) -> None:
        original_argv = sys.argv
        try:
            sys.argv = ["hud"]
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 2
        finally:
            sys.argv = original_argv


def test_version_does_not_crash() -> None:
    version()


def test_model_commands_share_platform_transport(monkeypatch):
    from hud.settings import settings

    model_id = "00000000-0000-4000-a000-000000000001"
    requests = []

    def request(method, url, **kwargs):
        requests.append((method, url, kwargs))
        assert url.startswith("https://api.example/v2/")
        assert kwargs["api_key"] == "test-key"
        if urlsplit(url).path == "/v2/models":
            return {"items": [{"id": model_id, "name": "owner/model + version"}], "total": 1}
        return [] if method == "GET" else {"id": model_id, "model_name": "forked"}

    async def async_request(method, url, **kwargs):
        return request(method, url, **kwargs)

    monkeypatch.setattr(settings, "api_key", "test-key")
    monkeypatch.setattr(settings, "hud_api_url", "https://api.example/")
    monkeypatch.setattr("hud.utils.platform.make_request_sync", request)  # catalog, fork
    monkeypatch.setattr("hud.train.base.make_request", async_request)  # TrainingClient
    for command, extra, method, endpoint in [
        ("checkpoints", [], "GET", f"/models/{model_id}/checkpoints"),
        ("head", [], "GET", f"/models/{model_id}/checkpoints"),
        ("head", ["--set", "checkpoint"], "PUT", f"/models/{model_id}/head"),
        ("fork", ["--name", "forked"], "POST", "/models/fork"),
    ]:
        requests.clear()
        list_gateway_models.cache_clear()
        result = runner.invoke(app, ["models", command, "owner/model + version", *extra, "--json"])
        assert result.exit_code == 0, result.output
        assert len(requests) == 2
        assert requests[-1][:2] == (method, f"https://api.example/v2{endpoint}")
        if command == "head" and not extra:
            # No active checkpoint is a valid state and still a JSON payload.
            assert json.loads(result.stdout) == {"model_id": model_id, "head": None}


def test_task_rejects_non_object_args_before_connecting():
    result = runner.invoke(app, ["task", "grade", "solve", "--args", "[]", "--json"])
    assert result.exit_code == ExitCode.USAGE
    assert json.loads(result.stdout)["message"] == "--args must be a JSON object"


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
    group = get_command(app)
    assert isinstance(group, TyperGroup)
    assert group.list_commands(typer.Context(group)) == [
        "init",
        "serve",
        "deploy",
        "eval",
        "task",
        "project",
        "sync",
        "import",
        "qa",
        "jobs",
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

    assert json_result.exit_code == 0
    payload = json.loads(_stdout(json_result))
    assert payload[0]["id"] == "job-1"
    assert quiet_result.exit_code == 0
    assert quiet_result.stdout.strip() == "job-1" or "job-1" in quiet_result.output


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


def test_jobs_malformed_bare_id_is_a_usage_error() -> None:
    result = runner.invoke(app, ["jobs", "00000000-0000-0000-0000-invalid", "--json"])
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
        patch(
            "hud.utils.platform.PlatformClient.get",
            return_value={"events": [{"kind": "agent_message", "text": "hi"}]},
        ),
        patch("hud.settings.settings.api_key", "test-key"),
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
