"""Tests for CLI deploy command."""

from __future__ import annotations

import io
import json
import sys
import tarfile
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from typer.testing import CliRunner

from hud.cli import AuthScope, DirectoryState
from hud.cli.__main__ import app
from hud.utils.hud_console import HUDConsole
from hud.utils.platform import PlatformClient

if TYPE_CHECKING:
    from collections.abc import Iterator

_PROJECT_ID = "44444444-4444-4444-8444-444444444444"
_REGISTRY_ID = "55555555-5555-4555-8555-555555555555"


class _FakeHttpxClient:
    uploaded: bytes = b""

    def __init__(self, *args: object, **kwargs: object) -> None:
        pass

    async def __aenter__(self) -> _FakeHttpxClient:
        return self

    async def __aexit__(self, *args: object) -> None:
        return None

    async def put(self, *args: object, **kwargs: object) -> MagicMock:
        content = kwargs["content"]
        assert isinstance(content, bytes)
        type(self).uploaded = content
        response = MagicMock()
        response.raise_for_status = MagicMock()
        return response


class _TtyCliRunner(CliRunner):
    """CliRunner replaces stdin with a pipe; this one still reports a TTY."""

    @contextmanager
    def isolation(self, *args: Any, **kwargs: Any) -> Iterator[Any]:
        with (
            super().isolation(*args, **kwargs) as streams,
            patch.object(sys.stdin, "isatty", return_value=True),
        ):
            yield streams


def _write_env(tmp_path: Path, name: str = "e") -> None:
    (tmp_path / "Dockerfile").write_text("FROM python:3.12\n", encoding="utf-8")
    (tmp_path / "env.py").write_text(f'env = Environment("{name}")\n', encoding="utf-8")


def _dry_run(tmp_path: Path, *args: str) -> dict[str, Any]:
    result = CliRunner().invoke(
        app, ["deploy", str(tmp_path), "--dry-run", "--json", "--no-env", *args]
    )
    assert result.exit_code == 0, result.output
    return json.loads(result.stdout)


def _stub_remote_build(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    captured: dict[str, Any] = {}

    async def request(method: str, url: str, **kwargs: Any) -> dict[str, Any]:
        if url.endswith("/upload-url"):
            return {"upload_url": "https://upload.example", "build_id": "build-1"}
        if url.endswith("/trigger"):
            captured.clear()
            captured.update(kwargs["json"])
            return {"id": "build-1", "registry_id": _REGISTRY_ID}
        if url.endswith("/status"):
            return {"status": "SUCCEEDED"}
        raise AssertionError(url)

    websocket = MagicMock()
    websocket.__aiter__.return_value = [
        json.dumps({"type": "complete", "final_status": "SUCCEEDED", "message": "Build SUCCEEDED"})
    ]
    connection = MagicMock()
    connection.__aenter__.return_value = websocket
    monkeypatch.setattr("hud.utils.platform.make_request", request)
    monkeypatch.setattr("hud.cli.deploy.httpx.AsyncClient", _FakeHttpxClient)
    monkeypatch.setattr("hud.cli.deploy.websockets.connect", lambda *a, **k: connection)
    monkeypatch.setattr("hud.cli.deploy.asyncio.sleep", AsyncMock())
    return captured


def _archive(source: Path, monkeypatch: pytest.MonkeyPatch) -> set[str]:
    directory = source.parent if source.is_file() else source
    if not any(
        "Environment(" in path.read_text(encoding="utf-8")
        for path in directory.glob("*.py")
        if path.is_file()
    ):
        (directory / "env.py").write_text('env = Environment("e")\n', encoding="utf-8")
    _FakeHttpxClient.uploaded = b""
    _stub_remote_build(monkeypatch)
    result = CliRunner().invoke(app, ["deploy", str(source), "--json", "--no-env"])
    assert result.exit_code == 0, result.output
    with tarfile.open(fileobj=io.BytesIO(_FakeHttpxClient.uploaded), mode="r:gz") as tar:
        return set(tar.getnames())


def test_runtime_flag_is_case_insensitive(tmp_path: Path) -> None:
    _write_env(tmp_path)

    assert _dry_run(tmp_path, "--runtime", "HUD")["runtime"] == "hud"


def test_unknown_runtime_is_a_usage_error_before_upload(tmp_path: Path) -> None:
    _write_env(tmp_path)

    message = _deploy_name_error(tmp_path, "--runtime", "moddal")
    assert "Unknown runtime 'moddal'" in message
    assert "hud, modal" in message


def _deploy_name_error(tmp_path: Path, *args: str) -> str:
    result = CliRunner().invoke(
        app, ["deploy", str(tmp_path), "--dry-run", "--json", "--no-env", *args]
    )
    assert result.exit_code == 2, result.output
    return json.loads(result.stdout)["message"]


class TestResolveEnvironmentName:
    """Tests for code-authoritative environment name resolution."""

    def test_single_declared_name_wins(self, tmp_path: Path) -> None:
        (tmp_path / "env.py").write_text('env = Environment("my-env")\n', encoding="utf-8")

        assert _dry_run(tmp_path)["name"] == "my-env"

    def test_repeated_same_name_is_fine(self, tmp_path: Path) -> None:
        (tmp_path / "a.py").write_text('a = Environment("same")\n', encoding="utf-8")
        (tmp_path / "b.py").write_text('b = Environment(name="same")\n', encoding="utf-8")
        (tmp_path / "c.py").write_text('c = hud.Environment("same")\n', encoding="utf-8")

        assert _dry_run(tmp_path)["name"] == "same"

    def test_nested_source_counts_but_junk_directories_do_not(self, tmp_path: Path) -> None:
        nested = tmp_path / "server" / "pkg"
        nested.mkdir(parents=True)
        (nested / "env.py").write_text('env = Environment("nested")\n', encoding="utf-8")
        for junk in (".venv", "node_modules", "__pycache__"):
            (tmp_path / junk).mkdir()
            (tmp_path / junk / "env.py").write_text(
                'env = Environment("excluded")\n', encoding="utf-8"
            )
        (tmp_path / "broken.py").write_text("def broken(:\n", encoding="utf-8")

        assert _dry_run(tmp_path)["name"] == "nested"

    def test_explicit_file_disambiguates_nested_environment(self, tmp_path: Path) -> None:
        selected = tmp_path / "env.py"
        selected.write_text('env = Environment("adapter")\n', encoding="utf-8")
        nested = tmp_path / "source"
        nested.mkdir()
        (nested / "env.py").write_text('env = Environment("original")\n', encoding="utf-8")

        assert _dry_run(selected)["name"] == "adapter"
        assert "Pass --name" in _deploy_name_error(tmp_path)

    def test_explicit_file_without_declaration_does_not_select_another_file(
        self, tmp_path: Path
    ) -> None:
        selected = tmp_path / "selected.py"
        selected.write_text("x = 1\n", encoding="utf-8")
        (tmp_path / "env.py").write_text('env = Environment("other")\n', encoding="utf-8")

        assert "No environment found" in _deploy_name_error(selected)

    def test_ignores_calls_without_a_literal(self, tmp_path: Path) -> None:
        (tmp_path / "env.py").write_text(
            'env = Environment("named")\nother = Environment(name=NAME)\n',
            encoding="utf-8",
        )

        assert _dry_run(tmp_path)["name"] == "named"

    def test_name_flag_selects_among_literals(self, tmp_path: Path) -> None:
        (tmp_path / "env.py").write_text(
            'actor = Environment("workspace")\nverifier = Environment("judge")\n',
            encoding="utf-8",
        )
        assert _dry_run(tmp_path, "--name", "judge")["name"] == "judge"
        assert "Pass --name" in _deploy_name_error(tmp_path)

    def test_serve_line_does_not_break_ties(self, tmp_path: Path) -> None:
        (tmp_path / "Dockerfile").write_text('CMD ["hud", "serve", "env:env"]\n', encoding="utf-8")
        (tmp_path / "env.py").write_text('env = Environment("trace-explorer")\n', encoding="utf-8")
        (tmp_path / "verify.py").write_text(
            'verify_env = Environment("qa-verifier")\n', encoding="utf-8"
        )

        assert "Pass --name" in _deploy_name_error(tmp_path)
        assert _dry_run(tmp_path, "--name", "trace-explorer")["name"] == "trace-explorer"
        assert "No environment named 'missing'" in _deploy_name_error(tmp_path, "--name", "missing")

    @pytest.mark.parametrize(
        "source", ["x = 1\n", "env = Environment()\n", "env = Environment(name=NAME)\n"]
    )
    def test_requires_a_literal(self, tmp_path: Path, source: str) -> None:
        (tmp_path / "env.py").write_text(source, encoding="utf-8")

        assert "No environment found" in _deploy_name_error(tmp_path)

    def test_registry_id_does_not_replace_missing_declaration(self, tmp_path: Path) -> None:
        (tmp_path / "server.py").write_text("x = 1\n", encoding="utf-8")

        assert "No environment found" in _deploy_name_error(tmp_path, "--registry-id", "r-1")

    def test_registry_id_must_name_the_declared_environment(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (tmp_path / "env.py").write_text('env = Environment("My Env")\n', encoding="utf-8")
        monkeypatch.setattr("hud.settings.settings.api_key", "test-key")

        def get(method: str, url: str, **kwargs: Any) -> dict[str, Any]:
            assert url.endswith(f"/registry/{_REGISTRY_ID}")
            return {"id": _REGISTRY_ID, "name": "other-env"}

        monkeypatch.setattr("hud.utils.platform.make_request_sync", get)
        message = _deploy_name_error(tmp_path, "--registry-id", _REGISTRY_ID)
        assert "Environment('My Env')" in message
        assert "targets 'other-env'" in message


class TestBuildContext:
    def test_excludes_secrets_and_keeps_source(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (tmp_path / "main.py").write_text("print('hi')", encoding="utf-8")
        (tmp_path / ".env").write_text("SECRET=1", encoding="utf-8")
        (tmp_path / ".git").mkdir()
        (tmp_path / ".git" / "config").write_text("x", encoding="utf-8")

        names = _archive(tmp_path, monkeypatch)
        assert "main.py" in names
        assert "env.py" in names
        assert ".env" not in names
        assert not any(name == ".git" or name.startswith(".git/") for name in names)

    def test_explicit_file_uploads_its_parent_tree(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        selected = tmp_path / "env.py"
        selected.write_text('env = Environment("selected")\n', encoding="utf-8")
        nested = tmp_path / "source"
        nested.mkdir()
        (nested / "env.py").write_text('env = Environment("original")\n', encoding="utf-8")
        (tmp_path / "compose.yaml").write_text("services: {}\n", encoding="utf-8")

        names = _archive(selected, monkeypatch)
        assert {"env.py", "compose.yaml", "source/env.py"} <= names

    def test_preserves_empty_directories(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (tmp_path / "packages").mkdir()
        (tmp_path / "ignored").mkdir()
        (tmp_path / ".dockerignore").write_text("ignored/\n", encoding="utf-8")

        names = _archive(tmp_path, monkeypatch)
        assert "packages" in names
        assert "ignored" not in names

    def test_gitignore_does_not_scope_the_context(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (tmp_path / ".gitignore").write_text("data/\nbundle.bin\n", encoding="utf-8")
        (tmp_path / "bundle.bin").write_text("weights", encoding="utf-8")
        (tmp_path / "data").mkdir()
        (tmp_path / "data" / "fixture.json").write_text("{}", encoding="utf-8")

        assert {"bundle.bin", "data/fixture.json"} <= _archive(tmp_path, monkeypatch)

    def test_dockerignore_can_reinclude_default_junk(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (tmp_path / "dist").mkdir()
        (tmp_path / "dist" / "app.whl").write_text("wheel", encoding="utf-8")
        (tmp_path / "node_modules").mkdir()
        (tmp_path / "node_modules" / "dep.js").write_text("x", encoding="utf-8")
        (tmp_path / ".dockerignore").write_text("!node_modules\n", encoding="utf-8")

        names = _archive(tmp_path, monkeypatch)
        assert "dist/app.whl" in names
        assert "node_modules/dep.js" in names

    def test_dockerignore_negations_cannot_reinclude_secrets(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        for name in [".env", ".env.prod", "service.env", ".git/config", "nested/.env"]:
            path = tmp_path / name
            path.parent.mkdir(exist_ok=True)
            path.write_text("secret", encoding="utf-8")
        (tmp_path / "keep.pyc").write_text("cache", encoding="utf-8")
        (tmp_path / ".dockerignore").write_text("!*\n", encoding="utf-8")

        assert _archive(tmp_path, monkeypatch) == {"nested", "keep.pyc", ".dockerignore", "env.py"}

    def test_dockerignore_skips_comments_and_honors_last_match(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (tmp_path / "keep.pyc").write_text("x", encoding="utf-8")
        (tmp_path / "drop.pyc").write_text("x", encoding="utf-8")
        (tmp_path / ".dockerignore").write_text("# comment\n\n*.pyc\n!keep.pyc\n", encoding="utf-8")

        names = _archive(tmp_path, monkeypatch)
        assert "keep.pyc" in names
        assert "drop.pyc" not in names

    def test_dockerignore_globs_and_directory_patterns(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (tmp_path / "a.py").write_text("x", encoding="utf-8")
        (tmp_path / "a.pyc").write_text("x", encoding="utf-8")
        (tmp_path / "pkg").mkdir()
        (tmp_path / "pkg" / "a.pyc").write_text("x", encoding="utf-8")
        (tmp_path / "node_modules").mkdir()
        (tmp_path / "node_modules" / "index.js").write_text("x", encoding="utf-8")
        (tmp_path / ".dockerignore").write_text("*.pyc\nnode_modules/\n", encoding="utf-8")

        names = _archive(tmp_path, monkeypatch)
        assert "a.py" in names
        assert "a.pyc" not in names
        assert "pkg/a.pyc" not in names
        assert "node_modules/index.js" not in names

    def test_dockerignore_double_star_and_anchored_patterns(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (tmp_path / "a" / "b" / "c").mkdir(parents=True)
        (tmp_path / "a" / "b" / "c" / "cache.tmp").write_text("x", encoding="utf-8")
        (tmp_path / "src" / "build").mkdir(parents=True)
        (tmp_path / "src" / "build" / "out.o").write_text("x", encoding="utf-8")
        (tmp_path / "build").mkdir()
        (tmp_path / "build" / "out.o").write_text("x", encoding="utf-8")
        (tmp_path / "foo" / "a" / "b").mkdir(parents=True)
        (tmp_path / "foo" / "a" / "b" / "bar").write_text("x", encoding="utf-8")
        (tmp_path / "foo" / "a" / "bar").write_text("x", encoding="utf-8")
        (tmp_path / "foo" / "bar").write_text("x", encoding="utf-8")
        (tmp_path / ".dockerignore").write_text("**/*.tmp\n/build\nfoo/**/bar\n", encoding="utf-8")

        names = _archive(tmp_path, monkeypatch)
        assert "a/b/c/cache.tmp" not in names
        assert "build/out.o" not in names
        assert "src/build/out.o" in names
        assert "foo/a/b/bar" not in names
        assert "foo/bar" not in names
        assert "foo/a/bar" not in names

    def test_dockerignore_single_star_does_not_span_directories(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (tmp_path / "foo" / "a" / "b").mkdir(parents=True)
        (tmp_path / "foo" / "a" / "b" / "bar").write_text("x", encoding="utf-8")
        (tmp_path / "foo" / "a" / "bar").write_text("x", encoding="utf-8")
        (tmp_path / ".dockerignore").write_text("foo/*/bar\n", encoding="utf-8")

        names = _archive(tmp_path, monkeypatch)
        assert "foo/a/bar" not in names
        assert "foo/a/b/bar" in names


class TestDeployEnvironmentVariables:
    def test_dry_run_has_no_env_keys_without_sources(self, tmp_path: Path) -> None:
        _write_env(tmp_path)
        assert _dry_run(tmp_path)["env_var_keys"] == []

    def test_env_file_keys_appear_in_dry_run(self, tmp_path: Path) -> None:
        _write_env(tmp_path)
        env_file = tmp_path / "vars.env"
        env_file.write_text("# comment\nKEY1=value1\nEMPTY=\nNOEQ\nKEY2=value2\n")

        assert _dry_run(tmp_path, "--env-file", str(env_file))["env_var_keys"] == [
            "EMPTY",
            "KEY1",
            "KEY2",
        ]

    def test_missing_env_file_fails(self, tmp_path: Path) -> None:
        _write_env(tmp_path)
        result = CliRunner().invoke(
            app,
            [
                "deploy",
                str(tmp_path),
                "--dry-run",
                "--json",
                "--env-file",
                str(tmp_path / "missing"),
            ],
        )
        assert result.exit_code == 1
        assert "Env file not found" in json.loads(result.stdout)["message"]

    def test_env_flags_override_file_on_trigger(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _write_env(tmp_path)
        env_file = tmp_path / "vars.env"
        env_file.write_text("KEY1=file_value\n")
        captured = _stub_remote_build(monkeypatch)

        result = CliRunner().invoke(
            app,
            [
                "deploy",
                str(tmp_path),
                "--json",
                "--env-file",
                str(env_file),
                "--env",
                "KEY1=flag_value",
                "--env",
                "KEY2=new_value",
            ],
        )
        assert result.exit_code == 0, result.output
        assert captured["environment_variables"] == {"KEY1": "flag_value", "KEY2": "new_value"}

    def test_env_flag_invalid_format(self, tmp_path: Path) -> None:
        _write_env(tmp_path)
        result = CliRunner().invoke(
            app, ["deploy", str(tmp_path), "--dry-run", "--json", "--no-env", "--env", "INVALID"]
        )
        assert result.exit_code == 2
        assert "Invalid --env format" in json.loads(result.stdout)["message"]

    def test_build_secret_from_env_reaches_trigger(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _write_env(tmp_path)
        monkeypatch.setenv("GITHUB_TOKEN", "tok")
        captured = _stub_remote_build(monkeypatch)

        result = CliRunner().invoke(
            app,
            [
                "deploy",
                str(tmp_path),
                "--json",
                "--no-env",
                "--secret",
                "id=GITHUB_TOKEN,env=GITHUB_TOKEN",
            ],
        )
        assert result.exit_code == 0, result.output
        assert captured["build_secrets"] == {"GITHUB_TOKEN": "tok"}


class TestRuntimeConfigFile:
    def test_compose_file_is_not_inferred_as_runtime_config(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _write_env(tmp_path, "compose-env")
        (tmp_path / "compose.yaml").write_text(
            "services:\n  main:\n    image: alpine\n",
            encoding="utf-8",
        )
        captured = _stub_remote_build(monkeypatch)

        result = CliRunner().invoke(app, ["deploy", str(tmp_path), "--json", "--no-env"])
        assert result.exit_code == 0, result.output
        assert "runtime_config" not in captured

    def test_runtime_config_flag_reaches_trigger(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _write_env(tmp_path)
        config_path = tmp_path / "runtime.json"
        config_path.write_text(
            json.dumps({"limits": {"startup_timeout_s": 300}}),
            encoding="utf-8",
        )
        captured = _stub_remote_build(monkeypatch)

        result = CliRunner().invoke(
            app,
            [
                "deploy",
                str(tmp_path),
                "--json",
                "--no-env",
                "--runtime-config",
                str(config_path),
            ],
        )
        assert result.exit_code == 0, result.output
        assert captured["runtime_config"] == {"limits": {"startup_timeout_s": 300}}

    def test_runtime_config_uses_sdk_shape(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _write_env(tmp_path)
        config_path = tmp_path / "runtime.json"
        config_path.write_text(
            json.dumps(
                {
                    "resources": {"gpu": {"type": "A10G", "count": 2}},
                    "limits": {"startup_timeout_s": 300},
                }
            ),
            encoding="utf-8",
        )
        captured = _stub_remote_build(monkeypatch)

        result = CliRunner().invoke(
            app,
            ["deploy", str(tmp_path), "--json", "--no-env", "--runtime-config", str(config_path)],
        )
        assert result.exit_code == 0, result.output
        assert captured["runtime_config"] == {
            "resources": {"gpu": {"type": "A10G", "count": 2}},
            "limits": {"startup_timeout_s": 300},
        }

    def test_runtime_config_preserves_null_override(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _write_env(tmp_path)
        config_path = tmp_path / "runtime.json"
        config_path.write_text(json.dumps({"resources": None}), encoding="utf-8")
        captured = _stub_remote_build(monkeypatch)

        result = CliRunner().invoke(
            app,
            ["deploy", str(tmp_path), "--json", "--no-env", "--runtime-config", str(config_path)],
        )
        assert result.exit_code == 0, result.output
        assert captured["runtime_config"] == {"resources": None}

    def test_runtime_config_resolves_compose_project_from_config_directory(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _write_env(tmp_path)
        project = tmp_path / "project"
        project.mkdir()
        (project / "compose.json").write_text('{"services":{"main":{"image":"postgres:16"}}}')
        config_path = tmp_path / "runtime.json"
        config_path.write_text('{"compose":{"document":"project/compose.json","root":"."}}')
        captured = _stub_remote_build(monkeypatch)

        result = CliRunner().invoke(
            app,
            ["deploy", str(tmp_path), "--json", "--no-env", "--runtime-config", str(config_path)],
        )
        assert result.exit_code == 0, result.output
        assert captured["runtime_config"]["compose"]["root"] == {
            "compose_path": "project/compose.json"
        }

    def test_runtime_config_rejects_empty_object(self, tmp_path: Path) -> None:
        _write_env(tmp_path)
        config_path = tmp_path / "runtime.json"
        config_path.write_text("{}", encoding="utf-8")

        result = CliRunner().invoke(
            app,
            [
                "deploy",
                str(tmp_path),
                "--dry-run",
                "--json",
                "--no-env",
                "--runtime-config",
                str(config_path),
            ],
        )
        assert result.exit_code == 2
        assert "at least one field" in json.loads(result.stdout)["message"]

    def test_runtime_config_rejects_unknown_fields(self, tmp_path: Path) -> None:
        _write_env(tmp_path)
        config_path = tmp_path / "runtime.json"
        config_path.write_text(json.dumps({"provider_config": {}}), encoding="utf-8")

        result = CliRunner().invoke(
            app,
            [
                "deploy",
                str(tmp_path),
                "--dry-run",
                "--json",
                "--no-env",
                "--runtime-config",
                str(config_path),
            ],
        )
        assert result.exit_code == 2
        assert "provider_config" in json.loads(result.stdout)["message"]


class TestPlanDeploy:
    def test_no_api_key_error(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test error when no API key is set."""
        from hud.settings import settings

        (tmp_path / "Dockerfile.hud").write_text("FROM python:3.12")
        monkeypatch.setattr(settings, "api_key", None)

        result = CliRunner().invoke(app, ["deploy", str(tmp_path), "--json"])
        assert result.exit_code == 1
        assert json.loads(result.stdout)["error"] == "permission_denied"

    def test_empty_tree_fails_on_missing_name(self, tmp_path: Path) -> None:
        result = CliRunner().invoke(app, ["deploy", str(tmp_path), "--json"])
        assert result.exit_code == 2
        payload = json.loads(result.stdout)
        assert "No environment found" in payload["message"]
        assert payload["suggestion"]

    def test_missing_license_does_not_block_dry_run(self, tmp_path: Path) -> None:
        _write_env(tmp_path)
        (tmp_path / "pyproject.toml").write_text('[project]\nlicense = {file = "LICENSE"}\n')

        assert _dry_run(tmp_path)["success"] is True


def test_hud_row_is_polled_only_while_pending(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_env(tmp_path)
    _stub_remote_build(monkeypatch)
    statuses = ["MIGRATING", "SUCCEEDED"]

    async def request(method: str, url: str, **kwargs: Any) -> dict[str, Any]:
        if url.endswith("/upload-url"):
            return {"upload_url": "https://upload.example", "build_id": "build-1"}
        if url.endswith("/trigger"):
            return {"id": "build-1", "registry_id": _REGISTRY_ID}
        if url.endswith("/status"):
            return {"status": statuses.pop(0)}
        raise AssertionError(url)

    slept = AsyncMock()
    monkeypatch.setattr("hud.utils.platform.make_request", request)
    monkeypatch.setattr("hud.cli.deploy.asyncio.sleep", slept)

    result = CliRunner().invoke(app, ["deploy", str(tmp_path), "--json", "--no-env"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["status"] == "SUCCEEDED"
    assert statuses == []
    slept.assert_awaited()


class TestDeployAsync:
    """Tests for the remote upload/trigger path."""

    def test_upload_url_failure(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test handling of upload URL failure."""
        from hud.utils.exceptions import HudRequestError

        _write_env(tmp_path)
        monkeypatch.setattr(
            "hud.utils.platform.make_request",
            AsyncMock(side_effect=HudRequestError("Unauthorized", status_code=401)),
        )

        result = CliRunner().invoke(app, ["deploy", str(tmp_path), "--json", "--no-env"])
        assert result.exit_code == 1
        payload = json.loads(result.stdout)
        assert payload["error"] == "permission_denied"
        assert payload["message"] == "Unauthorized"

    def test_upload_url_network_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test handling of network error during upload URL fetch."""
        _write_env(tmp_path)
        monkeypatch.setattr(
            "hud.utils.platform.make_request",
            AsyncMock(side_effect=Exception("Network error")),
        )

        result = CliRunner().invoke(app, ["deploy", str(tmp_path), "--json", "--no-env"])
        assert result.exit_code == 1
        assert json.loads(result.stdout)["message"] == "Network error"

    def test_trigger_build_sends_resolved_project(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A resolved placement reaches the platform as project_id."""
        _write_env(tmp_path, "test-env")
        captured = _stub_remote_build(monkeypatch)

        result = CliRunner().invoke(
            app, ["deploy", str(tmp_path), "--json", "--no-env", "--project", _PROJECT_ID]
        )
        assert result.exit_code == 0, result.output
        assert captured["project_id"] == _PROJECT_ID

    def test_trigger_build_omits_project_for_the_team_default(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The zero-config deploy stays byte-identical to before projects existed."""
        _write_env(tmp_path, "test-env")
        captured = _stub_remote_build(monkeypatch)

        result = CliRunner().invoke(app, ["deploy", str(tmp_path), "--json", "--no-env"])
        assert result.exit_code == 0, result.output
        assert "project_id" not in captured


class TestDeployCommand:
    """Tests for deploy_command typer function."""

    def test_command_exists(self) -> None:
        """Test deploy_command function exists and is callable."""
        from hud.cli.deploy import deploy_command

        assert callable(deploy_command)

    def test_command_docstring(self) -> None:
        """Test deploy_command has proper docstring."""
        from hud.cli.deploy import deploy_command

        assert deploy_command.__doc__ is not None
        assert "Deploy" in deploy_command.__doc__


@pytest.fixture(autouse=True)
def authenticated_scope(monkeypatch: pytest.MonkeyPatch) -> None:
    def get(method: str, url: str, **kwargs: object) -> dict[str, Any]:
        if url.endswith("/auth/me"):
            return {
                "user_id": "11111111-1111-4111-8111-111111111111",
                "team_id": "22222222-2222-4222-8222-222222222222",
            }
        if "/projects/" in url:
            return {
                "id": url.rsplit("/", 1)[-1],
                "name": "chosen",
                "capabilities": {"create": True},
            }
        raise AssertionError(url)

    monkeypatch.setattr("hud.utils.platform.make_request_sync", get)
    monkeypatch.setattr("hud.settings.settings.api_key", "test-key")
    monkeypatch.setattr("hud.settings.settings.default_project", None)


@pytest.mark.parametrize(
    "override",
    [
        [],
        ["--registry-id", "33333333-3333-4333-8333-333333333333"],
        ["--project", "44444444-4444-4444-8444-444444444444"],
    ],
)
@pytest.mark.parametrize("stream_status", ["SUCCEEDED", "UNKNOWN", None])
def test_deploy_lifecycle_preserves_links(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, override: list[str], stream_status: str | None
) -> None:
    env = tmp_path / "environment"
    env.mkdir()
    (env / "env.py").write_text('env = Environment("example")')
    (env / "Dockerfile").write_text("FROM python:3.12")
    (env / "pyproject.toml").write_text('[project]\nname="example"\nversion="0"')
    (env / ".env").write_text("SECRET=test-secret-value")
    identity = {
        "user_id": "11111111-1111-4111-8111-111111111111",
        "team_id": "22222222-2222-4222-8222-222222222222",
    }
    registry_id = "55555555-5555-4555-8555-555555555555"
    requests: list[dict[str, Any]] = []

    def get(method: str, url: str, **kwargs):
        if url.endswith("/auth/me"):
            return identity
        if "/registry/" in url:
            return {"id": url.rsplit("/", 1)[-1], "name": "example"}
        if "/projects/" in url:
            return {
                "id": url.rsplit("/", 1)[-1],
                "name": "chosen",
                "capabilities": {"create": True},
            }
        raise AssertionError(url)

    status_reads = 0

    async def request(method: str, url: str, **kwargs):
        nonlocal status_reads
        if url.endswith("/upload-url"):
            return {"upload_url": "https://upload.example", "build_id": "build-1"}
        if url.endswith("/trigger"):
            requests.append(kwargs["json"])
            return {"id": "build-1", "registry_id": kwargs["json"].get("registry_id", registry_id)}
        if url.endswith("/status"):
            status_reads += 1
            return {"status": "SUCCEEDED"}
        raise AssertionError(url)

    monkeypatch.setattr("hud.settings.settings.api_key", "test-key")
    monkeypatch.setattr("hud.utils.platform.make_request_sync", get)
    monkeypatch.setattr("hud.utils.platform.make_request", request)
    monkeypatch.setattr("hud.cli.deploy.httpx.AsyncClient", _FakeHttpxClient)
    websocket = MagicMock()
    websocket.__aiter__.return_value = (
        [
            json.dumps(
                {
                    "type": "complete",
                    "final_status": stream_status,
                    "message": f"Build {stream_status}",
                }
            )
        ]
        if stream_status
        else []
    )
    connection = MagicMock()
    connection.__aenter__.return_value = websocket
    monkeypatch.setattr("hud.cli.deploy.websockets.connect", lambda *a, **k: connection)
    monkeypatch.setattr("hud.cli.deploy.asyncio.sleep", AsyncMock())
    prompts = MagicMock(return_value=True)
    monkeypatch.setattr(HUDConsole, "confirm", prompts)
    first = _TtyCliRunner().invoke(app, ["deploy", str(env), "--json"])
    assert first.exit_code == 0, first.output
    assert json.loads(first.stdout)["registry_id"] == registry_id
    assert "test-secret-value" not in first.stdout
    state = DirectoryState(AuthScope.resolve(PlatformClient.from_settings()), env)
    before = state.load()
    assert before.registry_id is not None
    second = _TtyCliRunner().invoke(app, ["deploy", str(env), "--json", *override])
    assert second.exit_code == 0, second.output
    assert state.load() == before
    assert status_reads == 2
    assert (env / ".hud" / "config.json").exists()
    assert requests[0]["environment_variables"] == {"SECRET": "test-secret-value"}
    if not override:
        assert prompts.call_count == 1
        assert "registry_id" not in requests[-1]
        assert "environment_variables" not in requests[-1]
    elif override[:1] == ["--registry-id"]:
        assert prompts.call_count == 1
        assert "environment_variables" not in requests[-1]


def test_deploy_dry_run_has_no_prompt_or_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from typer.testing import CliRunner

    from hud.cli.__main__ import app

    env = tmp_path / "environment"
    env.mkdir()
    (env / "env.py").write_text('env = Environment("example")')
    (env / "Dockerfile").write_text("FROM python:3.12")
    (env / ".env").write_text("SECRET=hidden")
    legacy = env / ".hud" / "deploy.json"
    legacy.parent.mkdir()
    legacy.write_text('{"registryId":"old","syncEnv":true}')
    monkeypatch.setattr("hud.settings.settings.api_key", "test-key")
    monkeypatch.setattr(HUDConsole, "confirm", lambda *a, **k: pytest.fail("dry-run prompted"))
    result = CliRunner().invoke(app, ["deploy", str(env), "--dry-run", "--json"])
    assert result.exit_code == 0, result.output
    plan = json.loads(result.stdout)
    assert plan["dotenv_pending"] is True
    assert plan["env_var_keys"] == []
    assert plan["dry_run"] is True
    assert plan["name"] == "example"
    assert legacy.read_text() == '{"registryId":"old","syncEnv":true}'
    assert not (legacy.parent / "config.json").exists()
    assert not (Path.home() / ".hud").exists()
