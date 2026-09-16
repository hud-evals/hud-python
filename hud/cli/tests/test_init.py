"""Tests for ``hud init``."""

from __future__ import annotations

import io
import json
import tarfile
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import httpx
import pytest
from typer.testing import CliRunner

from hud.cli import CliError
from hud.cli import init as init_module
from hud.cli.__main__ import app
from hud.cli.init import init_command

if TYPE_CHECKING:
    from pathlib import Path


def _sdk_archive(source: str, files: dict[str, bytes]) -> bytes:
    payload = io.BytesIO()
    with tarfile.open(fileobj=payload, mode="w:gz") as archive:
        for name, content in files.items():
            info = tarfile.TarInfo(f"hud-python-release/environments/{source}/{name}")
            info.size = len(content)
            archive.addfile(info, io.BytesIO(content))
    return payload.getvalue()


@pytest.fixture
def installed_sdk(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Make ``hud init`` behave as an installed release: no ``environments/`` beside the package."""
    monkeypatch.setattr(init_module, "__file__", str(tmp_path / "site" / "hud" / "cli" / "init.py"))
    monkeypatch.setattr(init_module, "__version__", "1.2.3")


def _init(
    tmp_path: Path,
    name: str | None,
    preset: str | None,
    *,
    force: bool = False,
    dry_run: bool = False,
) -> dict[str, object]:
    """Call the command as the CLI would, with every option bound."""
    return init_command(
        name=name, directory=str(tmp_path), force=force, dry_run=dry_run, preset=preset
    )


def _release(payload: bytes) -> MagicMock:
    return MagicMock(
        return_value=httpx.Response(
            200, content=payload, request=httpx.Request("GET", "https://example.test")
        )
    )


# ─── choosing the example ───────────────────────────────────────────────


def test_name_alone_uses_the_coding_example(tmp_path: Path) -> None:
    _init(tmp_path, "my-cool-env", None)

    target = tmp_path / "my-cool-env"
    assert (target / "README.md").exists()
    assert 'Environment(name="my-cool-env")' in (target / "env.py").read_text()


def test_example_without_name_uses_its_id_as_directory(tmp_path: Path) -> None:
    _init(tmp_path, None, "coding")
    assert (tmp_path / "coding" / "env.py").exists()


def test_name_overrides_the_example_directory_and_env_name(tmp_path: Path) -> None:
    _init(tmp_path, "custom", "cua")

    assert 'Environment(name="custom")' in (tmp_path / "custom" / "env.py").read_text()
    assert not (tmp_path / "cua").exists()


def test_blank_materializes_a_runnable_example(tmp_path: Path) -> None:
    _init(tmp_path, "berry", "blank")

    target = tmp_path / "berry"
    assert {path.name for path in target.iterdir()} == {
        "README.md",
        "pyproject.toml",
        "env.py",
        "tasks.py",
        "Dockerfile.hud",
        ".dockerignore",
    }
    assert 'Environment(name="berry")' in (target / "env.py").read_text()
    assert "package = false" in (target / "pyproject.toml").read_text()
    assert 'CMD ["uv", "run", "hud", "serve"' in (target / "Dockerfile.hud").read_text()
    assert ".venv" in (target / ".dockerignore").read_text()


def test_env_name_is_normalized(tmp_path: Path) -> None:
    _init(tmp_path, "My Cool_Env", "blank")
    assert 'Environment(name="my-cool-env")' in (tmp_path / "My Cool_Env" / "env.py").read_text()


def test_without_name_or_example_errors_when_noninteractive(tmp_path: Path) -> None:
    with pytest.raises(CliError, match="Nothing to create"):
        _init(tmp_path, None, None)


def test_unknown_example_is_a_usage_error(tmp_path: Path) -> None:
    with pytest.raises(CliError, match="Unknown example environment 'does-not-exist'"):
        _init(tmp_path, None, "does-not-exist")


def test_dry_run_never_prompts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(init_module.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(init_module.sys.stdout, "isatty", lambda: True)
    plan = _init(tmp_path, "thing", None, dry_run=True)
    assert plan == {
        "dry_run": True,
        "action": "init",
        "path": str(tmp_path / "thing"),
        "preset": "coding",
    }
    assert not (tmp_path / "thing").exists()


# ─── destination safety ─────────────────────────────────────────────────


def test_refuses_to_clobber_nonempty_directory(tmp_path: Path) -> None:
    target = tmp_path / "taken"
    target.mkdir()
    (target / "precious.txt").write_text("data")

    with pytest.raises(CliError, match="not empty"):
        _init(tmp_path, "taken", "blank")
    assert (target / "precious.txt").read_text() == "data"


def test_force_overwrites_existing_files(tmp_path: Path) -> None:
    target = tmp_path / "env"
    target.mkdir()
    (target / "env.py").write_text("old")

    _init(tmp_path, "env", "blank", force=True)
    assert "Environment" in (target / "env.py").read_text()


def test_refuses_to_copy_over_a_symlinked_file(tmp_path: Path) -> None:
    target = tmp_path / "project"
    target.mkdir()
    outside = tmp_path / "outside.py"
    outside.write_text("original")
    (target / "env.py").symlink_to(outside)

    with pytest.raises(CliError, match="symlinks"):
        _init(tmp_path, "project", "blank", force=True)
    assert outside.read_text() == "original"


def test_refuses_a_symlinked_destination(tmp_path: Path) -> None:
    target = tmp_path / "project"
    target.symlink_to(tmp_path / "outside", target_is_directory=True)

    with pytest.raises(CliError, match="symlink"):
        _init(tmp_path, "project", "blank")


def test_failure_is_a_json_error_and_removes_the_partial_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(source: Path, target: Path, **_: object) -> None:
        target.mkdir()
        (target / "partial").touch()
        raise OSError("copy failed")

    monkeypatch.setattr(init_module.shutil, "copytree", fail)
    result = CliRunner().invoke(app, ["init", "example", "--dir", str(tmp_path), "--json"])
    assert result.exit_code != 0
    assert "copy failed" in json.loads(result.stdout)["message"]
    assert not (tmp_path / "example").exists()


# ─── installed SDK: the example comes from the release archive ─────────


def test_local_copy_skips_caches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repository = tmp_path / "repository"
    source = repository / "environments" / "coding"
    source.mkdir(parents=True)
    (source / "env.py").write_text('env = Environment(name="coding")')
    (source / ".venv").mkdir()
    (source / ".venv" / "ignored").write_text("ignored")
    (source / "__pycache__").mkdir()
    (source / "__pycache__" / "ignored.pyc").write_bytes(b"ignored")
    monkeypatch.setattr(init_module, "__file__", str(repository / "hud" / "cli" / "init.py"))

    _init(tmp_path, "coding", "coding")

    target = tmp_path / "coding"
    assert (target / "env.py").read_text() == 'env = Environment(name="coding")'
    assert not (target / ".venv").exists()
    assert not (target / "__pycache__").exists()


def test_release_archive_is_extracted_for_the_installed_version(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, installed_sdk: None
) -> None:
    get = _release(
        _sdk_archive(
            "coding",
            {"README.md": b"# Coding", "scripts/run.sh": b"#!/bin/sh\n", "env.py": b"x"},
        )
    )
    monkeypatch.setattr(init_module.httpx, "get", get)

    _init(tmp_path, "coding", "coding")

    assert get.call_args.args[0].endswith("/tar.gz/refs/tags/v1.2.3")
    target = tmp_path / "coding"
    assert (target / "README.md").read_text() == "# Coding"
    assert (target / "scripts" / "run.sh").read_text() == "#!/bin/sh\n"


def test_release_archive_paths_cannot_escape_the_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, installed_sdk: None
) -> None:
    monkeypatch.setattr(
        init_module.httpx, "get", _release(_sdk_archive("coding", {"../../escape": b"unsafe"}))
    )

    with pytest.raises(CliError, match="unsafe path"):
        _init(tmp_path, "project", "coding")
    assert not (tmp_path / "escape").exists()
    assert not (tmp_path / "project").exists()


def test_development_version_needs_a_checkout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, installed_sdk: None
) -> None:
    monkeypatch.setattr(init_module, "__version__", "1.2.3.dev0")
    with pytest.raises(CliError, match="development version"):
        _init(tmp_path, "project", "coding")
