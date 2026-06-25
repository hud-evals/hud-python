"""Tests for ``hud init`` scaffolding."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import typer

from hud.cli import init as init_module
from hud.cli.init import init_command
from hud.cli.presets import EnvironmentPreset

if TYPE_CHECKING:
    from pathlib import Path


def _fake_clone(record: dict[str, object]):
    """A stand-in for ``materialize_preset`` that records its args and writes a
    marker file instead of hitting the network."""

    def clone(preset: EnvironmentPreset, target: Path) -> None:
        record["preset"] = preset
        record["target"] = target
        target.mkdir(parents=True, exist_ok=True)
        (target / "README.md").write_text("# template")

    return clone


def test_init_scaffolds_a_runnable_package(tmp_path: Path) -> None:
    init_command(name="my-cool-env", directory=str(tmp_path), force=False, preset=None)

    target = tmp_path / "my-cool-env"
    assert {p.name for p in target.iterdir()} == {
        "pyproject.toml",
        "env.py",
        "tasks.py",
        "Dockerfile.hud",
    }

    env_py = (target / "env.py").read_text()
    assert 'Environment(name="my_cool_env")' in env_py
    assert (target / "tasks.py").read_text().startswith('"""')
    assert 'name = "my-cool-env"' in (target / "pyproject.toml").read_text()

    pyproject = (target / "pyproject.toml").read_text()
    assert "package = false" in pyproject
    assert "[build-system]" not in pyproject

    dockerfile = (target / "Dockerfile.hud").read_text()
    assert 'CMD ["uv", "run", "hud", "serve"' in dockerfile
    assert '"dev"' not in dockerfile


def test_init_blank_preset_writes_local_scaffold_without_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # 'blank' is the bundled local scaffold: it must never hit the network.
    def _boom(*args: object, **kwargs: object) -> None:
        raise AssertionError("materialize_preset should not be called for blank")

    monkeypatch.setattr(init_module, "materialize_preset", _boom)

    init_command(name="berry", directory=str(tmp_path), force=False, preset="blank")

    target = tmp_path / "berry"
    assert {p.name for p in target.iterdir()} == {
        "pyproject.toml",
        "env.py",
        "tasks.py",
        "Dockerfile.hud",
    }
    assert 'Environment(name="berry")' in (target / "env.py").read_text()


def test_init_refuses_to_clobber_nonempty_directory(tmp_path: Path) -> None:
    target = tmp_path / "taken"
    target.mkdir()
    (target / "precious.txt").write_text("data")

    with pytest.raises(typer.Exit):
        init_command(name="taken", directory=str(tmp_path), force=False, preset=None)

    assert (target / "precious.txt").read_text() == "data"


def test_init_force_overwrites_existing_files(tmp_path: Path) -> None:
    target = tmp_path / "env"
    target.mkdir()
    (target / "env.py").write_text("old")

    init_command(name="env", directory=str(tmp_path), force=True, preset=None)

    assert "Environment" in (target / "env.py").read_text()


def test_init_without_name_clones_preset_into_repo_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    record: dict[str, object] = {}
    monkeypatch.setattr(init_module, "materialize_preset", _fake_clone(record))

    init_command(name=None, directory=str(tmp_path), force=False, preset="browser")

    target = tmp_path / "hud-browser"
    assert (target / "README.md").read_text() == "# template"
    assert record["target"] == target
    assert isinstance(record["preset"], EnvironmentPreset)
    assert record["preset"].repo == "hud-browser"


def test_init_name_overrides_preset_repo_as_target_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    record: dict[str, object] = {}
    monkeypatch.setattr(init_module, "materialize_preset", _fake_clone(record))

    init_command(name="custom", directory=str(tmp_path), force=False, preset="cua")

    assert (tmp_path / "custom" / "README.md").exists()
    assert not (tmp_path / "cua-template").exists()


def test_init_without_name_or_preset_errors_when_noninteractive(tmp_path: Path) -> None:
    # No TTY (pytest), no name, no preset: nothing identifies a target, so bail out
    # loudly rather than guessing.
    with pytest.raises(typer.Exit):
        init_command(name=None, directory=str(tmp_path), force=False, preset=None)


def test_init_rejects_unknown_preset(tmp_path: Path) -> None:
    with pytest.raises(typer.Exit):
        init_command(name=None, directory=str(tmp_path), force=False, preset="does-not-exist")
