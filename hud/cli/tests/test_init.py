"""Tests for ``hud init`` scaffolding."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import typer

from hud.cli.init import init_command

if TYPE_CHECKING:
    from pathlib import Path


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
