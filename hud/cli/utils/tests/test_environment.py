from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

from hud.cli.utils.environment import get_image_name, image_exists, is_environment_directory

if TYPE_CHECKING:
    from pathlib import Path


def test_get_image_name_override():
    name, source = get_image_name(".", image_override="custom:dev")
    assert name == "custom:dev" and source == "override"


def test_get_image_name_auto(tmp_path: Path):
    env = tmp_path / "my_env"
    env.mkdir()
    # Provide Dockerfile and pyproject to pass directory check later if used
    (env / "Dockerfile").write_text("FROM python:3.11")
    (env / "pyproject.toml").write_text("[tool.hud]\nimage='x'")
    name, source = get_image_name(env)
    # Because pyproject exists with image key, source should be cache
    assert source == "cache"
    assert name == "x"


def test_is_environment_directory(tmp_path: Path):
    d = tmp_path / "env"
    d.mkdir()
    assert is_environment_directory(d) is False
    (d / "Dockerfile").write_text("FROM python:3.11")
    assert is_environment_directory(d) is False
    (d / "pyproject.toml").write_text("[tool.hud]")
    assert is_environment_directory(d) is True


@patch("subprocess.run")
def test_image_exists_true(mock_run):
    mock_run.return_value = MagicMock(returncode=0)
    assert image_exists("img") is True
