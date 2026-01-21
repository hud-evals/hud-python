from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

from hud.cli.utils.environment import (
    find_dockerfile,
    get_image_name,
    image_exists,
    is_environment_directory,
)

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


def test_is_environment_directory_with_dockerfile_hud(tmp_path: Path):
    """Test that Dockerfile.hud is recognized as a valid environment directory."""
    d = tmp_path / "env"
    d.mkdir()
    assert is_environment_directory(d) is False
    # Use Dockerfile.hud instead of Dockerfile
    (d / "Dockerfile.hud").write_text("FROM python:3.11")
    assert is_environment_directory(d) is False
    (d / "pyproject.toml").write_text("[tool.hud]")
    assert is_environment_directory(d) is True


def test_find_dockerfile_prefers_dockerfile_hud(tmp_path: Path):
    """Test that Dockerfile.hud is preferred over Dockerfile."""
    d = tmp_path / "env"
    d.mkdir()
    # No Dockerfile
    assert find_dockerfile(d) is None
    # Add Dockerfile
    (d / "Dockerfile").write_text("FROM python:3.11")
    assert find_dockerfile(d) == d / "Dockerfile"
    # Add Dockerfile.hud - should now be preferred
    (d / "Dockerfile.hud").write_text("FROM python:3.12")
    assert find_dockerfile(d) == d / "Dockerfile.hud"


def test_find_dockerfile_only_dockerfile_hud(tmp_path: Path):
    """Test that Dockerfile.hud alone is found."""
    d = tmp_path / "env"
    d.mkdir()
    (d / "Dockerfile.hud").write_text("FROM python:3.11")
    assert find_dockerfile(d) == d / "Dockerfile.hud"


@patch("subprocess.run")
def test_image_exists_true(mock_run):
    mock_run.return_value = MagicMock(returncode=0)
    assert image_exists("img") is True
