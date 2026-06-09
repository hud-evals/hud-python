"""Docker CLI helpers: daemon guard and per-env ``.env`` loading."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest
import typer

from hud.cli.utils import docker

if TYPE_CHECKING:
    from pathlib import Path


def test_load_env_vars_for_dir(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text("KEY=value\nOTHER=2\n", encoding="utf-8")
    assert docker.load_env_vars_for_dir(tmp_path) == {"KEY": "value", "OTHER": "2"}


def test_load_env_vars_missing_is_empty(tmp_path: Path) -> None:
    assert docker.load_env_vars_for_dir(tmp_path) == {}


def test_require_docker_running_passes_when_daemon_up() -> None:
    with (
        patch("shutil.which", return_value="/usr/bin/docker"),
        patch("subprocess.run") as mock_run,
    ):
        mock_run.return_value = MagicMock(returncode=0)
        docker.require_docker_running()


def test_require_docker_running_exits_without_cli() -> None:
    with patch("shutil.which", return_value=None), pytest.raises(typer.Exit):
        docker.require_docker_running()
