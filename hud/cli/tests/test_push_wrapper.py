from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
import typer

from hud.cli.push import push_environment

if TYPE_CHECKING:
    from pathlib import Path


@patch("hud.cli.push.ensure_built")
@patch("hud.cli.push.HUDConsole")
@patch("hud.cli.push.subprocess.run")
def test_push_environment_missing_lock_raises(mock_run, mock_console, _ensure, tmp_path: Path):
    # No hud.lock.yaml → Exit(1)
    with pytest.raises(typer.Exit):
        push_environment(
            directory=str(tmp_path), image=None, tag=None, sign=False, yes=True, verbose=False
        )
