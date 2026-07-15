"""Tests for the hud._entry console-script guard."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import hud._entry


def test_console_entry_runs_the_cli() -> None:
    code = "import sys; sys.argv = ['hud', '--help']; from hud._entry import main; main()"
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env={**os.environ, "HUD_SKIP_VERSION_CHECK": "1"},
        check=False,
    )
    assert result.returncode == 0
    assert "Usage" in result.stdout


def test_gutted_install_reports_recovery_command(tmp_path: Path) -> None:
    # Reproduce what `pip install -U hud-python` leaves behind when upgrading
    # across the rename: pip deletes every file on the old distribution's
    # RECORD, so hud/ survives only as a namespace package holding files that
    # are new in the renamed release — such as _entry.py.
    site = tmp_path / "site-packages"
    (site / "hud").mkdir(parents=True)
    shutil.copy(Path(hud._entry.__file__), site / "hud" / "_entry.py")

    result = subprocess.run(
        [sys.executable, "-S", "-c", "from hud._entry import main; main()"],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(site)},
        check=False,
    )
    assert result.returncode == 1
    assert "pip install --force-reinstall --no-deps hud" in result.stderr
    assert "Traceback" not in result.stderr
