"""Tests for hud.cli.__main__."""

from __future__ import annotations

import os
import subprocess
import sys


class TestMainModule:
    def test_main_module_imports_correctly(self) -> None:
        import hud.cli.__main__

        assert hasattr(hud.cli.__main__, "main")

    def test_main_module_executes(self) -> None:
        env = {**os.environ, "HUD_SKIP_VERSION_CHECK": "1"}
        result = subprocess.run(
            [sys.executable, "-m", "hud.cli", "--version"],
            capture_output=True,
            text=True,
            timeout=30,
            env=env,
        )
        assert result.returncode == 0
        assert "version" in result.stdout.lower() or "hud" in result.stdout.lower()
