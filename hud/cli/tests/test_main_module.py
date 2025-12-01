"""Tests for hud.cli.__main__ module."""

from __future__ import annotations

import os
import subprocess
import sys


class TestMainModule:
    """Tests for the CLI __main__ module."""

    def test_main_module_imports_correctly(self):
        """Test that __main__.py imports correctly."""
        # Simply importing the module should work without errors
        import hud.cli.__main__

        # Verify the module has the expected attributes
        assert hasattr(hud.cli.__main__, "main")

    def test_main_module_executes(self):
        """Test that running the module as main executes correctly."""
        # Use subprocess to run the module as __main__ and check it doesn't crash
        # Use --version flag for a quick, deterministic test that doesn't require user input
        env = {**os.environ, "HUD_SKIP_VERSION_CHECK": "1"}
        result = subprocess.run(
            [sys.executable, "-m", "hud.cli", "--version"],
            capture_output=True,
            text=True,
            timeout=30,
            env=env,
        )

        # Should exit successfully with version info
        assert result.returncode == 0
        assert "version" in result.stdout.lower() or "hud" in result.stdout.lower()
