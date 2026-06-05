"""Tests for hud.cli module commands."""

from __future__ import annotations

import logging
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from hud.cli import app, main

runner = CliRunner()

logger = logging.getLogger(__name__)


class TestCLICommands:
    """Test CLI command handling."""

    def test_main_shows_help_when_no_args(self) -> None:
        """Test that main() shows help when no arguments provided."""
        result = runner.invoke(app)
        assert result.exit_code == 2
        assert "Usage:" in result.output

    def test_version_command(self) -> None:
        """Test version command."""
        import re

        ansi_escape = re.compile(r"\x1b\[[0-9;]*m")
        with patch("hud.__version__", "1.2.3"):
            result = runner.invoke(app, ["version"])
            assert result.exit_code == 0
            clean_output = ansi_escape.sub("", result.output)
            assert "1.2.3" in clean_output

    def test_version_import_error(self) -> None:
        """Test version command when version unavailable."""
        import re

        ansi_escape = re.compile(r"\x1b\[[0-9;]*m")
        with patch.dict("sys.modules", {"hud": None}):
            result = runner.invoke(app, ["version"])
            assert result.exit_code == 0
            clean_output = ansi_escape.sub("", result.output)
            assert "HUD CLI version: unknown" in clean_output

    def test_mcp_command(self) -> None:
        """Test mcp server command."""
        result = runner.invoke(app, ["mcp"])
        assert result.exit_code == 2

    def test_help_command(self) -> None:
        """Test help command lists v6 commands."""
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "eval" in result.output
        assert "build" in result.output


class TestMainFunction:
    """Test the main() function specifically."""

    def test_main_with_help_flag(self) -> None:
        """Test main() with --help flag."""
        import sys

        original_argv = sys.argv
        try:
            sys.argv = ["hud", "--help"]
            with (
                patch("hud.cli.console") as mock_console,
                patch("hud.cli.app") as mock_app,
            ):
                main()
                assert mock_console.print.called or mock_app.called
        finally:
            sys.argv = original_argv

    def test_main_with_no_args(self) -> None:
        """Test main() with no arguments."""
        import sys

        original_argv = sys.argv
        try:
            sys.argv = ["hud"]
            with patch("hud.cli.console") as mock_console:
                with pytest.raises(SystemExit) as exc_info:
                    main()
                assert exc_info.value.code == 2
                assert any("Quick Start" in str(call) for call in mock_console.print.call_args_list)
        finally:
            sys.argv = original_argv


if __name__ == "__main__":
    pytest.main([__file__])
