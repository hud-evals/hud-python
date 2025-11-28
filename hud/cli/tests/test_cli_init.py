"""Tests for hud.cli.__init__ module."""

from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path
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
        # When no args, typer shows help but exits with code 2 (usage error)
        assert result.exit_code == 2
        assert "Usage:" in result.output

    def test_analyze_docker_image(self) -> None:
        """Test analyze command with Docker image."""
        with patch("hud.cli.asyncio.run") as mock_run:
            result = runner.invoke(app, ["analyze", "test-image:latest"])
            assert result.exit_code == 0
            mock_run.assert_called_once()
            # Get the coroutine that was passed to asyncio.run
            coro = mock_run.call_args[0][0]
            assert coro.__name__ == "analyze_from_metadata"

    def test_analyze_with_docker_args(self) -> None:
        """Test analyze command with additional Docker arguments."""
        with patch("hud.cli.asyncio.run") as mock_run:
            # Docker args need to come after -- to avoid being parsed as CLI options
            result = runner.invoke(
                app, ["analyze", "test-image", "--", "-e", "KEY=value", "-p", "8080:8080"]
            )
            assert result.exit_code == 0
            mock_run.assert_called_once()

    def test_analyze_with_config_file(self) -> None:
        """Test analyze command with config file."""
        import os

        fd, temp_path = tempfile.mkstemp(suffix=".json")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump({"test": {"command": "python", "args": ["server.py"]}}, f)

            with patch("hud.cli.asyncio.run") as mock_run:
                # Need to provide a dummy positional arg since params is required
                result = runner.invoke(app, ["analyze", "dummy", "--config", temp_path])
                assert result.exit_code == 0
                mock_run.assert_called_once()
                coro = mock_run.call_args[0][0]
                assert coro.__name__ == "analyze_environment_from_config"
        finally:
            try:
                os.unlink(temp_path)
            except Exception:
                logger.exception("Error deleting temp file")

    def test_analyze_with_cursor_server(self) -> None:
        """Test analyze command with Cursor server."""
        with patch("hud.cli.parse_cursor_config") as mock_parse:
            mock_parse.return_value = (["python", "server.py"], None)
            with patch("hud.cli.asyncio.run") as mock_run:
                # Need to provide a dummy positional arg since params is required
                result = runner.invoke(app, ["analyze", "dummy", "--cursor", "test-server"])
                assert result.exit_code == 0
                mock_run.assert_called_once()

    def test_analyze_cursor_server_not_found(self) -> None:
        """Test analyze with non-existent Cursor server."""
        with patch("hud.cli.parse_cursor_config") as mock_parse:
            mock_parse.return_value = (None, "Server 'test' not found")
            result = runner.invoke(app, ["analyze", "--cursor", "test"])
            assert result.exit_code == 1
            assert "Server 'test' not found" in result.output

    def test_analyze_no_arguments_shows_error(self) -> None:
        """Test analyze without arguments shows error."""
        result = runner.invoke(app, ["analyze"])
        assert result.exit_code == 1
        assert "Error" in result.output

    def test_analyze_output_formats(self) -> None:
        """Test analyze with different output formats."""
        for format_type in ["interactive", "json", "markdown"]:
            with patch("hud.cli.asyncio.run"):
                result = runner.invoke(app, ["analyze", "test-image", "--format", format_type])
                assert result.exit_code == 0

    def test_debug_docker_image(self) -> None:
        """Test debug command with Docker image."""
        with patch("hud.cli.asyncio.run") as mock_run:
            mock_run.return_value = 5  # All phases completed
            result = runner.invoke(app, ["debug", "test-image:latest"])
            assert result.exit_code == 0
            mock_run.assert_called_once()

    def test_debug_with_max_phase(self) -> None:
        """Test debug command with max phase limit."""
        with patch("hud.cli.asyncio.run") as mock_run:
            mock_run.return_value = 3  # Completed 3 phases
            result = runner.invoke(app, ["debug", "test-image", "--max-phase", "3"])
            assert result.exit_code == 0  # Exit code 0 when phases_completed == max_phase

    def test_debug_with_config_file(self) -> None:
        """Test debug command with config file."""
        import os

        fd, temp_path = tempfile.mkstemp(suffix=".json")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump({"test": {"command": "python", "args": ["server.py"]}}, f)

            with patch("hud.cli.asyncio.run") as mock_run:
                mock_run.return_value = 5
                # Need to provide a dummy positional arg since params is required
                result = runner.invoke(app, ["debug", "dummy", "--config", temp_path])
                assert result.exit_code == 0
        finally:
            try:
                os.unlink(temp_path)
            except Exception:
                logger.exception("Error deleting temp file")

    def test_debug_with_cursor_server(self) -> None:
        """Test debug command with Cursor server."""
        with patch("hud.cli.parse_cursor_config") as mock_parse:
            mock_parse.return_value = (["python", "server.py"], None)
            with patch("hud.cli.asyncio.run") as mock_run:
                mock_run.return_value = 5
                # Need to provide a dummy positional arg since params is required
                result = runner.invoke(app, ["debug", "dummy", "--cursor", "test-server"])
                assert result.exit_code == 0

    def test_debug_no_arguments_shows_error(self) -> None:
        """Test debug without arguments shows error."""
        result = runner.invoke(app, ["debug"])
        assert result.exit_code == 1
        assert "Error" in result.output

    def test_cursor_list_command(self) -> None:
        """Test cursor-list command."""
        with patch("hud.cli.list_cursor_servers") as mock_list:
            mock_list.return_value = (["server1", "server2"], None)
            with patch("hud.cli.get_cursor_config_path") as mock_path:
                mock_path.return_value = Path("/home/user/.cursor/mcp.json")
                with patch("pathlib.Path.exists") as mock_exists:
                    mock_exists.return_value = True
                    with patch("builtins.open", create=True) as mock_open:
                        mock_open.return_value.__enter__.return_value.read.return_value = (
                            json.dumps(
                                {
                                    "mcpServers": {
                                        "server1": {"command": "python", "args": ["srv1.py"]},
                                        "server2": {"command": "node", "args": ["srv2.js"]},
                                    }
                                }
                            )
                        )
                        result = runner.invoke(app, ["cursor-list"])
                        assert result.exit_code == 0
                        assert "Available Servers" in result.output

    def test_cursor_list_no_servers(self) -> None:
        """Test cursor-list with no servers."""
        with patch("hud.cli.list_cursor_servers") as mock_list:
            mock_list.return_value = ([], None)
            result = runner.invoke(app, ["cursor-list"])
            assert result.exit_code == 0
            assert "No servers found" in result.output

    def test_cursor_list_error(self) -> None:
        """Test cursor-list with error."""
        with patch("hud.cli.list_cursor_servers") as mock_list:
            mock_list.return_value = (None, "Config not found")
            result = runner.invoke(app, ["cursor-list"])
            assert result.exit_code == 1
            assert "Config not found" in result.output

    def test_version_command(self) -> None:
        """Test version command."""
        import re

        # Strip ANSI escape codes
        ansi_escape = re.compile(r"\x1b\[[0-9;]*m")
        with patch("hud.__version__", "1.2.3"):
            result = runner.invoke(app, ["version"])
            assert result.exit_code == 0
            clean_output = ansi_escape.sub("", result.output)
            assert "1.2.3" in clean_output

    def test_version_import_error(self) -> None:
        """Test version command when version unavailable."""
        # Patch the specific import of __version__ from hud
        with patch.dict("sys.modules", {"hud": None}):
            result = runner.invoke(app, ["version"])
            assert result.exit_code == 0
            assert "HUD CLI version: unknown" in result.output

    def test_mcp_command(self) -> None:
        """Test mcp server command."""
        # MCP command has been removed from the CLI
        result = runner.invoke(app, ["mcp"])
        assert result.exit_code == 2  # Command not found

    def test_help_command(self) -> None:
        """Test help command shows proper info."""
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "HUD CLI for MCP environment analysis" in result.output
        assert "analyze" in result.output
        assert "debug" in result.output
        # assert "mcp" in result.output  # mcp command has been removed


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
                # Should print the header panel
                # Check that either console was used or app was called
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
                # Should exit with code 2 (missing command)
                assert exc_info.value.code == 2
                # Should print Quick Start guide before exiting
                assert any("Quick Start" in str(call) for call in mock_console.print.call_args_list)
        finally:
            sys.argv = original_argv


if __name__ == "__main__":
    pytest.main([__file__])
