"""Tests for hud.cli.cursor module."""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import mock_open, patch

import pytest

from hud.cli.cursor import get_cursor_config_path, list_cursor_servers, parse_cursor_config


class TestParseCursorConfig:
    """Test Cursor config parsing."""

    def test_parse_cursor_config_success(self) -> None:
        """Test successful parsing of Cursor config."""
        config_data = {
            "mcpServers": {
                "test-server": {
                    "command": "python",
                    "args": ["server.py", "--port", "8080"],
                    "env": {"KEY": "value"},
                }
            }
        }

        with (
            patch("pathlib.Path.exists", return_value=True),
            patch("builtins.open", mock_open(read_data=json.dumps(config_data))),
        ):
            command, error = parse_cursor_config("test-server")
            assert error is None
            assert command == ["python", "server.py", "--port", "8080"]

    def test_parse_cursor_config_not_found(self) -> None:
        """Test parsing when config file doesn't exist."""
        with patch("pathlib.Path.exists", return_value=False):
            command, error = parse_cursor_config("test-server")
            assert command is None
            assert error is not None
            assert "Cursor config not found" in error

    def test_parse_cursor_config_server_not_found(self) -> None:
        """Test parsing when server doesn't exist in config."""
        config_data = {"mcpServers": {"other-server": {"command": "node", "args": ["server.js"]}}}

        with (
            patch("pathlib.Path.exists", return_value=True),
            patch("builtins.open", mock_open(read_data=json.dumps(config_data))),
        ):
            command, error = parse_cursor_config("test-server")
            assert command is None
            assert error is not None
            assert "Server 'test-server' not found" in error
            assert "Available: other-server" in error

    def test_parse_cursor_config_reloaderoo(self) -> None:
        """Test parsing config with reloaderoo wrapper."""
        config_data = {
            "mcpServers": {
                "test-server": {
                    "command": "npx",
                    "args": ["reloaderoo", "--watch", "src", "--", "python", "server.py"],
                }
            }
        }

        with (
            patch("pathlib.Path.exists", return_value=True),
            patch("builtins.open", mock_open(read_data=json.dumps(config_data))),
        ):
            command, error = parse_cursor_config("test-server")
            assert error is None
            # Should extract command after --
            assert command == ["python", "server.py"]

    def test_parse_cursor_config_reloaderoo_no_dash(self) -> None:
        """Test parsing reloaderoo without -- separator."""
        config_data = {
            "mcpServers": {
                "test-server": {"command": "npx", "args": ["reloaderoo", "python", "server.py"]}
            }
        }

        with (
            patch("pathlib.Path.exists", return_value=True),
            patch("builtins.open", mock_open(read_data=json.dumps(config_data))),
        ):
            command, error = parse_cursor_config("test-server")
            assert error is None
            # Should return full command since no -- found
            assert command == ["npx", "reloaderoo", "python", "server.py"]

    def test_parse_cursor_config_windows_path(self) -> None:
        """Test parsing with Windows user profile path."""
        config_data = {"mcpServers": {"test": {"command": "cmd"}}}

        # First path doesn't exist, try Windows path
        with (
            patch("pathlib.Path.exists", side_effect=[False, True]),
            patch.dict(os.environ, {"USERPROFILE": "C:\\Users\\Test"}),
            patch("builtins.open", mock_open(read_data=json.dumps(config_data))),
        ):
            command, error = parse_cursor_config("test")
            assert error is None
            assert command == ["cmd"]

    def test_parse_cursor_config_json_error(self) -> None:
        """Test parsing with invalid JSON."""
        with (
            patch("pathlib.Path.exists", return_value=True),
            patch("builtins.open", mock_open(read_data="invalid json")),
        ):
            command, error = parse_cursor_config("test-server")
            assert command is None
            assert error is not None
            assert "Error reading config" in error

    def test_parse_cursor_config_no_command(self) -> None:
        """Test parsing server with no command."""
        config_data = {"mcpServers": {"test-server": {"args": ["--port", "8080"]}}}

        with (
            patch("pathlib.Path.exists", return_value=True),
            patch("builtins.open", mock_open(read_data=json.dumps(config_data))),
        ):
            command, error = parse_cursor_config("test-server")
            assert error is None
            assert command == ["", "--port", "8080"]  # Empty command


class TestListCursorServers:
    """Test listing Cursor servers."""

    def test_list_cursor_servers_success(self) -> None:
        """Test successful listing of servers."""
        config_data = {
            "mcpServers": {
                "server1": {"command": "python"},
                "server2": {"command": "node"},
                "server3": {"command": "ruby"},
            }
        }

        with (
            patch("pathlib.Path.exists", return_value=True),
            patch("builtins.open", mock_open(read_data=json.dumps(config_data))),
        ):
            servers, error = list_cursor_servers()
            assert error is None
            assert servers == ["server1", "server2", "server3"]

    def test_list_cursor_servers_empty(self) -> None:
        """Test listing when no servers configured."""
        config_data = {"mcpServers": {}}

        with (
            patch("pathlib.Path.exists", return_value=True),
            patch("builtins.open", mock_open(read_data=json.dumps(config_data))),
        ):
            servers, error = list_cursor_servers()
            assert error is None
            assert servers == []

    def test_list_cursor_servers_no_mcp_section(self) -> None:
        """Test listing when mcpServers section missing."""
        config_data = {"otherConfig": {}}

        with (
            patch("pathlib.Path.exists", return_value=True),
            patch("builtins.open", mock_open(read_data=json.dumps(config_data))),
        ):
            servers, error = list_cursor_servers()
            assert error is None
            assert servers == []

    def test_list_cursor_servers_file_not_found(self) -> None:
        """Test listing when config file doesn't exist."""
        with patch("pathlib.Path.exists", return_value=False):
            servers, error = list_cursor_servers()
            assert servers is None
            assert error is not None
            assert "Cursor config not found" in error

    def test_list_cursor_servers_windows_path(self) -> None:
        """Test listing with Windows path fallback."""
        config_data = {"mcpServers": {"winserver": {"command": "cmd"}}}

        # First path doesn't exist, second (Windows) does
        with (
            patch("pathlib.Path.exists", side_effect=[False, True]),
            patch.dict(os.environ, {"USERPROFILE": "C:\\Users\\Test"}),
            patch("builtins.open", mock_open(read_data=json.dumps(config_data))),
        ):
            servers, error = list_cursor_servers()
            assert error is None
            assert servers == ["winserver"]

    def test_list_cursor_servers_read_error(self) -> None:
        """Test listing with file read error."""
        with (
            patch("pathlib.Path.exists", return_value=True),
            patch("builtins.open", side_effect=PermissionError("Access denied")),
        ):
            servers, error = list_cursor_servers()
            assert servers is None
            assert error is not None
            assert "Error reading config" in error
            assert "Access denied" in error


class TestGetCursorConfigPath:
    """Test getting Cursor config path."""

    def test_get_cursor_config_path_unix(self) -> None:
        """Test getting config path on Unix-like systems."""
        with (
            patch("pathlib.Path.home", return_value=Path("/home/user")),
            patch("pathlib.Path.exists", return_value=True),
        ):
            path = get_cursor_config_path()
            assert str(path) == str(Path("/home/user/.cursor/mcp.json"))

    def test_get_cursor_config_path_windows(self) -> None:
        """Test getting config path on Windows."""
        with (
            patch("pathlib.Path.home", return_value=Path("/home/user")),
            patch("pathlib.Path.exists", return_value=False),
            patch.dict(os.environ, {"USERPROFILE": "C:\\Users\\Test"}),
        ):
            path = get_cursor_config_path()
            assert "Test" in str(path)
            assert ".cursor" in str(path)
            assert "mcp.json" in str(path)

    def test_get_cursor_config_path_no_userprofile(self) -> None:
        """Test getting config path when USERPROFILE not set."""
        with (
            patch("pathlib.Path.home", return_value=Path("/home/user")),
            patch("pathlib.Path.exists", return_value=False),
            patch.dict(os.environ, {}, clear=True),
        ):
            path = get_cursor_config_path()
            # Should still return something based on empty USERPROFILE
            assert ".cursor" in str(path)
            assert "mcp.json" in str(path)


if __name__ == "__main__":
    pytest.main([__file__])
