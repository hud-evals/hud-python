"""Tests for CLI dev module."""

from __future__ import annotations

from unittest import mock

from hud.cli.dev import auto_detect_module, should_use_docker_mode


class TestShouldUseDockerMode:
    """Test Docker mode detection."""

    def test_docker_mode_with_dockerfile(self, tmp_path):
        """Test detection when Dockerfile exists."""
        dockerfile = tmp_path / "Dockerfile"
        dockerfile.write_text("FROM python:3.11")

        assert should_use_docker_mode(tmp_path) is True

    def test_no_docker_mode_without_dockerfile(self, tmp_path):
        """Test detection when Dockerfile doesn't exist."""
        assert should_use_docker_mode(tmp_path) is False

    def test_docker_mode_empty_dockerfile(self, tmp_path):
        """Test detection with empty Dockerfile."""
        dockerfile = tmp_path / "Dockerfile"
        dockerfile.write_text("")

        assert should_use_docker_mode(tmp_path) is True


class TestAutoDetectModule:
    """Test MCP module auto-detection."""

    def test_detect_module_from_init_with_mcpserver(self, tmp_path, monkeypatch):
        """Test detection from __init__.py with MCPServer."""
        monkeypatch.chdir(tmp_path)

        init_file = tmp_path / "__init__.py"
        init_file.write_text("""
from hud.server import MCPServer
mcp = MCPServer(name='test')
""")

        module_name, extra_path = auto_detect_module()

        assert module_name == tmp_path.name
        assert extra_path is None

    def test_detect_module_from_init_with_fastmcp(self, tmp_path, monkeypatch):
        """Test detection from __init__.py with FastMCP."""
        monkeypatch.chdir(tmp_path)

        init_file = tmp_path / "__init__.py"
        init_file.write_text("""
from fastmcp import FastMCP
mcp = FastMCP(name='test')
""")

        module_name, extra_path = auto_detect_module()

        assert module_name == tmp_path.name
        assert extra_path is None

    def test_detect_module_from_main_py(self, tmp_path, monkeypatch):
        """Test detection from main.py with MCPServer."""
        monkeypatch.chdir(tmp_path)

        # Need both __init__.py and main.py
        init_file = tmp_path / "__init__.py"
        init_file.write_text("")

        main_file = tmp_path / "main.py"
        main_file.write_text("""
from hud.server import MCPServer
mcp = MCPServer(name='test')
""")

        module_name, extra_path = auto_detect_module()

        assert module_name == f"{tmp_path.name}.main"
        assert extra_path == tmp_path.parent

    def test_no_detection_without_mcp(self, tmp_path, monkeypatch):
        """Test no detection when mcp not defined."""
        monkeypatch.chdir(tmp_path)

        init_file = tmp_path / "__init__.py"
        init_file.write_text("# Just a comment")

        module_name, extra_path = auto_detect_module()

        assert module_name is None
        assert extra_path is None

    def test_no_detection_empty_dir(self, tmp_path, monkeypatch):
        """Test no detection in empty directory."""
        monkeypatch.chdir(tmp_path)

        module_name, extra_path = auto_detect_module()

        assert module_name is None
        assert extra_path is None


class TestShowDevServerInfo:
    """Test dev server info display."""

    @mock.patch("hud.cli.dev.hud_console")
    def test_show_dev_server_info_http(self, mock_console):
        """Test showing server info for HTTP transport."""
        from hud.cli.dev import show_dev_server_info

        result = show_dev_server_info(
            server_name="test-server",
            port=8000,
            transport="http",
            inspector=False,
            interactive=False,
        )

        # Returns cursor deeplink
        assert result.startswith("cursor://")
        assert "test-server" in result

        # Console should have been called
        assert mock_console.section_title.called
        assert mock_console.info.called

    @mock.patch("hud.cli.dev.hud_console")
    def test_show_dev_server_info_stdio(self, mock_console):
        """Test showing server info for stdio transport."""
        from hud.cli.dev import show_dev_server_info

        result = show_dev_server_info(
            server_name="test-server",
            port=8000,
            transport="stdio",
            inspector=False,
            interactive=False,
        )

        # Returns cursor deeplink
        assert result.startswith("cursor://")

    @mock.patch("hud.cli.dev.hud_console")
    def test_show_dev_server_info_with_telemetry(self, mock_console):
        """Test showing server info with telemetry URLs."""
        from hud.cli.dev import show_dev_server_info

        result = show_dev_server_info(
            server_name="browser-env",
            port=8000,
            transport="http",
            inspector=False,
            interactive=False,
            telemetry={
                "live_url": "https://hud.ai/trace/123",
                "vnc_url": "http://localhost:5900",
            },
        )

        assert result.startswith("cursor://")
