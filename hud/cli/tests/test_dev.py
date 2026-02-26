"""Tests for CLI dev module."""

from __future__ import annotations

import asyncio
import socket
from contextlib import suppress
from unittest import mock

import pytest

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

    def test_detect_module_from_init_with_environment(self, tmp_path, monkeypatch):
        """Test detection from __init__.py with Environment."""
        monkeypatch.chdir(tmp_path)

        init_file = tmp_path / "__init__.py"
        init_file.write_text("""
from hud import Environment
env = Environment(name='test')
""")

        module_name, extra_path = auto_detect_module()

        assert module_name == tmp_path.name
        assert extra_path is None

    def test_detect_module_from_main_py_with_environment(self, tmp_path, monkeypatch):
        """Test detection from main.py with Environment."""
        monkeypatch.chdir(tmp_path)

        # Need both __init__.py and main.py
        init_file = tmp_path / "__init__.py"
        init_file.write_text("")

        main_file = tmp_path / "main.py"
        main_file.write_text("""
from hud import Environment
env = Environment(name='test')
""")

        module_name, extra_path = auto_detect_module()

        assert module_name == f"{tmp_path.name}.main"
        assert extra_path == tmp_path.parent

    def test_no_detection_without_mcp_or_env(self, tmp_path, monkeypatch):
        """Test no detection when neither mcp nor env is defined."""
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

    @mock.patch("hud.cli.dev.hud_console")
    def test_show_dev_server_info_without_hot_reload(self, mock_console):
        """Test that no-watch mode does not claim hot-reload is enabled."""
        from hud.cli.dev import show_dev_server_info

        result = show_dev_server_info(
            server_name="test-server",
            port=8000,
            transport="stdio",
            inspector=False,
            interactive=False,
            hot_reload_enabled=False,
        )

        assert result.startswith("cursor://")
        info_messages = [
            str(call.args[0]) for call in mock_console.info.call_args_list if call.args
        ]
        assert any("Hot-reload disabled" in msg for msg in info_messages)
        assert not any("Hot-reload enabled" in msg for msg in info_messages)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class TestDockerProxyPassthrough:
    """Integration test: the Docker dev proxy forwards unlisted tools.

    Mirrors the real ``hud dev --docker`` flow end-to-end over HTTP:

    1. An Environment with a scenario runs on an HTTP port (simulates Docker).
    2. ``build_proxy`` — the same function ``run_docker_dev_server`` calls —
       constructs the proxy with its _call_tool passthrough.
    3. A client connects to the proxy and calls ``_hud_submit``.

    If ``build_proxy`` is changed or its passthrough breaks, this test fails.
    """

    @pytest.mark.asyncio
    async def test_hud_submit_forwarded_over_http(self) -> None:
        from fastmcp import Client
        from fastmcp.server.proxy import ProxyClient

        from hud.cli.dev import build_proxy
        from hud.environment import Environment

        backend = Environment("test-env")

        @backend.tool()
        def public_tool() -> str:
            return "public"

        @backend.scenario("greet")
        async def greet(name: str = "world"):
            yield f"Hello, {name}!"
            yield 1.0

        backend_port = _free_port()
        backend_task = asyncio.create_task(
            backend.run_async(
                transport="http",
                host="127.0.0.1",
                port=backend_port,
                path="/mcp",
                log_level="ERROR",
                show_banner=False,
            )
        )
        await asyncio.sleep(0.1)

        try:
            backend_url = f"http://127.0.0.1:{backend_port}/mcp"
            proxy_client = ProxyClient(backend_url, name="test-proxy-client")
            proxy = await build_proxy(proxy_client, name="test-proxy")

            # _hud_submit should be hidden from listings but still callable
            proxy_tools = await proxy.get_tools()
            assert "_hud_submit" not in proxy_tools
            assert "public_tool" in proxy_tools

            proxy_port = _free_port()
            proxy_task = asyncio.create_task(
                proxy.run_async(
                    transport="http",
                    host="127.0.0.1",
                    port=proxy_port,
                    path="/mcp",
                    log_level="ERROR",
                    show_banner=False,
                )
            )
            await asyncio.sleep(0.1)

            try:
                proxy_url = f"http://127.0.0.1:{proxy_port}/mcp"

                async with Client(proxy_url) as client:
                    await client.get_prompt("test-env:greet", {"name": "world"})

                    result = await client.call_tool(
                        "_hud_submit", {"scenario": "greet", "answer": "42"}
                    )
                    text = str(result)
                    assert "submitted" in text.lower() or "answer" in text.lower()

                    result = await client.call_tool("public_tool", {})
                    assert "public" in str(result).lower()
            finally:
                proxy_task.cancel()
                with suppress(asyncio.CancelledError):
                    await proxy_task
        finally:
            backend_task.cancel()
            with suppress(asyncio.CancelledError):
                await backend_task
