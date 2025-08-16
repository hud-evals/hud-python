"""Tests for hud.cli.mcp_server module."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from mcp.types import TextContent

from hud.cli.mcp_server import create_mcp_server, run_mcp_server


class TestCreateMCPServer:
    """Test MCP server creation."""

    def test_create_mcp_server(self) -> None:
        """Test that MCP server is created with correct configuration."""
        mcp = create_mcp_server()
        assert mcp._mcp_server.name == "hud-cli"
        # Version is no longer exposed directly

        # Check that tools are registered
        assert hasattr(mcp, "_tool_manager")
        # Can't directly access tools - they're private
        # We'll verify they exist by testing their functionality below


class TestDebugTools:
    """Test debug tool implementations."""

    @pytest.mark.asyncio
    async def test_debug_docker_image(self) -> None:
        """Test debug_docker_image tool."""
        mcp = create_mcp_server()
        tool = mcp._tool_manager._tools["debug_docker_image"]
        tool_func = tool.fn

        with patch("hud.cli.mcp_server.debug_mcp_stdio") as mock_debug:
            mock_debug.return_value = 3  # Completed 3 phases

            result = await tool_func(
                image="test-image:latest", docker_args=["-e", "KEY=value"], max_phase=5
            )

            assert isinstance(result, list)
            assert len(result) == 1
            assert isinstance(result[0], TextContent)
            assert "Completed 3/5 phases successfully" in result[0].text

            # Verify debug was called correctly
            mock_debug.assert_called_once()
            command = mock_debug.call_args[0][0]
            assert command == [
                "docker",
                "run",
                "--rm",
                "-i",
                "-e",
                "KEY=value",
                "test-image:latest",
            ]

    @pytest.mark.asyncio
    async def test_debug_docker_image_no_args(self) -> None:
        """Test debug_docker_image with no docker args."""
        mcp = create_mcp_server()
        tool = mcp._tool_manager._tools["debug_docker_image"]
        tool_func = tool.fn

        with patch("hud.cli.mcp_server.debug_mcp_stdio") as mock_debug:
            mock_debug.return_value = 5

            await tool_func(image="test-image", max_phase=5)

            command = mock_debug.call_args[0][0]
            assert command == ["docker", "run", "--rm", "-i", "test-image"]

    @pytest.mark.asyncio
    async def test_debug_cursor_config_success(self) -> None:
        """Test debug_cursor_config tool success."""
        mcp = create_mcp_server()
        tool = mcp._tool_manager._tools["debug_cursor_config"]
        tool_func = tool.fn

        with patch("hud.cli.mcp_server.parse_cursor_config") as mock_parse:
            mock_parse.return_value = (["python", "server.py"], None)

            with patch("hud.cli.mcp_server.debug_mcp_stdio") as mock_debug:
                mock_debug.return_value = 4

                result = await tool_func(server_name="test-server", max_phase=5)

                assert isinstance(result, list)
                assert "Completed 4/5 phases successfully" in result[0].text

    @pytest.mark.asyncio
    async def test_debug_cursor_config_error(self) -> None:
        """Test debug_cursor_config with parse error."""
        mcp = create_mcp_server()
        tool = mcp._tool_manager._tools["debug_cursor_config"]
        tool_func = tool.fn

        with patch("hud.cli.mcp_server.parse_cursor_config") as mock_parse:
            mock_parse.return_value = (None, "Server not found")

            result = await tool_func(server_name="missing-server")

            assert isinstance(result, list)
            assert result[0].text == "❌ Server not found"

    @pytest.mark.asyncio
    async def test_debug_config(self) -> None:
        """Test debug_config tool."""
        mcp = create_mcp_server()
        tool = mcp._tool_manager._tools["debug_config"]
        tool_func = tool.fn

        config = {"test-server": {"command": "node", "args": ["app.js", "--port", "3000"]}}

        with patch("hud.cli.mcp_server.debug_mcp_stdio") as mock_debug:
            mock_debug.return_value = 5

            result = await tool_func(config=config, max_phase=5)

            assert "Completed 5/5 phases successfully" in result[0].text

            # Verify command extraction
            command = mock_debug.call_args[0][0]
            assert command == ["node", "app.js", "--port", "3000"]


class TestAnalyzeTools:
    """Test analyze tool implementations."""

    @pytest.mark.asyncio
    async def test_analyze_docker_image_success(self) -> None:
        """Test analyze_docker_image tool success."""
        mcp = create_mcp_server()
        tool = mcp._tool_manager._tools["analyze_docker_image"]
        tool_func = tool.fn

        mock_analysis = {
            "metadata": {"servers": ["test"], "initialized": True},
            "tools": [{"name": "tool1", "description": "Test tool"}],
            "resources": [],
        }

        with patch("hud.clients.MCPClient") as MockClient:
            mock_client = MockClient.return_value
            mock_client.initialize = AsyncMock()
            mock_client.analyze_environment = AsyncMock(return_value=mock_analysis)
            mock_client.close = AsyncMock()

            result = await tool_func(
                image="test-image:latest", docker_args=["-p", "8080:8080"], verbose=True
            )

            assert isinstance(result, list)
            assert isinstance(result[0], TextContent)
            # Should return JSON
            parsed = json.loads(result[0].text)
            assert parsed["metadata"]["servers"] == ["test"]

    @pytest.mark.asyncio
    async def test_analyze_docker_image_failure(self) -> None:
        """Test analyze_docker_image tool failure."""
        mcp = create_mcp_server()
        tool = mcp._tool_manager._tools["analyze_docker_image"]
        tool_func = tool.fn

        with patch("hud.clients.MCPClient") as MockClient:
            mock_client = MockClient.return_value
            mock_client.initialize = AsyncMock(side_effect=Exception("Connection failed"))

            result = await tool_func(image="test-image")

            assert "❌ Analysis failed: Connection failed" in result[0].text
            assert "Make sure the environment passes debug phase 3" in result[0].text

    @pytest.mark.asyncio
    async def test_analyze_cursor_config_success(self) -> None:
        """Test analyze_cursor_config tool success."""
        mcp = create_mcp_server()
        tool = mcp._tool_manager._tools["analyze_cursor_config"]
        tool_func = tool.fn

        with patch("hud.cli.mcp_server.parse_cursor_config") as mock_parse:
            mock_parse.return_value = (["python", "server.py"], None)

            mock_analysis = {"tools": [], "resources": []}

            with patch("hud.clients.MCPClient") as MockClient:
                mock_client = MockClient.return_value
                mock_client.initialize = AsyncMock()
                mock_client.analyze_environment = AsyncMock(return_value=mock_analysis)
                mock_client.close = AsyncMock()

                result = await tool_func(server_name="test-server", verbose=False)

                assert isinstance(result[0].text, str)
                parsed = json.loads(result[0].text)
                assert "tools" in parsed

    @pytest.mark.asyncio
    async def test_analyze_cursor_config_parse_error(self) -> None:
        """Test analyze_cursor_config with parse error."""
        mcp = create_mcp_server()
        tool = mcp._tool_manager._tools["analyze_cursor_config"]
        tool_func = tool.fn

        with patch("hud.cli.mcp_server.parse_cursor_config") as mock_parse:
            mock_parse.return_value = (None, "Config not found")

            result = await tool_func(server_name="missing")

            assert result[0].text == "❌ Config not found"

    @pytest.mark.asyncio
    async def test_analyze_config(self) -> None:
        """Test analyze_config tool."""
        mcp = create_mcp_server()
        tool = mcp._tool_manager._tools["analyze_config"]
        tool_func = tool.fn

        config = {"server": {"command": "python", "args": ["app.py"]}}
        mock_analysis = {"metadata": {"initialized": True}}

        with patch("hud.clients.MCPClient") as MockClient:
            mock_client = MockClient.return_value
            mock_client.initialize = AsyncMock()
            mock_client.analyze_environment = AsyncMock(return_value=mock_analysis)
            mock_client.close = AsyncMock()

            result = await tool_func(config=config, verbose=True)

            parsed = json.loads(result[0].text)
            assert parsed["metadata"]["initialized"] is True


class TestListCursorServers:
    """Test list_cursor_servers tool."""

    @pytest.mark.asyncio
    async def test_list_cursor_servers_success(self) -> None:
        """Test listing Cursor servers successfully."""
        mcp = create_mcp_server()
        tool = mcp._tool_manager._tools["list_cursor_servers"]
        tool_func = tool.fn

        with patch("hud.cli.cursor.list_cursor_servers") as mock_list:
            mock_list.return_value = (["server1", "server2", "server3"], None)

            result = await tool_func()

            assert "📋 Available Cursor MCP Servers:" in result[0].text
            assert "• server1" in result[0].text
            assert "• server2" in result[0].text
            assert "• server3" in result[0].text

    @pytest.mark.asyncio
    async def test_list_cursor_servers_empty(self) -> None:
        """Test listing when no servers found."""
        mcp = create_mcp_server()
        tool = mcp._tool_manager._tools["list_cursor_servers"]
        tool_func = tool.fn

        with patch("hud.cli.cursor.list_cursor_servers") as mock_list:
            mock_list.return_value = ([], None)

            result = await tool_func()

            assert result[0].text == "No servers found in Cursor config"

    @pytest.mark.asyncio
    async def test_list_cursor_servers_error(self) -> None:
        """Test listing with error."""
        mcp = create_mcp_server()
        tool = mcp._tool_manager._tools["list_cursor_servers"]
        tool_func = tool.fn

        with patch("hud.cli.cursor.list_cursor_servers") as mock_list:
            mock_list.return_value = (None, "Permission denied")

            result = await tool_func()

            assert result[0].text == "❌ Permission denied"


class TestRunMCPServer:
    """Test run_mcp_server function."""

    def test_run_mcp_server(self) -> None:
        """Test that run_mcp_server creates and runs server."""
        with patch("hud.cli.mcp_server.create_mcp_server") as mock_create:
            mock_mcp = MagicMock()
            mock_create.return_value = mock_mcp

            run_mcp_server()

            mock_create.assert_called_once()
            mock_mcp.run.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__])
