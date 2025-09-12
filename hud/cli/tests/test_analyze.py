"""Tests for hud.cli.analyze module."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, mock_open, patch

import pytest

from hud.cli.analyze import (
    _analyze_with_config,
    analyze_environment,
    analyze_environment_from_config,
    analyze_environment_from_mcp_config,
    display_interactive,
    display_markdown,
    parse_docker_command,
)


class TestParseDockerCommand:
    """Test Docker command parsing."""

    def test_parse_simple_docker_command(self) -> None:
        """Test parsing simple Docker command."""
        docker_cmd = ["docker", "run", "image:latest"]
        result = parse_docker_command(docker_cmd)
        assert result == {"local": {"command": "docker", "args": ["run", "image:latest"]}}

    def test_parse_docker_command_no_args(self) -> None:
        """Test parsing Docker command with no arguments."""
        docker_cmd = ["docker"]
        result = parse_docker_command(docker_cmd)
        assert result == {"local": {"command": "docker", "args": []}}


class TestAnalyzeEnvironment:
    """Test main analyze_environment function."""

    @pytest.mark.asyncio
    async def test_analyze_environment_success(self) -> None:
        """Test successful environment analysis."""
        mock_analysis = {
            "metadata": {"servers": ["test"], "initialized": True},
            "tools": [{"name": "tool1", "description": "Test tool"}],
            "hub_tools": {},
            "resources": [],
            "telemetry": {},
        }

        with (
            patch("hud.cli.analyze.MCPClient") as MockClient,
            patch("hud.cli.analyze.console"),
            patch("hud.cli.analyze.display_interactive") as mock_interactive,
        ):
            # Setup mock client - return an instance with async methods
            mock_client = MagicMock()
            mock_client.initialize = AsyncMock()
            mock_client.analyze_environment = AsyncMock(return_value=mock_analysis)
            mock_client.shutdown = AsyncMock()
            MockClient.return_value = mock_client

            await analyze_environment(
                ["docker", "run", "test"],
                output_format="interactive",
                verbose=False,
            )

            # Check client was used correctly
            MockClient.assert_called_once()
            mock_client.initialize.assert_called_once()
            mock_client.analyze_environment.assert_called_once()
            mock_client.shutdown.assert_called_once()

            # Check interactive display was called
            mock_interactive.assert_called_once_with(mock_analysis)

    @pytest.mark.asyncio
    async def test_analyze_environment_failure(self) -> None:
        """Test handling analysis failure."""
        with (
            patch("hud.cli.analyze.MCPClient") as MockClient,
            patch("hud.cli.analyze.console") as mock_console,
            patch("platform.system", return_value="Windows"),
        ):
            # Setup mock client that will raise exception during initialization
            mock_client = MagicMock()
            mock_client.initialize = AsyncMock(side_effect=RuntimeError("Connection failed"))
            mock_client.shutdown = AsyncMock()
            MockClient.return_value = mock_client

            # Test should not raise exception
            await analyze_environment(
                ["docker", "run", "test"],
                output_format="json",
                verbose=False,
            )

            # Check error was handled
            mock_client.initialize.assert_called_once()
            mock_client.shutdown.assert_called_once()

            # Check console printed Windows-specific error hints
            calls = mock_console.print.call_args_list
            assert any("Docker logs may not show on Windows" in str(call) for call in calls)

    @pytest.mark.asyncio
    async def test_analyze_environment_formats(self) -> None:
        """Test different output formats."""
        mock_analysis = {
            "metadata": {"servers": ["test"], "initialized": True},
            "tools": [],
            "hub_tools": {},
            "resources": [],
            "telemetry": {},
            "verbose": False,
        }

        for output_format in ["json", "markdown", "interactive"]:
            with (
                patch("hud.cli.analyze.MCPClient") as MockClient,
                patch("hud.cli.analyze.console") as mock_console,
                patch("hud.cli.analyze.display_interactive") as mock_interactive,
                patch("hud.cli.analyze.display_markdown") as mock_markdown,
            ):
                # Setup mock client
                mock_client = MagicMock()
                mock_client.initialize = AsyncMock()
                mock_client.analyze_environment = AsyncMock(return_value=mock_analysis)
                mock_client.shutdown = AsyncMock()
                MockClient.return_value = mock_client

                # Run analysis
                await analyze_environment(
                    ["docker", "run", "test"],
                    output_format=output_format,
                    verbose=False,
                )

                # Check correct display function was called
                if output_format == "json":
                    mock_console.print_json.assert_called()
                elif output_format == "markdown":
                    mock_markdown.assert_called_once_with(mock_analysis)
                else:  # interactive
                    mock_interactive.assert_called_once_with(mock_analysis)


class TestAnalyzeWithConfig:
    """Test config-based analysis functions."""

    @pytest.mark.asyncio
    async def test_analyze_with_config_success(self) -> None:
        """Test successful config-based analysis."""
        mock_config = {"server": {"command": "test", "args": ["--arg"]}}
        mock_analysis = {
            "metadata": {"servers": ["server"], "initialized": True},
            "tools": [],
            "hub_tools": {},
            "resources": [],
            "telemetry": {},
        }

        with (
            patch("hud.cli.analyze.MCPClient") as MockClient,
            patch("hud.cli.analyze.console"),
            patch("hud.cli.analyze.display_interactive") as mock_interactive,
        ):
            # Setup mock client
            mock_client = MagicMock()
            mock_client.initialize = AsyncMock()
            mock_client.analyze_environment = AsyncMock(return_value=mock_analysis)
            mock_client.shutdown = AsyncMock()
            MockClient.return_value = mock_client

            await _analyze_with_config(
                mock_config,
                output_format="interactive",
                verbose=False,
            )

            # Check client was created with correct config
            MockClient.assert_called_once_with(mcp_config=mock_config, verbose=False)
            mock_interactive.assert_called_once_with(mock_analysis)

    @pytest.mark.asyncio
    async def test_analyze_with_config_exception(self) -> None:
        """Test config analysis handles exceptions gracefully."""
        mock_config = {"server": {"command": "test"}}

        with (
            patch("hud.cli.analyze.MCPClient") as MockClient,
            patch("hud.cli.analyze.console"),
        ):
            # Setup mock client that fails
            mock_client = MagicMock()
            mock_client.initialize = AsyncMock(side_effect=Exception("Test error"))
            mock_client.shutdown = AsyncMock()
            MockClient.return_value = mock_client

            # Should not raise
            await _analyze_with_config(
                mock_config,
                output_format="json",
                verbose=False,
            )

            mock_client.shutdown.assert_called_once()

    @pytest.mark.asyncio
    async def test_analyze_environment_from_config(self) -> None:
        """Test analyze_environment_from_config."""
        config_data = {"server": {"command": "test"}}
        mock_path = Path("test.json")

        with (
            patch("builtins.open", mock_open(read_data=json.dumps(config_data))),
            patch("hud.cli.analyze._analyze_with_config") as mock_analyze,
        ):
            await analyze_environment_from_config(mock_path, "json", False)

            mock_analyze.assert_called_once_with(config_data, "json", False)

    @pytest.mark.asyncio
    async def test_analyze_environment_from_mcp_config(self) -> None:
        """Test analyze_environment_from_mcp_config."""
        config_data = {"server": {"command": "test"}}

        with patch("hud.cli.analyze._analyze_with_config") as mock_analyze:
            await analyze_environment_from_mcp_config(config_data, "markdown", True)

            mock_analyze.assert_called_once_with(config_data, "markdown", True)


class TestDisplayFunctions:
    """Test display formatting functions."""

    def test_display_interactive_basic(self) -> None:
        """Test basic interactive display."""
        analysis = {
            "metadata": {"servers": ["test"], "initialized": True},
            "tools": [{"name": "tool1", "description": "Test tool"}],
            "hub_tools": {"hub1": ["func1", "func2"]},
            "resources": [{"uri": "file:///test", "name": "Test", "description": "Resource"}],
            "telemetry": {"status": "running", "live_url": "http://test"},
        }

        with patch("hud.cli.analyze.console") as mock_console:
            display_interactive(analysis)

            # Check console was called multiple times
            assert mock_console.print.call_count > 0
            # The hud_console.section_title uses its own console, not the patched one
            # Just verify the function ran without errors

    def test_display_markdown_basic(self) -> None:
        """Test basic markdown display."""
        analysis = {
            "metadata": {"servers": ["test1", "test2"], "initialized": True},
            "tools": [
                {"name": "tool1", "description": "Tool 1"},
                {"name": "setup", "description": "Hub tool"},
            ],
            "hub_tools": {"setup": ["init", "config"]},
            "resources": [{"uri": "telemetry://live", "name": "Telemetry"}],
            "telemetry": {"status": "active"},
        }

        with patch("hud.cli.analyze.console") as mock_console:
            display_markdown(analysis)

            # Get the markdown output
            mock_console.print.assert_called_once()
            markdown = mock_console.print.call_args[0][0]

            # Check markdown structure
            assert "# MCP Environment Analysis" in markdown
            assert "## Environment Overview" in markdown
            assert "## Available Tools" in markdown
            assert "### Regular Tools" in markdown
            assert "### Hub Tools" in markdown
            assert "- **tool1**: Tool 1" in markdown
            assert "- **setup**" in markdown
            assert "  - init" in markdown
