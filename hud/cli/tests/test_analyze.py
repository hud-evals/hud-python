"""Tests for hud.cli.analyze module."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, mock_open, patch

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
        ):
            # Setup mock client
            mock_client = MockClient.return_value
            mock_client.initialize = AsyncMock()
            mock_client.analyze_environment = AsyncMock(return_value=mock_analysis)
            mock_client.close = AsyncMock()

            # Run analysis
            await analyze_environment(
                ["docker", "run", "test"], output_format="json", verbose=False
            )

            # Verify calls
            mock_client.initialize.assert_called_once()
            mock_client.analyze_environment.assert_called_once()
            mock_client.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_analyze_environment_failure(self) -> None:
        """Test environment analysis with initialization failure."""
        with (
            patch("hud.cli.analyze.MCPClient") as MockClient,
            patch("hud.cli.analyze.console"),
        ):
            # Setup mock client to fail
            mock_client = MockClient.return_value
            mock_client.initialize = AsyncMock(side_effect=Exception("Connection failed"))
            mock_client.close = AsyncMock()

            # Run analysis
            await analyze_environment(
                ["docker", "run", "test"], output_format="json", verbose=False
            )

            # Should still close client
            mock_client.close.assert_called_once()

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
                mock_client = MockClient.return_value
                mock_client.initialize = AsyncMock()
                mock_client.analyze_environment = AsyncMock(return_value=mock_analysis)
                mock_client.close = AsyncMock()

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


class TestAnalyzeEnvironmentFromConfig:
    """Test config file based analysis."""

    @pytest.mark.asyncio
    async def test_analyze_from_config_success(self) -> None:
        """Test successful analysis from config file."""
        config_data = {"test": {"command": "python", "args": ["server.py"]}}

        with (
            patch("builtins.open", mock_open(read_data=json.dumps(config_data))),
            patch("hud.cli.analyze._analyze_with_config") as mock_analyze,
        ):
            await analyze_environment_from_config(Path("test.json"), "json", False)
            mock_analyze.assert_called_once_with(config_data, "json", False)

    @pytest.mark.asyncio
    async def test_analyze_from_config_file_error(self) -> None:
        """Test analysis when config file cannot be read."""
        with (
            patch("builtins.open", side_effect=FileNotFoundError()),
            patch("hud.cli.analyze.console") as mock_console,
        ):
            await analyze_environment_from_config(Path("missing.json"), "json", False)
            # Should print error
            assert any(
                "Error loading config" in str(call) for call in mock_console.print.call_args_list
            )


class TestAnalyzeEnvironmentFromMCPConfig:
    """Test MCP config dict based analysis."""

    @pytest.mark.asyncio
    async def test_analyze_from_mcp_config(self) -> None:
        """Test analysis from MCP config dict."""
        mcp_config = {"test": {"command": "python", "args": ["server.py"]}}

        with patch("hud.cli.analyze._analyze_with_config") as mock_analyze:
            await analyze_environment_from_mcp_config(mcp_config, "json", True)
            mock_analyze.assert_called_once_with(mcp_config, "json", True)


class TestDisplayFunctions:
    """Test display formatting functions."""

    def test_display_interactive(self) -> None:
        """Test interactive display format."""
        analysis = {
            "metadata": {"servers": ["test-server"], "initialized": True},
            "tools": [
                {"name": "tool1", "description": "First tool", "input_schema": {"type": "object"}},
                {"name": "tool2", "description": None},
            ],
            "hub_tools": {"hub1": ["func1", "func2"]},
            "resources": [
                {"uri": "res://test", "name": "Test Resource", "mime_type": "text/plain"}
            ],
            "telemetry": {
                "live_url": "http://example.com",
                "status": "running",
                "services": {"service1": "running", "service2": "stopped"},
            },
            "verbose": True,
        }

        with patch("hud.cli.analyze.console") as mock_console:
            display_interactive(analysis)

            # Verify various elements were printed
            print_calls = [str(call) for call in mock_console.print.call_args_list]
            assert any("Environment Overview" in call for call in print_calls)
            assert any("Available Tools" in call for call in print_calls)
            assert any("Available Resources" in call for call in print_calls)
            assert any("Telemetry Data" in call for call in print_calls)

    def test_display_interactive_minimal(self) -> None:
        """Test interactive display with minimal data."""
        analysis = {
            "metadata": {"servers": ["test"], "initialized": False},
            "tools": [],
            "hub_tools": {},
            "resources": [],
            "telemetry": {},
        }

        with patch("hud.cli.analyze.console"):
            # Should not raise any exceptions
            display_interactive(analysis)

    def test_display_markdown(self) -> None:
        """Test markdown display format."""
        analysis = {
            "metadata": {"servers": ["test-server"], "initialized": True},
            "tools": [
                {"name": "tool1", "description": "First tool"},
                {"name": "tool2", "description": None},
            ],
            "hub_tools": {"hub1": ["func1", "func2"]},
            "resources": [
                {"uri": "res://test", "name": "Test Resource", "mime_type": "text/plain"}
            ],
            "telemetry": {
                "live_url": "http://example.com",
                "status": "running",
                "services": {"service1": "running"},
            },
        }

        with patch("hud.cli.analyze.console") as mock_console:
            display_markdown(analysis)

            # Get the markdown output
            output = mock_console.print.call_args[0][0]

            # Verify markdown structure
            assert "# MCP Environment Analysis" in output
            assert "## Environment Overview" in output
            assert "## Available Tools" in output
            assert "### Regular Tools" in output
            assert "### Hub Tools" in output
            assert "## Available Resources" in output
            assert "## Telemetry" in output
            assert "- **tool1**:" in output
            assert "| URI | Name | Type |" in output

    def test_display_markdown_empty_resources(self) -> None:
        """Test markdown display with no resources."""
        analysis = {
            "metadata": {"servers": ["test"], "initialized": True},
            "tools": [],
            "hub_tools": {},
            "resources": [],
            "telemetry": {},
        }

        with patch("hud.cli.analyze.console") as mock_console:
            display_markdown(analysis)
            output = mock_console.print.call_args[0][0]
            # Should not have resources section
            assert "## Available Resources" not in output


class TestAnalyzeWithConfig:
    """Test internal _analyze_with_config function."""

    @pytest.mark.asyncio
    async def test_analyze_with_config_success(self) -> None:
        """Test successful analysis with config."""
        mcp_config = {"test": {"command": "python", "args": ["server.py"]}}
        mock_analysis = {
            "metadata": {"servers": ["test"], "initialized": True},
            "tools": [],
            "hub_tools": {},
            "resources": [],
            "telemetry": {},
        }

        with (
            patch("hud.cli.analyze.MCPClient") as MockClient,
            patch("hud.cli.analyze.console"),
            patch("hud.cli.analyze.display_interactive") as mock_display,
        ):
            # Setup mock client
            mock_client = MockClient.return_value
            mock_client.initialize = AsyncMock()
            mock_client.analyze_environment = AsyncMock(return_value=mock_analysis)
            mock_client.close = AsyncMock()

            # Run analysis
            await _analyze_with_config(mcp_config, "interactive", False)

            # Verify
            MockClient.assert_called_once_with(mcp_config=mcp_config, verbose=False)
            mock_display.assert_called_once_with(mock_analysis)

    @pytest.mark.asyncio
    async def test_analyze_with_config_exception(self) -> None:
        """Test analysis with exception during initialization."""
        mcp_config = {"test": {"command": "python", "args": ["server.py"]}}

        with (
            patch("hud.cli.analyze.MCPClient") as MockClient,
            patch("hud.cli.analyze.console"),
        ):
            # Setup mock client to fail
            mock_client = MockClient.return_value
            mock_client.initialize = AsyncMock(side_effect=RuntimeError("Init failed"))
            mock_client.close = AsyncMock()

            # Run analysis - should handle exception gracefully
            await _analyze_with_config(mcp_config, "json", False)

            # Should still close client
            mock_client.close.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__])
