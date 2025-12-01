"""Tests for CLI flows dev module."""

from __future__ import annotations

import base64
import json
from unittest import mock

import pytest

from hud.cli.flows.dev import generate_cursor_deeplink


class TestGenerateCursorDeeplink:
    """Test Cursor deeplink generation."""

    def test_generate_deeplink_basic(self):
        """Test basic deeplink generation."""
        result = generate_cursor_deeplink("my-server", 8000)

        assert result.startswith("cursor://anysphere.cursor-deeplink/mcp/install?")
        assert "name=my-server" in result
        assert "config=" in result

    def test_generate_deeplink_config_content(self):
        """Test that config contains correct URL."""
        result = generate_cursor_deeplink("test-server", 9999)

        # Extract and decode the config
        config_part = result.split("config=")[1]
        decoded = base64.b64decode(config_part).decode()
        config = json.loads(decoded)

        assert config["url"] == "http://localhost:9999/mcp"

    def test_generate_deeplink_different_ports(self):
        """Test deeplink generation with different ports."""
        result_8000 = generate_cursor_deeplink("server", 8000)
        result_3000 = generate_cursor_deeplink("server", 3000)

        # Decode configs
        config_8000 = json.loads(base64.b64decode(result_8000.split("config=")[1]))
        config_3000 = json.loads(base64.b64decode(result_3000.split("config=")[1]))

        assert "8000" in config_8000["url"]
        assert "3000" in config_3000["url"]

    def test_generate_deeplink_special_characters_in_name(self):
        """Test deeplink with special characters in server name."""
        # Server name with special characters should still work
        result = generate_cursor_deeplink("my-cool_server.v2", 8000)

        assert "name=my-cool_server.v2" in result


class TestCreateDynamicTrace:
    """Test dynamic trace creation."""

    @pytest.mark.asyncio
    @mock.patch("hud.cli.flows.dev.make_request")
    @mock.patch("hud.cli.utils.git.get_git_info")
    @mock.patch("hud.cli.flows.dev.settings")
    async def test_create_dynamic_trace_success(self, mock_settings, mock_git, mock_request):
        """Test successful trace creation."""
        from hud.cli.flows.dev import create_dynamic_trace

        mock_settings.hud_api_url = "https://api.hud.ai"
        mock_settings.api_key = "test-key"
        mock_git.return_value = {"remote_url": "https://github.com/user/repo"}
        mock_request.return_value = {"id": "trace-123"}

        trace_id, url = await create_dynamic_trace(
            mcp_config={"server": {"url": "http://localhost:8000"}},
            build_status=True,
            environment_name="test-env",
        )

        assert trace_id == "trace-123"
        assert url == "https://hud.ai/trace/trace-123"
        mock_request.assert_called_once()

    @pytest.mark.asyncio
    @mock.patch("hud.cli.flows.dev.make_request")
    @mock.patch("hud.cli.utils.git.get_git_info")
    @mock.patch("hud.cli.flows.dev.settings")
    async def test_create_dynamic_trace_no_git(self, mock_settings, mock_git, mock_request):
        """Test trace creation without git info."""
        from hud.cli.flows.dev import create_dynamic_trace

        mock_settings.hud_api_url = "https://api.hud.ai"
        mock_settings.api_key = "test-key"
        mock_git.return_value = {}  # No remote_url
        mock_request.return_value = {"id": "trace-456"}

        trace_id, _ = await create_dynamic_trace(
            mcp_config={"server": {"url": "http://localhost:8000"}},
            build_status=False,
            environment_name="test-env",
        )

        assert trace_id == "trace-456"
        # Verify git_info was not included in payload
        call_args = mock_request.call_args
        assert "git_info" not in call_args.kwargs.get("json", {})

    @pytest.mark.asyncio
    @mock.patch("hud.cli.flows.dev.make_request")
    @mock.patch("hud.cli.utils.git.get_git_info")
    @mock.patch("hud.cli.flows.dev.settings")
    async def test_create_dynamic_trace_api_error(self, mock_settings, mock_git, mock_request):
        """Test trace creation when API fails."""
        from hud.cli.flows.dev import create_dynamic_trace

        mock_settings.hud_api_url = "https://api.hud.ai"
        mock_settings.api_key = "test-key"
        mock_git.return_value = {}
        mock_request.side_effect = Exception("API Error")

        trace_id, url = await create_dynamic_trace(
            mcp_config={"server": {}},
            build_status=True,
            environment_name="test-env",
        )

        assert trace_id is None
        assert url is None
