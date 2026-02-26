"""Tests for metadata.py - Fast metadata analysis functions."""

from __future__ import annotations

import json
from unittest import mock

import pytest

from hud.cli.utils.metadata import (
    analyze_from_metadata,
    fetch_lock_from_registry,
)


@pytest.fixture
def sample_lock_data():
    """Sample lock data for testing."""
    return {
        "image": "test/environment:latest",
        "digest": "sha256:abc123",
        "build": {
            "timestamp": 1234567890,
            "version": "1.0.0",
            "hud_version": "0.1.0",
        },
        "environment": {
            "initializeMs": 1500,
            "toolCount": 5,
            "variables": {"API_KEY": "required"},
        },
        "tools": [
            {
                "name": "test_tool",
                "description": "A test tool",
                "inputSchema": {
                    "type": "object",
                    "properties": {"message": {"type": "string"}},
                },
            }
        ],
        "resources": [
            {
                "uri": "test://resource",
                "name": "Test Resource",
                "description": "A test resource",
                "mimeType": "text/plain",
            }
        ],
        "prompts": [
            {
                "name": "test_prompt",
                "description": "A test prompt",
                "arguments": [{"name": "arg1", "description": "First argument"}],
            }
        ],
    }


class TestFetchLockFromRegistry:
    """Test fetching lock data from HUD registry."""

    @mock.patch("requests.get")
    def test_fetch_lock_success(self, mock_get):
        import yaml

        mock_response = mock.Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"lock": yaml.dump({"test": "data"})}
        mock_get.return_value = mock_response

        result = fetch_lock_from_registry("test/env:latest")
        assert result == {"test": "data"}
        mock_get.assert_called_once()

    @mock.patch("requests.get")
    def test_fetch_lock_with_lock_data(self, mock_get):
        mock_response = mock.Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"lock_data": {"test": "data"}}
        mock_get.return_value = mock_response

        result = fetch_lock_from_registry("test/env:latest")
        assert result == {"test": "data"}

    @mock.patch("requests.get")
    def test_fetch_lock_direct_data(self, mock_get):
        mock_response = mock.Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"test": "data"}
        mock_get.return_value = mock_response

        result = fetch_lock_from_registry("test/env:latest")
        assert result == {"test": "data"}

    @mock.patch("requests.get")
    def test_fetch_lock_adds_latest_tag(self, mock_get):
        mock_response = mock.Mock()
        mock_response.status_code = 404
        mock_get.return_value = mock_response

        fetch_lock_from_registry("test/env")

        call_args = mock_get.call_args
        assert "test/env%3Alatest" in call_args[0][0]

    @mock.patch("requests.get")
    def test_fetch_lock_failure(self, mock_get):
        mock_response = mock.Mock()
        mock_response.status_code = 404
        mock_get.return_value = mock_response

        result = fetch_lock_from_registry("test/env:latest")
        assert result is None

    @mock.patch("requests.get")
    def test_fetch_lock_exception(self, mock_get):
        mock_get.side_effect = Exception("Network error")

        result = fetch_lock_from_registry("test/env:latest")
        assert result is None


@pytest.mark.asyncio
class TestAnalyzeFromMetadata:
    """Test the main analyze_from_metadata function."""

    @mock.patch("hud.cli.utils.metadata.fetch_lock_from_registry")
    @mock.patch("hud.cli.utils.metadata.console")
    async def test_analyze_from_registry(self, mock_console, mock_fetch, sample_lock_data):
        mock_fetch.return_value = sample_lock_data

        await analyze_from_metadata("test/env:latest", "json", verbose=False)

        mock_fetch.assert_called_once()
        mock_console.print_json.assert_called_once()

    @mock.patch("hud.cli.utils.metadata.fetch_lock_from_registry")
    @mock.patch("hud.cli.utils.metadata.hud_console")
    @mock.patch("hud.cli.utils.metadata.console")
    async def test_analyze_not_found(self, mock_console, mock_hud_console, mock_fetch):
        mock_fetch.return_value = None

        await analyze_from_metadata("test/notfound:latest", "json", verbose=False)

        mock_hud_console.error.assert_called_with("Environment metadata not found")
        mock_console.print.assert_called()

    @mock.patch("hud.cli.utils.metadata.fetch_lock_from_registry")
    @mock.patch("hud.cli.utils.metadata.console")
    async def test_analyze_verbose_mode(self, mock_console, mock_fetch, sample_lock_data):
        mock_fetch.return_value = sample_lock_data

        await analyze_from_metadata("test/env:latest", "json", verbose=True)

        mock_console.print_json.assert_called_once()
        call_args = mock_console.print_json.call_args[0][0]
        output_data = json.loads(call_args)
        assert "inputSchema" in output_data["tools"][0]

    @mock.patch("hud.cli.utils.metadata.fetch_lock_from_registry")
    async def test_analyze_registry_reference_parsing(self, mock_fetch):
        mock_fetch.return_value = {"test": "data"}

        test_cases = [
            ("docker.io/org/name:tag", "org/name:tag"),
            ("registry-1.docker.io/org/name", "org/name"),
            ("org/name@sha256:abc", "org/name"),
            ("org/name", "org/name"),
            ("name:tag", "name:tag"),
        ]

        for input_ref, expected_call in test_cases:
            await analyze_from_metadata(input_ref, "json", verbose=False)

            calls = mock_fetch.call_args_list
            last_call = calls[-1][0][0]
            assert expected_call.split(":")[0] in last_call
