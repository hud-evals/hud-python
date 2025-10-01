"""Tests for analyze_metadata.py - Fast metadata analysis functions."""

from __future__ import annotations

import json
from unittest import mock

import pytest
import yaml

from hud.cli.utils.metadata import (
    analyze_from_metadata,
    check_local_cache,
    fetch_lock_from_registry,
)
from hud.cli.utils.registry import save_to_registry


@pytest.fixture
def mock_registry_dir(tmp_path):
    """Create a mock registry directory."""
    registry_dir = tmp_path / ".hud" / "envs"
    registry_dir.mkdir(parents=True)
    return registry_dir


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
    """Test fetching lock data from registry."""

    @mock.patch("requests.get")
    def test_fetch_lock_success(self, mock_get):
        """Test successful fetch from registry."""
        mock_response = mock.Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"lock": yaml.dump({"test": "data"})}
        mock_get.return_value = mock_response

        result = fetch_lock_from_registry("test/env:latest")
        assert result == {"test": "data"}
        mock_get.assert_called_once()

    @mock.patch("requests.get")
    def test_fetch_lock_with_lock_data(self, mock_get):
        """Test fetch when response has lock_data key."""
        mock_response = mock.Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"lock_data": {"test": "data"}}
        mock_get.return_value = mock_response

        result = fetch_lock_from_registry("test/env:latest")
        assert result == {"test": "data"}

    @mock.patch("requests.get")
    def test_fetch_lock_direct_data(self, mock_get):
        """Test fetch when response is direct lock data."""
        mock_response = mock.Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"test": "data"}
        mock_get.return_value = mock_response

        result = fetch_lock_from_registry("test/env:latest")
        assert result == {"test": "data"}

    @mock.patch("requests.get")
    def test_fetch_lock_adds_latest_tag(self, mock_get):
        """Test that :latest tag is added if missing."""
        mock_response = mock.Mock()
        mock_response.status_code = 404
        mock_get.return_value = mock_response

        fetch_lock_from_registry("test/env")

        # Check that the URL includes :latest (URL-encoded)
        call_args = mock_get.call_args
        assert "test/env%3Alatest" in call_args[0][0]

    @mock.patch("requests.get")
    def test_fetch_lock_failure(self, mock_get):
        """Test fetch failure returns None."""
        mock_response = mock.Mock()
        mock_response.status_code = 404
        mock_get.return_value = mock_response

        result = fetch_lock_from_registry("test/env:latest")
        assert result is None

    @mock.patch("requests.get")
    def test_fetch_lock_exception(self, mock_get):
        """Test fetch exception returns None."""
        mock_get.side_effect = Exception("Network error")

        result = fetch_lock_from_registry("test/env:latest")
        assert result is None


class TestCheckLocalCache:
    """Test checking local cache for lock data."""

    def test_check_local_cache_found(self, mock_registry_dir, sample_lock_data, monkeypatch):
        """Test finding lock data in local cache."""
        # Mock registry directory
        monkeypatch.setattr("hud.cli.utils.registry.get_registry_dir", lambda: mock_registry_dir)

        # Save sample data to registry
        save_to_registry(sample_lock_data, "test/environment:latest", verbose=False)

        # Check cache
        result = check_local_cache("test/environment:latest")
        assert result is not None
        assert result["image"] == "test/environment:latest"

    def test_check_local_cache_not_found(self, mock_registry_dir, monkeypatch):
        """Test when lock data not in local cache."""
        monkeypatch.setattr("hud.cli.utils.registry.get_registry_dir", lambda: mock_registry_dir)

        result = check_local_cache("nonexistent/env:latest")
        assert result is None

    def test_check_local_cache_invalid_yaml(self, mock_registry_dir, monkeypatch):
        """Test when lock file has invalid YAML."""
        monkeypatch.setattr("hud.cli.utils.registry.get_registry_dir", lambda: mock_registry_dir)

        # Create invalid lock file
        digest = "invalid"
        lock_file = mock_registry_dir / digest / "hud.lock.yaml"
        lock_file.parent.mkdir(parents=True)
        lock_file.write_text("invalid: yaml: content:")

        result = check_local_cache("test/invalid:latest")
        assert result is None


# Note: TestFormatAnalysisOutput class removed since format_analysis_output function doesn't exist
# The formatting is done inline within analyze_from_metadata


@pytest.mark.asyncio
class TestAnalyzeFromMetadata:
    """Test the main analyze_from_metadata function."""

    @mock.patch("hud.cli.utils.metadata.check_local_cache")
    @mock.patch("hud.cli.utils.metadata.console")
    async def test_analyze_from_local_cache(self, mock_console, mock_check, sample_lock_data):
        """Test analyzing from local cache."""
        mock_check.return_value = sample_lock_data

        await analyze_from_metadata("test/env:latest", "json", verbose=False)

        mock_check.assert_called_once_with("test/env:latest")
        # Should output JSON
        mock_console.print_json.assert_called_once()

    @mock.patch("hud.cli.utils.metadata.check_local_cache")
    @mock.patch("hud.cli.utils.metadata.fetch_lock_from_registry")
    @mock.patch("hud.cli.utils.registry.save_to_registry")
    @mock.patch("hud.cli.utils.metadata.console")
    async def test_analyze_from_registry(
        self, mock_console, mock_save, mock_fetch, mock_check, sample_lock_data
    ):
        """Test analyzing from registry when not in cache."""
        mock_check.return_value = None
        mock_fetch.return_value = sample_lock_data

        await analyze_from_metadata("test/env:latest", "json", verbose=False)

        mock_check.assert_called_once()
        mock_fetch.assert_called_once()
        mock_save.assert_called_once()  # Should save to cache
        mock_console.print_json.assert_called_once()

    @mock.patch("hud.cli.utils.metadata.check_local_cache")
    @mock.patch("hud.cli.utils.metadata.fetch_lock_from_registry")
    @mock.patch("hud.cli.utils.metadata.hud_console")
    @mock.patch("hud.cli.utils.metadata.console")
    async def test_analyze_not_found(self, mock_console, mock_hud_console, mock_fetch, mock_check):
        """Test when environment not found anywhere."""
        mock_check.return_value = None
        mock_fetch.return_value = None

        await analyze_from_metadata("test/notfound:latest", "json", verbose=False)

        # Should show error via hud_console
        mock_hud_console.error.assert_called_with("Environment metadata not found")
        # Should print suggestions via console
        mock_console.print.assert_called()

    @mock.patch("hud.cli.utils.metadata.check_local_cache")
    @mock.patch("hud.cli.utils.metadata.console")
    async def test_analyze_verbose_mode(self, mock_console, mock_check, sample_lock_data):
        """Test verbose mode includes input schemas."""
        mock_check.return_value = sample_lock_data

        await analyze_from_metadata("test/env:latest", "json", verbose=True)

        # In verbose mode, the JSON output should include input schemas
        mock_console.print_json.assert_called_once()
        # Get the JSON string that was printed
        call_args = mock_console.print_json.call_args[0][0]
        output_data = json.loads(call_args)
        assert "inputSchema" in output_data["tools"][0]

    @mock.patch("hud.cli.utils.metadata.check_local_cache")
    @mock.patch("hud.cli.utils.metadata.fetch_lock_from_registry")
    async def test_analyze_registry_reference_parsing(self, mock_fetch, mock_check):
        """Test parsing of different registry reference formats."""
        mock_check.return_value = None
        mock_fetch.return_value = {"test": "data"}

        # Test different reference formats
        test_cases = [
            ("docker.io/org/name:tag", "org/name:tag"),
            ("registry-1.docker.io/org/name", "org/name"),
            ("org/name@sha256:abc", "org/name"),
            ("org/name", "org/name"),
            ("name:tag", "name:tag"),
        ]

        for input_ref, expected_call in test_cases:
            await analyze_from_metadata(input_ref, "json", verbose=False)

            # Check what was passed to fetch_lock_from_registry
            calls = mock_fetch.call_args_list
            last_call = calls[-1][0][0]

            # The function might add :latest, so check base name
            assert expected_call.split(":")[0] in last_call
