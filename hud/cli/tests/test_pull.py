"""Tests for pull.py - Pull HUD environments from registry."""

from __future__ import annotations

import json
from unittest import mock

import pytest
import typer
import yaml

from hud.cli.pull import (
    fetch_lock_from_registry,
    format_size,
    get_docker_manifest,
    get_image_size_from_manifest,
    pull_command,
    pull_environment,
)


class TestGetDockerManifest:
    """Test getting Docker manifest."""

    @mock.patch("subprocess.run")
    def test_get_docker_manifest_success(self, mock_run):
        """Test successfully getting Docker manifest."""
        manifest_data = {
            "schemaVersion": 2,
            "mediaType": "application/vnd.docker.distribution.manifest.v2+json",
            "layers": [{"size": 1024}, {"size": 2048}],
        }
        mock_run.return_value = mock.Mock(returncode=0, stdout=json.dumps(manifest_data))

        result = get_docker_manifest("test:latest")
        assert result == manifest_data
        mock_run.assert_called_once_with(
            ["docker", "manifest", "inspect", "test:latest"], capture_output=True, text=True
        )

    @mock.patch("subprocess.run")
    def test_get_docker_manifest_failure(self, mock_run):
        """Test failed Docker manifest fetch."""
        mock_run.return_value = mock.Mock(returncode=1, stdout="")

        result = get_docker_manifest("test:latest")
        assert result is None

    @mock.patch("subprocess.run")
    def test_get_docker_manifest_exception(self, mock_run):
        """Test Docker manifest fetch with exception."""
        mock_run.side_effect = Exception("Command failed")

        result = get_docker_manifest("test:latest")
        assert result is None


class TestGetImageSizeFromManifest:
    """Test extracting image size from manifest."""

    def test_get_size_v2_manifest(self):
        """Test getting size from v2 manifest with layers."""
        manifest = {"layers": [{"size": 1024}, {"size": 2048}, {"size": 512}]}

        size = get_image_size_from_manifest(manifest)
        assert size == 3584  # Sum of all layers

    def test_get_size_manifest_list(self):
        """Test getting size from manifest list."""
        manifest = {
            "manifests": [
                {"size": 5120, "platform": {"os": "linux"}},
                {"size": 4096, "platform": {"os": "windows"}},
            ]
        }

        size = get_image_size_from_manifest(manifest)
        assert size == 5120  # First manifest size

    def test_get_size_empty_manifest(self):
        """Test getting size from empty manifest."""
        manifest = {}

        size = get_image_size_from_manifest(manifest)
        assert size is None

    def test_get_size_invalid_manifest(self):
        """Test getting size from invalid manifest."""
        manifest = {"invalid": "data"}

        size = get_image_size_from_manifest(manifest)
        assert size is None


class TestFetchLockFromRegistry:
    """Test fetching lock data from HUD registry."""

    @mock.patch("requests.get")
    def test_fetch_lock_success(self, mock_get):
        """Test successful lock file fetch."""
        lock_data = {"test": "data"}
        mock_response = mock.Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"lock": yaml.dump(lock_data)}
        mock_get.return_value = mock_response

        result = fetch_lock_from_registry("org/env:latest")
        assert result == lock_data

    @mock.patch("requests.get")
    def test_fetch_lock_adds_latest_tag(self, mock_get):
        """Test that :latest is added if missing."""
        mock_response = mock.Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"lock_data": {"test": "data"}}
        mock_get.return_value = mock_response

        fetch_lock_from_registry("org/env")

        # Check URL includes :latest (URL-encoded)
        call_args = mock_get.call_args
        assert "org/env%3Alatest" in call_args[0][0]

    @mock.patch("hud.cli.pull.settings")
    @mock.patch("requests.get")
    def test_fetch_lock_with_auth(self, mock_get, mock_settings):
        """Test fetching with API key."""
        mock_settings.api_key = "test-key"

        mock_response = mock.Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"test": "data"}
        mock_get.return_value = mock_response

        fetch_lock_from_registry("org/env:latest")

        # Check auth header was set
        call_kwargs = mock_get.call_args[1]
        assert call_kwargs["headers"]["Authorization"] == "Bearer test-key"

    @mock.patch("requests.get")
    def test_fetch_lock_failure(self, mock_get):
        """Test failed lock file fetch."""
        mock_response = mock.Mock()
        mock_response.status_code = 404
        mock_get.return_value = mock_response

        result = fetch_lock_from_registry("org/env:latest")
        assert result is None


class TestFormatSize:
    """Test size formatting."""

    def test_format_bytes(self):
        """Test formatting bytes."""
        assert format_size(512) == "512.0 B"
        assert format_size(1023) == "1023.0 B"

    def test_format_kilobytes(self):
        """Test formatting kilobytes."""
        assert format_size(1024) == "1.0 KB"
        assert format_size(2048) == "2.0 KB"

    def test_format_megabytes(self):
        """Test formatting megabytes."""
        assert format_size(1024 * 1024) == "1.0 MB"
        assert format_size(5 * 1024 * 1024) == "5.0 MB"

    def test_format_gigabytes(self):
        """Test formatting gigabytes."""
        assert format_size(1024 * 1024 * 1024) == "1.0 GB"
        assert format_size(2 * 1024 * 1024 * 1024) == "2.0 GB"

    def test_format_terabytes(self):
        """Test formatting terabytes."""
        assert format_size(1024 * 1024 * 1024 * 1024) == "1.0 TB"


class TestPullEnvironment:
    """Test the main pull_environment function."""

    @mock.patch("hud.cli.pull.HUDConsole")
    @mock.patch("hud.cli.pull.save_to_registry")
    @mock.patch("subprocess.Popen")
    def test_pull_with_lock_file(self, mock_popen, mock_save, mock_hud_console_class, tmp_path):
        """Test pulling with a lock file."""
        # Create mock hud_console instance
        mock_hud_console = mock.Mock()
        mock_hud_console.console = mock.Mock()
        mock_hud_console_class.return_value = mock_hud_console

        # Create lock file
        lock_data = {
            "image": "test/env:latest@sha256:abc123",
            "build": {"generatedAt": "2024-01-01T00:00:00Z", "hudVersion": "1.0.0"},
            "environment": {"toolCount": 5, "initializeMs": 1500},
            "tools": [{"name": "tool1", "description": "Tool 1"}],
        }
        lock_file = tmp_path / "hud.lock.yaml"
        lock_file.write_text(yaml.dump(lock_data))

        # Mock docker pull
        mock_process = mock.Mock()
        mock_process.stdout = ["Pulling test/env:latest...\n", "Pull complete\n"]
        mock_process.wait.return_value = None
        mock_process.returncode = 0
        mock_popen.return_value = mock_process

        # Run pull
        pull_environment(str(lock_file), yes=True)

        # Verify docker pull was called
        mock_popen.assert_called_once()
        call_args = mock_popen.call_args[0][0]
        assert call_args == ["docker", "pull", "test/env:latest@sha256:abc123"]

        # Verify lock was saved to registry
        mock_save.assert_called_once()

    @mock.patch("hud.cli.pull.HUDConsole")
    @mock.patch("hud.cli.pull.fetch_lock_from_registry")
    @mock.patch("subprocess.Popen")
    def test_pull_from_registry(self, mock_popen, mock_fetch, mock_hud_console_class):
        """Test pulling from HUD registry."""
        # Create mock hud_console instance
        mock_hud_console = mock.Mock()
        mock_hud_console.console = mock.Mock()
        mock_hud_console_class.return_value = mock_hud_console

        # Mock registry response
        lock_data = {"image": "docker.io/org/env:latest@sha256:def456", "tools": []}
        mock_fetch.return_value = lock_data

        # Mock docker pull
        mock_process = mock.Mock()
        mock_process.stdout = []
        mock_process.wait.return_value = None
        mock_process.returncode = 0
        mock_popen.return_value = mock_process

        # Run pull
        pull_environment("org/env:latest", yes=True)

        # Verify registry was checked
        mock_fetch.assert_called_once_with("org/env:latest")

        # Verify docker pull was called with registry image
        mock_popen.assert_called_once()
        call_args = mock_popen.call_args[0][0]
        assert "docker.io/org/env:latest@sha256:def456" in call_args

    @mock.patch("hud.cli.pull.HUDConsole")
    @mock.patch("hud.cli.pull.get_docker_manifest")
    @mock.patch("hud.cli.pull.fetch_lock_from_registry")
    @mock.patch("subprocess.Popen")
    def test_pull_docker_image_direct(
        self, mock_popen, mock_fetch, mock_manifest, mock_hud_console_class
    ):
        """Test pulling Docker image directly."""
        # Create mock hud_console instance
        mock_hud_console = mock.Mock()
        mock_hud_console.console = mock.Mock()
        mock_hud_console_class.return_value = mock_hud_console

        # Mock no registry data
        mock_fetch.return_value = None

        # Mock manifest
        mock_manifest.return_value = {"layers": [{"size": 1024}]}

        # Mock docker pull
        mock_process = mock.Mock()
        mock_process.stdout = []
        mock_process.wait.return_value = None
        mock_process.returncode = 0
        mock_popen.return_value = mock_process

        # Run pull
        pull_environment("ubuntu:latest", yes=True)

        # Verify docker pull was called
        mock_popen.assert_called_once()
        call_args = mock_popen.call_args[0][0]
        assert call_args == ["docker", "pull", "ubuntu:latest"]

    @mock.patch("hud.cli.pull.HUDConsole")
    def test_pull_verify_only(self, mock_hud_console_class):
        """Test verify-only mode."""
        # Create mock hud_console instance
        mock_hud_console = mock.Mock()
        mock_hud_console.console = mock.Mock()
        mock_hud_console_class.return_value = mock_hud_console

        # Should not actually pull
        pull_environment("test:latest", verify_only=True)

        # Check success message
        mock_hud_console.success.assert_called_with("Verification complete")

    @mock.patch("hud.cli.pull.HUDConsole")
    @mock.patch("subprocess.Popen")
    def test_pull_docker_failure(self, mock_popen, mock_hud_console_class):
        """Test handling Docker pull failure."""
        # Create mock hud_console instance
        mock_hud_console = mock.Mock()
        mock_hud_console.console = mock.Mock()
        mock_hud_console_class.return_value = mock_hud_console

        # Mock docker pull failure
        mock_process = mock.Mock()
        mock_process.stdout = ["Error: manifest unknown\n"]
        mock_process.wait.return_value = None
        mock_process.returncode = 1
        mock_popen.return_value = mock_process

        # Run pull
        with pytest.raises(typer.Exit):
            pull_environment("invalid:image", yes=True)

        mock_hud_console.error.assert_called_with("Pull failed")

    @mock.patch("hud.cli.pull.HUDConsole")
    @mock.patch("typer.confirm")
    def test_pull_user_cancels(self, mock_confirm, mock_hud_console_class):
        """Test when user cancels pull."""
        # Create mock hud_console instance
        mock_hud_console = mock.Mock()
        mock_hud_console.console = mock.Mock()
        mock_hud_console_class.return_value = mock_hud_console

        mock_confirm.return_value = False

        with pytest.raises(typer.Exit) as exc_info:
            pull_environment("test:latest", yes=False)

        assert exc_info.value.exit_code == 0
        mock_hud_console.info.assert_called_with("Aborted")

    @mock.patch("hud.cli.pull.HUDConsole")
    def test_pull_nonexistent_lock_file(self, mock_hud_console_class):
        """Test pulling with non-existent lock file."""
        # Create mock hud_console instance
        mock_hud_console = mock.Mock()
        mock_hud_console.console = mock.Mock()
        mock_hud_console_class.return_value = mock_hud_console

        with pytest.raises(typer.Exit):
            pull_environment("nonexistent.yaml")


class TestPullCommand:
    """Test the CLI command wrapper."""

    def test_pull_command_basic(self):
        """Test basic pull command runs without error."""
        # Just test it doesn't crash
        # Note: we can't easily test the exact arguments because Typer
        # passes OptionInfo objects as defaults
        with mock.patch("hud.cli.pull.pull_environment"):
            pull_command("test:latest")

    def test_pull_command_with_options(self):
        """Test pull command with options runs without error."""
        # Just test it doesn't crash with explicit values
        with mock.patch("hud.cli.pull.pull_environment"):
            pull_command(
                "org/env:v1.0", lock_file="lock.yaml", yes=True, verify_only=True, verbose=True
            )
