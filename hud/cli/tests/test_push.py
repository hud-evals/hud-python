"""Tests for push.py - Push HUD environments to registry."""

from __future__ import annotations

import base64
import json
import subprocess
from unittest import mock

import pytest
import typer
import yaml

from hud.cli.push import (
    get_docker_image_labels,
    get_docker_username,
    push_command,
    push_environment,
)


class TestGetDockerUsername:
    """Test getting Docker username."""

    def test_get_username_from_config(self, tmp_path):
        """Test getting username from Docker config."""
        # Create mock Docker config
        docker_dir = tmp_path / ".docker"
        docker_dir.mkdir()

        config_file = docker_dir / "config.json"
        config = {
            "auths": {
                "https://index.docker.io/v1/": {
                    "auth": base64.b64encode(b"testuser:testpass").decode()
                }
            }
        }
        config_file.write_text(json.dumps(config))

        with mock.patch("pathlib.Path.home", return_value=tmp_path):
            username = get_docker_username()

        assert username == "testuser"

    def test_get_username_no_config(self, tmp_path):
        """Test when no Docker config exists."""
        with mock.patch("pathlib.Path.home", return_value=tmp_path):
            username = get_docker_username()

        assert username is None

    def test_get_username_token_auth(self, tmp_path):
        """Test skipping token-based auth."""
        docker_dir = tmp_path / ".docker"
        docker_dir.mkdir()

        config_file = docker_dir / "config.json"
        config = {"auths": {"docker.io": {"auth": base64.b64encode(b"token:xyz").decode()}}}
        config_file.write_text(json.dumps(config))

        with mock.patch("pathlib.Path.home", return_value=tmp_path):
            username = get_docker_username()

        assert username is None

    @mock.patch("subprocess.run")
    def test_get_username_credential_helper(self, mock_run, tmp_path):
        """Test getting username from credential helper."""
        docker_dir = tmp_path / ".docker"
        docker_dir.mkdir()

        config_file = docker_dir / "config.json"
        config = {"credsStore": "desktop"}
        config_file.write_text(json.dumps(config))

        # Mock credential helper calls
        mock_run.side_effect = [
            mock.Mock(returncode=0, stdout='{"https://index.docker.io/v1/": "creds"}'),
            mock.Mock(returncode=0, stdout='{"Username": "helperuser", "Secret": "pass"}'),
        ]

        with mock.patch("pathlib.Path.home", return_value=tmp_path):
            username = get_docker_username()

        assert username == "helperuser"


class TestGetDockerImageLabels:
    """Test getting Docker image labels."""

    @mock.patch("subprocess.run")
    def test_get_labels_success(self, mock_run):
        """Test successfully getting image labels."""
        labels = {"org.hud.manifest.head": "abc123", "org.hud.version": "1.0.0"}
        mock_run.return_value = mock.Mock(returncode=0, stdout=json.dumps(labels), stderr="")

        result = get_docker_image_labels("test:latest")
        assert result == labels

    @mock.patch("subprocess.run")
    def test_get_labels_failure(self, mock_run):
        """Test handling failure to get labels."""
        mock_run.side_effect = Exception("Command failed")

        result = get_docker_image_labels("test:latest")
        assert result == {}


class TestPushEnvironment:
    """Test the main push_environment function."""

    @mock.patch("hud.cli.push.HUDConsole")
    def test_push_no_lock_file(self, mock_hud_console_class, tmp_path):
        """Test pushing when no lock file exists."""
        mock_hud_console = mock.Mock()
        mock_hud_console_class.return_value = mock_hud_console

        with pytest.raises(typer.Exit) as exc_info:
            push_environment(str(tmp_path))

        assert exc_info.value.exit_code == 1
        mock_hud_console.error.assert_called()

    @mock.patch("hud.cli.push.HUDConsole")
    @mock.patch("hud.settings.settings")
    def test_push_no_api_key(self, mock_settings, mock_hud_console_class, tmp_path):
        """Test pushing without API key."""
        mock_hud_console = mock.Mock()
        mock_hud_console_class.return_value = mock_hud_console
        mock_settings.api_key = None

        # Create lock file
        lock_file = tmp_path / "hud.lock.yaml"
        lock_file.write_text(yaml.dump({"image": "test:latest"}))

        with pytest.raises(typer.Exit) as exc_info:
            push_environment(str(tmp_path))

        assert exc_info.value.exit_code == 1

    @mock.patch("requests.post")
    @mock.patch("subprocess.Popen")
    @mock.patch("subprocess.run")
    @mock.patch("hud.cli.push.get_docker_username")
    @mock.patch("hud.settings.settings")
    @mock.patch("hud.cli.push.HUDConsole")
    def test_push_auto_detect_username(
        self,
        mock_hud_console_class,
        mock_settings,
        mock_get_username,
        mock_run,
        mock_popen,
        mock_post,
        tmp_path,
    ):
        """Test auto-detecting Docker username and pushing."""
        # Setup mocks
        mock_hud_console = mock.Mock()
        mock_hud_console_class.return_value = mock_hud_console
        mock_settings.api_key = "test-key"
        mock_settings.hud_telemetry_url = "https://api.hud.test"
        mock_get_username.return_value = "testuser"

        # Create lock file
        lock_data = {"image": "original/image:v1.0", "build": {"version": "0.1.0"}}
        lock_file = tmp_path / "hud.lock.yaml"
        lock_file.write_text(yaml.dump(lock_data))

        # Mock docker commands
        def mock_run_impl(*args, **kwargs):
            cmd = args[0]
            if cmd[1] == "inspect":
                if len(cmd) == 3:  # docker inspect <image>
                    return mock.Mock(returncode=0, stdout="")
                else:  # docker inspect --format ... <image>
                    return mock.Mock(returncode=0, stdout="testuser/image:0.1.0@sha256:abc123")
            elif cmd[1] == "tag":
                return mock.Mock(returncode=0)
            return mock.Mock(returncode=0)

        mock_run.side_effect = mock_run_impl

        # Mock docker push
        mock_process = mock.Mock()
        mock_process.stdout = ["Pushing image...", "Push complete"]
        mock_process.wait.return_value = None
        mock_process.returncode = 0
        mock_popen.return_value = mock_process

        # Mock registry upload
        mock_post.return_value = mock.Mock(status_code=201)

        # Run push
        push_environment(str(tmp_path), yes=True)

        # Verify docker commands
        assert mock_run.call_count >= 2
        mock_popen.assert_called_once()

        # Verify registry upload
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert "testuser/image%3A0.1.0" in call_args[0][0]

    @mock.patch("subprocess.run")
    @mock.patch("hud.settings.settings")
    @mock.patch("hud.cli.push.HUDConsole")
    def test_push_explicit_image(self, mock_hud_console_class, mock_settings, mock_run, tmp_path):
        """Test pushing with explicit image name."""
        mock_hud_console = mock.Mock()
        mock_hud_console_class.return_value = mock_hud_console
        mock_settings.api_key = "test-key"

        # Create lock file
        lock_data = {"image": "local:latest"}
        lock_file = tmp_path / "hud.lock.yaml"
        lock_file.write_text(yaml.dump(lock_data))

        # Mock docker inspect for non-existent local image
        mock_run.side_effect = subprocess.CalledProcessError(1, "docker")

        with pytest.raises(typer.Exit):
            push_environment(str(tmp_path), image="myrepo/myimage:v2")

    @mock.patch("subprocess.Popen")
    @mock.patch("subprocess.run")
    @mock.patch("hud.settings.settings")
    @mock.patch("hud.cli.push.HUDConsole")
    def test_push_with_tag(
        self, mock_hud_console_class, mock_settings, mock_run, mock_popen, tmp_path
    ):
        """Test pushing with explicit tag."""
        mock_hud_console = mock.Mock()
        mock_hud_console_class.return_value = mock_hud_console
        mock_settings.api_key = "test-key"

        # Create lock file
        lock_data = {"image": "test:latest"}
        lock_file = tmp_path / "hud.lock.yaml"
        lock_file.write_text(yaml.dump(lock_data))

        # Mock docker commands
        def mock_run_impl(*args, **kwargs):
            cmd = args[0]
            if cmd[1] == "inspect":
                if len(cmd) == 3:  # docker inspect <image>
                    return mock.Mock(returncode=0)
                else:  # docker inspect --format ... <image>
                    return mock.Mock(returncode=0, stdout="user/test:v2.0")
            elif cmd[1] == "tag":
                return mock.Mock(returncode=0)
            return mock.Mock(returncode=0)

        mock_run.side_effect = mock_run_impl

        # Mock docker push
        mock_process = mock.Mock()
        mock_process.stdout = []
        mock_process.wait.return_value = None
        mock_process.returncode = 0
        mock_popen.return_value = mock_process

        # Run push
        push_environment(str(tmp_path), image="user/test", tag="v2.0", yes=True)

        # Verify tag was used
        tag_call = [c for c in mock_run.call_args_list if c[0][0][1] == "tag"]
        assert len(tag_call) > 0
        assert "user/test:v2.0" in tag_call[0][0][0]

    @mock.patch("subprocess.Popen")
    @mock.patch("hud.cli.push.HUDConsole")
    def test_push_docker_failure(self, mock_hud_console_class, mock_popen):
        """Test handling Docker push failure."""
        mock_hud_console = mock.Mock()
        mock_hud_console_class.return_value = mock_hud_console

        # Mock docker push failure
        mock_process = mock.Mock()
        mock_process.stdout = ["Error: access denied"]
        mock_process.wait.return_value = None
        mock_process.returncode = 1
        mock_popen.return_value = mock_process

        with mock.patch("hud.settings.settings") as mock_settings:
            mock_settings.api_key = "test-key"
            with (
                mock.patch("subprocess.run"),
                pytest.raises(typer.Exit),
            ):
                push_environment(".", image="test:latest", yes=True)

    @mock.patch("hud.cli.push.get_docker_image_labels")
    @mock.patch("subprocess.run")
    @mock.patch("hud.settings.settings")
    @mock.patch("hud.cli.push.HUDConsole")
    def test_push_with_labels(
        self, mock_hud_console_class, mock_settings, mock_run, mock_get_labels, tmp_path
    ):
        """Test pushing with image labels."""
        mock_hud_console = mock.Mock()
        mock_hud_console_class.return_value = mock_hud_console
        mock_settings.api_key = "test-key"

        # Create lock file
        lock_data = {"image": "test:latest"}
        lock_file = tmp_path / "hud.lock.yaml"
        lock_file.write_text(yaml.dump(lock_data))

        # Mock labels
        mock_get_labels.return_value = {
            "org.hud.manifest.head": "abc123def456",
            "org.hud.version": "1.2.3",
        }

        # Mock docker commands - first inspect succeeds to get to label check
        # Provide explicit image to bypass username check
        def mock_run_impl(*args, **kwargs):
            cmd = args[0]
            if cmd[1] == "inspect" and len(cmd) == 3:
                # First inspect to check if image exists
                return mock.Mock(returncode=0)
            elif cmd[1] == "tag":
                # Fail on tag to exit after labels are checked
                raise subprocess.CalledProcessError(1, cmd)
            return mock.Mock(returncode=0)

        mock_run.side_effect = mock_run_impl

        # Provide explicit image to ensure we reach label check
        with pytest.raises(subprocess.CalledProcessError):
            push_environment(str(tmp_path), image="test:v2", verbose=True)

        # Verify labels were checked
        mock_get_labels.assert_called_once_with("test:latest")


class TestPushCommand:
    """Test the CLI command wrapper."""

    def test_push_command_basic(self):
        """Test basic push command."""
        with mock.patch("hud.cli.push.push_environment") as mock_push:
            push_command()

            mock_push.assert_called_once_with(".", None, None, False, False, False)

    def test_push_command_with_options(self):
        """Test push command with all options."""
        with mock.patch("hud.cli.push.push_environment") as mock_push:
            push_command(
                directory="./myenv",
                image="myrepo/myimage",
                tag="v1.0",
                sign=True,
                yes=True,
                verbose=True,
            )

            mock_push.assert_called_once_with("./myenv", "myrepo/myimage", "v1.0", True, True, True)
