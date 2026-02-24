"""Tests for CLI deploy command."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestCollectEnvironmentVariables:
    """Tests for collect_environment_variables function."""

    def test_empty_sources(self, tmp_path: Path) -> None:
        """Test with no env sources."""
        from hud.cli.deploy import collect_environment_variables
        from hud.utils.hud_console import HUDConsole

        console = HUDConsole()
        result = collect_environment_variables(tmp_path, None, None, console)
        assert result == {}

    def test_env_file_loading(self, tmp_path: Path) -> None:
        """Test loading from .env file."""
        from hud.cli.deploy import collect_environment_variables
        from hud.utils.hud_console import HUDConsole

        env_file = tmp_path / ".env"
        env_file.write_text("KEY1=value1\nKEY2=value2\n")

        console = HUDConsole()
        result = collect_environment_variables(tmp_path, None, None, console)

        assert result["KEY1"] == "value1"
        assert result["KEY2"] == "value2"

    def test_custom_env_file(self, tmp_path: Path) -> None:
        """Test loading from custom env file."""
        from hud.cli.deploy import collect_environment_variables
        from hud.utils.hud_console import HUDConsole

        custom_env = tmp_path / "custom.env"
        custom_env.write_text("CUSTOM_KEY=custom_value\n")

        console = HUDConsole()
        result = collect_environment_variables(tmp_path, None, str(custom_env), console)

        assert result["CUSTOM_KEY"] == "custom_value"

    def test_env_flags_override(self, tmp_path: Path) -> None:
        """Test --env flags override file values."""
        from hud.cli.deploy import collect_environment_variables
        from hud.utils.hud_console import HUDConsole

        env_file = tmp_path / ".env"
        env_file.write_text("KEY1=file_value\n")

        console = HUDConsole()
        result = collect_environment_variables(
            tmp_path,
            ["KEY1=flag_value", "KEY2=new_value"],
            None,
            console,
        )

        assert result["KEY1"] == "flag_value"
        assert result["KEY2"] == "new_value"

    def test_env_flag_invalid_format(self, tmp_path: Path) -> None:
        """Test invalid --env flag format is warned."""
        from hud.cli.deploy import collect_environment_variables
        from hud.utils.hud_console import HUDConsole

        console = HUDConsole()
        result = collect_environment_variables(
            tmp_path,
            ["INVALID_FORMAT"],  # Missing =
            None,
            console,
        )

        # Invalid format should be skipped
        assert "INVALID_FORMAT" not in result


class TestDeployEnvironment:
    """Tests for deploy_environment function."""

    def test_no_api_key_error(self, tmp_path: Path) -> None:
        """Test error when no API key is set."""
        import click

        from hud.cli.deploy import deploy_environment

        # Create a Dockerfile
        (tmp_path / "Dockerfile.hud").write_text("FROM python:3.12")

        with (
            patch("hud.settings.settings") as mock_settings,
            pytest.raises(click.exceptions.Exit) as exc_info,
        ):
            mock_settings.api_key = None

            deploy_environment(directory=str(tmp_path))

        assert exc_info.value.exit_code == 1

    def test_no_dockerfile_error(self, tmp_path: Path) -> None:
        """Test error when no Dockerfile found."""
        import click

        from hud.cli.deploy import deploy_environment

        with (
            patch("hud.settings.settings") as mock_settings,
            pytest.raises(click.exceptions.Exit) as exc_info,
        ):
            mock_settings.api_key = "test-key"

            deploy_environment(directory=str(tmp_path))

        assert exc_info.value.exit_code == 1

    def test_validation_errors_exit(self, tmp_path: Path) -> None:
        """Test that validation errors cause exit."""
        import click

        from hud.cli.deploy import deploy_environment
        from hud.cli.utils.validation import ValidationIssue

        (tmp_path / "Dockerfile.hud").write_text("FROM python:3.12")

        with (
            patch("hud.settings.settings") as mock_settings,
            patch("hud.cli.deploy.validate_environment") as mock_validate,
            pytest.raises(click.exceptions.Exit) as exc_info,
        ):
            mock_settings.api_key = "test-key"
            mock_validate.return_value = [
                ValidationIssue(
                    severity="error",
                    message="Test error",
                    file="test.py",
                    hint="Fix this",
                )
            ]

            deploy_environment(directory=str(tmp_path))

        assert exc_info.value.exit_code == 1


class TestDeployAsync:
    """Tests for _deploy_async function."""

    @pytest.mark.asyncio
    async def test_upload_url_failure(self) -> None:
        """Test handling of upload URL failure."""
        import httpx

        from hud.cli.deploy import _deploy_async
        from hud.utils.hud_console import HUDConsole

        console = HUDConsole()

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client

            # Simulate HTTP error
            mock_response = MagicMock()
            mock_response.status_code = 401
            mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
                "Unauthorized", request=MagicMock(), response=mock_response
            )
            mock_client.post.return_value = mock_response

            result = await _deploy_async(
                tarball_path=Path("test.tar.gz"),
                name="test-env",
                env_vars={},
                build_args={},
                build_secrets={},
                no_cache=False,
                registry_id=None,
                console=console,
            )

        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_upload_url_network_error(self) -> None:
        """Test handling of network error during upload URL fetch."""
        from hud.cli.deploy import _deploy_async
        from hud.utils.hud_console import HUDConsole

        console = HUDConsole()

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client

            # Simulate network error
            mock_client.post.side_effect = Exception("Network error")

            result = await _deploy_async(
                tarball_path=Path("test.tar.gz"),
                name="test-env",
                env_vars={},
                build_args={},
                build_secrets={},
                no_cache=False,
                registry_id=None,
                console=console,
            )

        assert result["success"] is False


class TestSaveDeployLink:
    """Tests for _save_deploy_link function."""

    def test_saves_deploy_link(self, tmp_path: Path) -> None:
        """Test saving deploy link creates correct file."""
        from hud.cli.deploy import _save_deploy_link
        from hud.utils.hud_console import HUDConsole

        console = HUDConsole()
        result = {
            "registry_id": "test-registry-id-12345",
            "version": "1.0.0",
        }

        _save_deploy_link(tmp_path, result, console)

        deploy_link_path = tmp_path / ".hud" / "deploy.json"
        assert deploy_link_path.exists()

        with open(deploy_link_path) as f:
            saved = json.load(f)

        assert saved["registryId"] == "test-registry-id-12345"
        assert saved["version"] == "1.0.0"

    def test_creates_hud_directory(self, tmp_path: Path) -> None:
        """Test that .hud directory is created if missing."""
        from hud.cli.deploy import _save_deploy_link
        from hud.utils.hud_console import HUDConsole

        console = HUDConsole()
        result = {"registry_id": "test-id"}

        _save_deploy_link(tmp_path, result, console)

        assert (tmp_path / ".hud").is_dir()

    def test_handles_missing_registry_id(self, tmp_path: Path) -> None:
        """Test handling when registry_id is None."""
        from hud.cli.deploy import _save_deploy_link
        from hud.utils.hud_console import HUDConsole

        console = HUDConsole()
        result = {"registry_id": None, "version": "1.0.0"}

        # Should not raise
        _save_deploy_link(tmp_path, result, console)


class TestDeployCommand:
    """Tests for deploy_command typer function."""

    def test_command_exists(self) -> None:
        """Test deploy_command function exists and is callable."""
        from hud.cli.deploy import deploy_command

        assert callable(deploy_command)

    def test_command_docstring(self) -> None:
        """Test deploy_command has proper docstring."""
        from hud.cli.deploy import deploy_command

        assert deploy_command.__doc__ is not None
        assert "Deploy" in deploy_command.__doc__
