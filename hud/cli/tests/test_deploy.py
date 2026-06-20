"""Tests for CLI deploy command."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
import typer

from hud.cli.deploy import _resolve_environment_name
from hud.cli.utils.registry import RegistryEnvironment
from hud.cli.utils.source import EnvironmentSource
from hud.utils.hud_console import HUDConsole
from hud.utils.platform import PlatformClient


class TestResolveEnvironmentName:
    """Tests for code-authoritative environment name resolution."""

    @staticmethod
    def _resolve(tmp_path: Path, registry_id: str | None = None) -> str:
        return _resolve_environment_name(
            EnvironmentSource.open(tmp_path),
            registry_id,
            PlatformClient("https://api.example", "key"),
            HUDConsole(),
        )

    def test_single_declared_name_wins(self, tmp_path: Path) -> None:
        (tmp_path / "env.py").write_text('env = Environment("my-env")\n', encoding="utf-8")

        assert self._resolve(tmp_path) == "my-env"

    def test_repeated_same_name_is_fine(self, tmp_path: Path) -> None:
        (tmp_path / "a.py").write_text('a = Environment("same")\n', encoding="utf-8")
        (tmp_path / "b.py").write_text('b = Environment(name="same")\n', encoding="utf-8")

        assert self._resolve(tmp_path) == "same"

    def test_multiple_distinct_names_exit(self, tmp_path: Path) -> None:
        (tmp_path / "a.py").write_text('a = Environment("one")\n', encoding="utf-8")
        (tmp_path / "b.py").write_text('b = Environment("two")\n', encoding="utf-8")

        with pytest.raises(typer.Exit):
            self._resolve(tmp_path)

    def test_entrypoint_disambiguates_subagent(self, tmp_path: Path) -> None:
        (tmp_path / "Dockerfile").write_text(
            'CMD ["hud", "serve", "env:env", "--port", "8765"]\n', encoding="utf-8"
        )
        (tmp_path / "env.py").write_text('env = Environment("trace-explorer")\n', encoding="utf-8")
        (tmp_path / "verify.py").write_text(
            'verify_env = Environment("qa-verifier")\n', encoding="utf-8"
        )

        assert self._resolve(tmp_path) == "trace-explorer"

    def test_unnamed_environment_exit(self, tmp_path: Path) -> None:
        (tmp_path / "env.py").write_text("env = Environment()\n", encoding="utf-8")

        with pytest.raises(typer.Exit):
            self._resolve(tmp_path)

    def test_no_references_falls_back_to_directory(self, tmp_path: Path) -> None:
        env_dir = tmp_path / "My Legacy_Env"
        env_dir.mkdir()
        (env_dir / "server.py").write_text("x = 1\n", encoding="utf-8")

        assert self._resolve(env_dir) == "my-legacy-env"

    def test_registry_id_name_mismatch_exit(self, tmp_path: Path) -> None:
        (tmp_path / "env.py").write_text('env = Environment("code-name")\n', encoding="utf-8")
        registry_env = RegistryEnvironment(id="r-1", name="other-name")

        with (
            patch(
                "hud.cli.deploy.get_registry_environment",
                return_value=registry_env,
            ),
            pytest.raises(typer.Exit),
        ):
            self._resolve(tmp_path, registry_id="r-1")

    def test_registry_id_matching_name_passes(self, tmp_path: Path) -> None:
        (tmp_path / "env.py").write_text('env = Environment("Code Name")\n', encoding="utf-8")
        registry_env = RegistryEnvironment(id="r-1", name="code-name")

        with patch(
            "hud.cli.deploy.get_registry_environment",
            return_value=registry_env,
        ):
            assert self._resolve(tmp_path, registry_id="r-1") == "Code Name"

    def test_registry_id_supplies_name_for_legacy_env(self, tmp_path: Path) -> None:
        (tmp_path / "server.py").write_text("x = 1\n", encoding="utf-8")
        registry_env = RegistryEnvironment(id="r-1", name="platform-name")

        with patch(
            "hud.cli.deploy.get_registry_environment",
            return_value=registry_env,
        ):
            assert self._resolve(tmp_path, registry_id="r-1") == "platform-name"


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
        from hud.cli.deploy import deploy_environment

        # Create a Dockerfile
        (tmp_path / "Dockerfile.hud").write_text("FROM python:3.12")

        with (
            patch("hud.settings.settings") as mock_settings,
            pytest.raises(typer.Exit) as exc_info,
        ):
            mock_settings.api_key = None

            deploy_environment(directory=str(tmp_path))

        assert exc_info.value.exit_code == 1

    def test_no_dockerfile_error(self, tmp_path: Path) -> None:
        """Test error when no Dockerfile found."""
        from hud.cli.deploy import deploy_environment

        with (
            patch("hud.settings.settings") as mock_settings,
            pytest.raises(typer.Exit) as exc_info,
        ):
            mock_settings.api_key = "test-key"

            deploy_environment(directory=str(tmp_path))

        assert exc_info.value.exit_code == 1

    def test_validation_errors_exit(self, tmp_path: Path) -> None:
        """Test that validation errors cause exit."""
        from hud.cli.deploy import deploy_environment
        from hud.cli.utils.source import ValidationIssue

        (tmp_path / "Dockerfile.hud").write_text("FROM python:3.12")

        with (
            patch("hud.settings.settings") as mock_settings,
            patch("hud.cli.utils.source.EnvironmentSource.validate") as mock_validate,
            pytest.raises(typer.Exit) as exc_info,
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
        from hud.cli.deploy import _deploy_async, _DeployPlan
        from hud.utils.exceptions import HudRequestError
        from hud.utils.hud_console import HUDConsole
        from hud.utils.platform import PlatformClient

        console = HUDConsole()
        error = HudRequestError("Unauthorized", status_code=401)

        with patch("hud.utils.platform.make_request", AsyncMock(side_effect=error)):
            result = await _deploy_async(
                tarball_path=Path("test.tar.gz"),
                no_cache=False,
                plan=_DeployPlan(
                    name="test-env",
                    registry_id=None,
                    runtime=None,
                    env_vars={},
                    build_args={},
                    build_secrets={},
                ),
                platform=PlatformClient("https://api.example", "key"),
                console=console,
            )

        assert result.success is False

    @pytest.mark.asyncio
    async def test_upload_url_network_error(self) -> None:
        """Test handling of network error during upload URL fetch."""
        from hud.cli.deploy import _deploy_async, _DeployPlan
        from hud.utils.hud_console import HUDConsole
        from hud.utils.platform import PlatformClient

        console = HUDConsole()

        with patch(
            "hud.utils.platform.make_request",
            AsyncMock(side_effect=Exception("Network error")),
        ):
            result = await _deploy_async(
                tarball_path=Path("test.tar.gz"),
                no_cache=False,
                plan=_DeployPlan(
                    name="test-env",
                    registry_id=None,
                    runtime=None,
                    env_vars={},
                    build_args={},
                    build_secrets={},
                ),
                platform=PlatformClient("https://api.example", "key"),
                console=console,
            )

        assert result.success is False

    @pytest.mark.asyncio
    async def test_trigger_build_sends_runtime_provider(self) -> None:
        """Test deploy runtime flag maps to the platform runtime_provider field."""
        from hud.cli.deploy import _DeployPlan, _trigger_build
        from hud.utils.hud_console import HUDConsole
        from hud.utils.platform import PlatformClient

        class FakePlatform(PlatformClient):
            payload: dict[str, object] | None = None

            async def apost(
                self,
                path: str,
                *,
                json: object | None = None,
            ) -> dict[str, object]:
                assert path == "/builds/trigger"
                assert isinstance(json, dict)
                object.__setattr__(self, "payload", json)
                return {"id": "build-1", "registry_id": "registry-1"}

        platform = FakePlatform("https://api.example", "key")
        result = await _trigger_build(
            platform,
            build_id="build-1",
            plan=_DeployPlan(
                name="test-env",
                registry_id=None,
                runtime="modal",
                env_vars={},
                build_args={},
                build_secrets={},
            ),
            no_cache=False,
            console=HUDConsole(),
        )

        assert result == {"id": "build-1", "registry_id": "registry-1"}
        assert platform.payload is not None
        assert platform.payload["runtime_provider"] == "modal"


class TestSaveDeployLink:
    """Tests for _save_deploy_link function."""

    def test_saves_deploy_link(self, tmp_path: Path) -> None:
        """Test saving deploy link creates correct config.json file."""
        from hud.cli.deploy import _save_deploy_link
        from hud.utils.hud_console import HUDConsole

        console = HUDConsole()

        _save_deploy_link(tmp_path, "test-registry-id-12345", console)

        config_path = tmp_path / ".hud" / "config.json"
        assert config_path.exists()

        with open(config_path) as f:
            saved = json.load(f)

        assert saved["registryId"] == "test-registry-id-12345"

    def test_creates_hud_directory(self, tmp_path: Path) -> None:
        """Test that .hud directory is created if missing."""
        from hud.cli.deploy import _save_deploy_link
        from hud.utils.hud_console import HUDConsole

        console = HUDConsole()

        _save_deploy_link(tmp_path, "test-id", console)

        assert (tmp_path / ".hud").is_dir()


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
