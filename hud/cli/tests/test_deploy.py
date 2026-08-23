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


@pytest.mark.parametrize(("value", "expected"), [("HUD", "hud"), ("modal", "modal")])
def test_normalize_runtime_uses_public_runtime_names(value: str, expected: str) -> None:
    from hud.cli.deploy import _normalize_runtime

    assert _normalize_runtime(value, HUDConsole()) == expected


def test_normalize_runtime_rejects_internal_provider_name() -> None:
    from hud.cli.deploy import _normalize_runtime

    with pytest.raises(typer.Exit):
        _normalize_runtime("ec2", HUDConsole())


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

    def test_dotted_entrypoint_disambiguates_nested_environment(self, tmp_path: Path) -> None:
        source_dir = tmp_path / "src" / "acme"
        source_dir.mkdir(parents=True)
        (tmp_path / "Dockerfile").write_text(
            'CMD ["hud", "serve", "src.acme.env:env", "--port", "8765"]\n', encoding="utf-8"
        )
        (source_dir / "env.py").write_text(
            'env = Environment("trace-explorer")\n', encoding="utf-8"
        )
        (tmp_path / "verify.py").write_text(
            'verify_env = Environment("qa-verifier")\n', encoding="utf-8"
        )

        assert self._resolve(tmp_path) == "trace-explorer"

    def test_unnamed_environment_exit(self, tmp_path: Path) -> None:
        (tmp_path / "env.py").write_text("env = Environment()\n", encoding="utf-8")

        with pytest.raises(typer.Exit):
            self._resolve(tmp_path)

    def test_no_environment_declaration_exits(self, tmp_path: Path) -> None:
        (tmp_path / "server.py").write_text("x = 1\n", encoding="utf-8")

        with pytest.raises(typer.Exit):
            self._resolve(tmp_path)

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

    def test_registry_id_does_not_replace_missing_declaration(self, tmp_path: Path) -> None:
        (tmp_path / "server.py").write_text("x = 1\n", encoding="utf-8")
        registry_env = RegistryEnvironment(id="r-1", name="platform-name")

        with (
            patch("hud.cli.deploy.get_registry_environment", return_value=registry_env),
            pytest.raises(typer.Exit),
        ):
            self._resolve(tmp_path, registry_id="r-1")


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


@pytest.mark.asyncio
async def test_trigger_build_sends_project_id() -> None:
    from hud.cli.deploy import _DeployPlan, _trigger_build

    platform = AsyncMock(spec=PlatformClient)
    platform.apost.return_value = {"id": "build-1", "registry_id": "registry-1"}
    plan = _DeployPlan(
        name="project-env",
        registry_id=None,
        runtime=None,
        runtime_config=None,
        env_vars={},
        build_args={},
        build_secrets={},
        project_id="project-1",
    )

    await _trigger_build(platform, build_id="build-1", plan=plan, no_cache=False)

    platform.apost.assert_awaited_once_with(
        "/builds/trigger",
        json={
            "source": "direct",
            "build_id": "build-1",
            "name": "project-env",
            "no_cache": False,
            "project_id": "project-1",
        },
    )


class TestRuntimeConfigFile:
    @pytest.mark.parametrize("filename", ["compose.yaml", "compose.yml", "docker-compose.json"])
    def test_prepare_deploy_uses_context_recipe(
        self,
        tmp_path: Path,
        filename: str,
    ) -> None:
        from hud.cli.deploy import _prepare_deploy_plan

        (tmp_path / "env.py").write_text(
            'env = Environment("compose-env")\n',
            encoding="utf-8",
        )
        (tmp_path / filename).write_text(
            "services:\n  main:\n    build: ./main\n  redis:\n    image: redis:7\n",
            encoding="utf-8",
        )

        plan = _prepare_deploy_plan(
            EnvironmentSource.open(tmp_path),
            env_dir=tmp_path,
            env=None,
            env_file=None,
            no_env=True,
            registry_id=None,
            build_args=None,
            build_secrets=None,
            runtime=None,
            runtime_config=None,
            verbose=False,
            platform=PlatformClient("https://api.example", "key"),
            console=HUDConsole(),
        )

        assert plan.runtime_config is not None
        payload = plan.runtime_config.model_dump(mode="json", exclude_unset=True)
        assert payload["compose"]["root"] == {"compose_path": filename}
        assert set(payload["compose"]["document"]["services"]) == {"main", "redis"}

    def test_load_runtime_config_uses_sdk_shape(self, tmp_path: Path) -> None:
        from hud.cli.deploy import _load_runtime_config
        from hud.utils.hud_console import HUDConsole

        config_path = tmp_path / "runtime.json"
        config_path.write_text(
            json.dumps(
                {
                    "resources": {"gpu": {"type": "A10G", "count": 2}},
                    "limits": {"startup_timeout_s": 300},
                }
            ),
            encoding="utf-8",
        )

        config = _load_runtime_config(str(config_path), HUDConsole())
        assert config is not None
        assert config.model_dump(mode="json", exclude_unset=True) == {
            "resources": {"gpu": {"type": "A10G", "count": 2}},
            "limits": {"startup_timeout_s": 300},
        }

    def test_load_runtime_config_preserves_null_override(self, tmp_path: Path) -> None:
        from hud.cli.deploy import _load_runtime_config
        from hud.utils.hud_console import HUDConsole

        config_path = tmp_path / "runtime.json"
        config_path.write_text(json.dumps({"resources": None}), encoding="utf-8")

        config = _load_runtime_config(str(config_path), HUDConsole())
        assert config is not None
        assert config.model_dump(mode="json", exclude_unset=True) == {"resources": None}

    def test_load_runtime_config_resolves_compose_project_from_config_directory(
        self,
        tmp_path: Path,
    ) -> None:
        from hud.cli.deploy import _load_runtime_config

        project = tmp_path / "project"
        project.mkdir()
        compose = project / "compose.json"
        compose.write_text('{"services":{"main":{"image":"postgres:16"}}}')
        config_path = tmp_path / "runtime.json"
        config_path.write_text('{"compose":{"document":"project/compose.json","root":"."}}')

        config = _load_runtime_config(str(config_path), HUDConsole())

        assert config is not None
        assert config.model_dump(mode="json", exclude_unset=True)["compose"]["root"] == {
            "compose_path": "project/compose.json"
        }

    def test_load_runtime_config_rejects_unknown_fields(self, tmp_path: Path) -> None:
        from hud.cli.deploy import _load_runtime_config
        from hud.utils.hud_console import HUDConsole

        config_path = tmp_path / "runtime.json"
        config_path.write_text(json.dumps({"provider_config": {}}), encoding="utf-8")

        with pytest.raises(typer.Exit):
            _load_runtime_config(str(config_path), HUDConsole())

    def test_prepare_deploy_rejects_image_config_for_compose_context(
        self,
        tmp_path: Path,
    ) -> None:
        from hud.cli.deploy import _prepare_deploy_plan

        (tmp_path / "env.py").write_text(
            'env = Environment("compose-env")\n',
            encoding="utf-8",
        )
        (tmp_path / "compose.yaml").write_text("services:\n  main:\n    image: example\n")
        runtime_config = tmp_path / "runtime.json"
        runtime_config.write_text('{"image":"other:latest"}', encoding="utf-8")

        with pytest.raises(typer.Exit):
            _prepare_deploy_plan(
                EnvironmentSource.open(tmp_path),
                env_dir=tmp_path,
                env=None,
                env_file=None,
                no_env=True,
                registry_id=None,
                build_args=None,
                build_secrets=None,
                runtime=None,
                runtime_config=str(runtime_config),
                verbose=False,
                platform=PlatformClient("https://api.example", "key"),
                console=HUDConsole(),
            )


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

    def test_compose_recipe_does_not_require_a_dockerfile(self, tmp_path: Path) -> None:
        from hud.cli.deploy import _compose_recipe

        (tmp_path / "compose.yaml").write_text("services: {main: {image: alpine}}\n")

        assert _compose_recipe(tmp_path) == tmp_path / "compose.yaml"

    def test_compose_recipe_prefers_base_file_over_override(self, tmp_path: Path) -> None:
        from hud.cli.deploy import _compose_recipe

        base = tmp_path / "docker-compose.yml"
        base.write_text("services: {main: {image: alpine}}\n")
        (tmp_path / "docker-compose.override.yml").write_text(
            "services: {main: {environment: {DEBUG: '1'}}}\n"
        )

        assert _compose_recipe(tmp_path) == base

    def test_compose_recipe_ignores_override_without_base(self, tmp_path: Path) -> None:
        from hud.cli.deploy import _compose_recipe

        (tmp_path / "docker-compose.override.yml").write_text(
            "services: {main: {environment: {DEBUG: '1'}}}\n"
        )

        assert _compose_recipe(tmp_path) is None

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
                    runtime_config=None,
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
                    runtime_config=None,
                    env_vars={},
                    build_args={},
                    build_secrets={},
                ),
                platform=PlatformClient("https://api.example", "key"),
                console=console,
            )

        assert result.success is False


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
