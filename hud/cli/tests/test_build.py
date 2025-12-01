"""Tests for build.py - Build HUD environments and generate lock files."""

from __future__ import annotations

import subprocess
from unittest import mock

import pytest
import typer
import yaml

from hud.cli.build import (
    analyze_mcp_environment,
    build_docker_image,
    build_environment,
    extract_env_vars_from_dockerfile,
    get_docker_image_digest,
    get_docker_image_id,
    get_existing_version,
    increment_version,
    parse_version,
)


class TestParseVersion:
    """Test version parsing functionality."""

    def test_parse_standard_version(self):
        """Test parsing standard semantic version."""
        assert parse_version("1.2.3") == (1, 2, 3)
        assert parse_version("10.20.30") == (10, 20, 30)

    def test_parse_version_with_v_prefix(self):
        """Test parsing version with v prefix."""
        assert parse_version("v1.2.3") == (1, 2, 3)
        assert parse_version("v2.0.0") == (2, 0, 0)

    def test_parse_incomplete_version(self):
        """Test parsing versions with missing parts."""
        assert parse_version("1.2") == (1, 2, 0)
        assert parse_version("1") == (1, 0, 0)
        assert parse_version("") == (0, 0, 0)

    def test_parse_invalid_version(self):
        """Test parsing invalid versions."""
        assert parse_version("abc") == (0, 0, 0)
        assert parse_version("1.x.3") == (0, 0, 0)
        assert parse_version("not-a-version") == (0, 0, 0)


class TestIncrementVersion:
    """Test version incrementing functionality."""

    def test_increment_patch(self):
        """Test incrementing patch version."""
        assert increment_version("1.2.3") == "1.2.4"
        assert increment_version("1.2.3", "patch") == "1.2.4"
        assert increment_version("1.0.0") == "1.0.1"

    def test_increment_minor(self):
        """Test incrementing minor version."""
        assert increment_version("1.2.3", "minor") == "1.3.0"
        assert increment_version("0.5.10", "minor") == "0.6.0"

    def test_increment_major(self):
        """Test incrementing major version."""
        assert increment_version("1.2.3", "major") == "2.0.0"
        assert increment_version("0.5.10", "major") == "1.0.0"

    def test_increment_with_v_prefix(self):
        """Test incrementing version with v prefix."""
        assert increment_version("v1.2.3") == "1.2.4"
        assert increment_version("v2.0.0", "major") == "3.0.0"


class TestGetExistingVersion:
    """Test getting version from lock file."""

    def test_get_version_from_lock(self, tmp_path):
        """Test extracting version from lock file."""
        lock_data = {"build": {"version": "1.2.3"}}
        lock_path = tmp_path / "hud.lock.yaml"
        lock_path.write_text(yaml.dump(lock_data))

        assert get_existing_version(lock_path) == "1.2.3"

    def test_get_version_no_build_section(self, tmp_path):
        """Test when lock file has no build section."""
        lock_data = {"other": "data"}
        lock_path = tmp_path / "hud.lock.yaml"
        lock_path.write_text(yaml.dump(lock_data))

        assert get_existing_version(lock_path) is None

    def test_get_version_no_file(self, tmp_path):
        """Test when lock file doesn't exist."""
        lock_path = tmp_path / "hud.lock.yaml"
        assert get_existing_version(lock_path) is None

    def test_get_version_invalid_yaml(self, tmp_path):
        """Test when lock file has invalid YAML."""
        lock_path = tmp_path / "hud.lock.yaml"
        lock_path.write_text("invalid: yaml: content:")
        assert get_existing_version(lock_path) is None


class TestGetDockerImageDigest:
    """Test getting Docker image digest."""

    @mock.patch("subprocess.run")
    def test_get_digest_success(self, mock_run):
        """Test successfully getting image digest."""
        # Note: The function expects to parse a list from the string representation
        mock_run.return_value = mock.Mock(
            stdout="['docker.io/library/test@sha256:abc123']", returncode=0
        )

        result = get_docker_image_digest("test:latest")
        assert result == "docker.io/library/test@sha256:abc123"

    @mock.patch("subprocess.run")
    def test_get_digest_empty(self, mock_run):
        """Test when docker returns empty digest list."""
        mock_run.return_value = mock.Mock(stdout="[]", returncode=0)

        result = get_docker_image_digest("test:latest")
        assert result is None

    @mock.patch("subprocess.run")
    def test_get_digest_failure(self, mock_run):
        """Test when docker command fails."""
        mock_run.side_effect = subprocess.CalledProcessError(1, ["docker"])

        result = get_docker_image_digest("test:latest")
        assert result is None


class TestGetDockerImageId:
    """Test getting Docker image ID."""

    @mock.patch("subprocess.run")
    def test_get_id_success(self, mock_run):
        """Test successfully getting image ID."""
        mock_run.return_value = mock.Mock(stdout="sha256:abc123def456", returncode=0)

        result = get_docker_image_id("test:latest")
        assert result == "sha256:abc123def456"

    @mock.patch("subprocess.run")
    def test_get_id_empty(self, mock_run):
        """Test when docker returns empty ID."""
        mock_run.return_value = mock.Mock(stdout="", returncode=0)

        result = get_docker_image_id("test:latest")
        assert result is None

    @mock.patch("subprocess.run")
    def test_get_id_failure(self, mock_run):
        """Test when docker command fails."""
        mock_run.side_effect = subprocess.CalledProcessError(1, ["docker"])

        result = get_docker_image_id("test:latest")
        assert result is None


class TestExtractEnvVarsFromDockerfile:
    """Test extracting environment variables from Dockerfile."""

    def test_extract_required_env_vars(self, tmp_path):
        """Test extracting required environment variables."""
        dockerfile = tmp_path / "Dockerfile"
        dockerfile.write_text("""
FROM python:3.11
ENV API_KEY
ENV SECRET_TOKEN=
ENV OTHER_VAR=default_value
""")

        required, optional = extract_env_vars_from_dockerfile(dockerfile)
        assert "API_KEY" in required
        assert "SECRET_TOKEN" in required
        assert "OTHER_VAR" not in required
        assert len(optional) == 0

    def test_extract_no_env_vars(self, tmp_path):
        """Test Dockerfile with no ENV directives."""
        dockerfile = tmp_path / "Dockerfile"
        dockerfile.write_text("""
FROM python:3.11
RUN pip install fastmcp
""")

        required, optional = extract_env_vars_from_dockerfile(dockerfile)
        assert len(required) == 0
        assert len(optional) == 0

    def test_extract_no_dockerfile(self, tmp_path):
        """Test when Dockerfile doesn't exist."""
        dockerfile = tmp_path / "Dockerfile"
        required, optional = extract_env_vars_from_dockerfile(dockerfile)
        assert len(required) == 0
        assert len(optional) == 0


@pytest.mark.asyncio
class TestAnalyzeMcpEnvironment:
    """Test analyzing MCP environment."""

    @mock.patch("hud.cli.build.MCPClient")
    async def test_analyze_success(self, mock_client_class):
        """Test successful environment analysis."""
        # Setup mock client
        mock_client = mock.AsyncMock()
        mock_client_class.return_value = mock_client

        # Mock tool
        mock_tool = mock.Mock()
        mock_tool.name = "test_tool"
        mock_tool.description = "Test tool"
        mock_tool.inputSchema = {"type": "object"}

        # Prefer analyze_environment path (aligns with analyze CLI tests)
        mock_client.analyze_environment = mock.AsyncMock(
            return_value={
                "metadata": {"servers": ["local"], "initialized": True},
                "tools": [{"name": "test_tool", "description": "Test tool"}],
                "hub_tools": {},
                "resources": [],
                "telemetry": {},
            }
        )
        # Fallback still defined for completeness
        mock_client.list_tools.return_value = [mock_tool]

        result = await analyze_mcp_environment("test:latest")

        assert result["success"] is True
        assert result["toolCount"] == 1
        assert len(result["tools"]) == 1
        assert result["tools"][0]["name"] == "test_tool"
        assert "initializeMs" in result

    @mock.patch("hud.cli.build.MCPClient")
    async def test_analyze_failure(self, mock_client_class):
        """Test failed environment analysis."""
        # Setup mock client to fail
        mock_client = mock.AsyncMock()
        mock_client_class.return_value = mock_client
        mock_client.initialize.side_effect = ConnectionError("Connection failed")

        from hud.shared.exceptions import HudException

        with pytest.raises(HudException, match="Connection failed"):
            await analyze_mcp_environment("test:latest")

    @mock.patch("hud.cli.build.MCPClient")
    async def test_analyze_verbose_mode(self, mock_client_class):
        """Test analysis in verbose mode."""
        mock_client = mock.AsyncMock()
        mock_client_class.return_value = mock_client
        mock_client.analyze_environment = mock.AsyncMock(
            return_value={
                "metadata": {"servers": ["local"], "initialized": True},
                "tools": [],
                "hub_tools": {},
                "resources": [],
                "telemetry": {},
            }
        )
        mock_client.list_tools.return_value = []

        # Just test that it runs without error in verbose mode
        result = await analyze_mcp_environment("test:latest", verbose=True)

        assert result["success"] is True
        assert "initializeMs" in result


class TestBuildDockerImage:
    """Test building Docker images."""

    @mock.patch("subprocess.run")
    def test_build_success(self, mock_run, tmp_path):
        """Test successful Docker build."""
        # Create Dockerfile
        dockerfile = tmp_path / "Dockerfile"
        dockerfile.write_text("FROM python:3.11")

        # Mock successful process
        mock_result = mock.Mock()
        mock_result.returncode = 0
        mock_run.return_value = mock_result

        result = build_docker_image(tmp_path, "test:latest")
        assert result is True

    @mock.patch("subprocess.run")
    def test_build_failure(self, mock_run, tmp_path):
        """Test failed Docker build."""
        dockerfile = tmp_path / "Dockerfile"
        dockerfile.write_text("FROM python:3.11")

        # Mock failed process
        mock_result = mock.Mock()
        mock_result.returncode = 1
        mock_run.return_value = mock_result

        result = build_docker_image(tmp_path, "test:latest")
        assert result is False

    def test_build_no_dockerfile(self, tmp_path):
        """Test build when Dockerfile missing."""
        result = build_docker_image(tmp_path, "test:latest")
        assert result is False

    @mock.patch("subprocess.run")
    def test_build_with_no_cache(self, mock_run, tmp_path):
        """Test build with --no-cache flag."""
        dockerfile = tmp_path / "Dockerfile"
        dockerfile.write_text("FROM python:3.11")

        mock_result = mock.Mock()
        mock_result.returncode = 0
        mock_run.return_value = mock_result

        build_docker_image(tmp_path, "test:latest", no_cache=True)

        # Check that --no-cache was included
        call_args = mock_run.call_args[0][0]
        assert "--no-cache" in call_args


class TestBuildEnvironment:
    """Test the main build_environment function."""

    @mock.patch("hud.cli.build.build_docker_image")
    @mock.patch("hud.cli.build.collect_runtime_metadata")
    @mock.patch("hud.cli.build.analyze_mcp_environment")
    @mock.patch("hud.cli.build.save_to_registry")
    @mock.patch("hud.cli.build.get_docker_image_id")
    @mock.patch("subprocess.run")
    def test_build_environment_success(
        self,
        mock_run,
        mock_get_id,
        mock_save_registry,
        mock_analyze,
        mock_collect_runtime,
        mock_build_docker,
        tmp_path,
    ):
        """Test successful environment build."""
        # Setup directory structure
        env_dir = tmp_path / "test-env"
        env_dir.mkdir()

        # Create pyproject.toml
        pyproject = env_dir / "pyproject.toml"
        pyproject.write_text("""
[tool.hud]
image = "test/env:dev"
""")

        # Create Dockerfile
        dockerfile = env_dir / "Dockerfile"
        dockerfile.write_text("""
FROM python:3.11
ENV API_KEY
""")

        # Mock functions
        mock_build_docker.return_value = True
        mock_analyze.return_value = {
            "success": True,
            "toolCount": 2,
            "initializeMs": 1500,
            "tools": [
                {"name": "tool1", "description": "Tool 1"},
                {"name": "tool2", "description": "Tool 2"},
            ],
        }
        mock_get_id.return_value = "sha256:abc123"
        mock_collect_runtime.return_value = {
            "python": "3.11.6",
            "cuda": None,
            "cudnn": None,
            "pytorch": None,
        }

        # Mock final rebuild
        mock_result = mock.Mock()
        mock_result.returncode = 0
        mock_run.return_value = mock_result

        # Run build
        build_environment(str(env_dir), "test-env:latest")

        # Check lock file was created
        lock_file = env_dir / "hud.lock.yaml"
        assert lock_file.exists()

        # Verify lock file content
        with open(lock_file) as f:
            lock_data = yaml.safe_load(f)

        assert lock_data["images"]["full"] == "test-env:0.1.0@sha256:abc123"
        assert lock_data["images"]["local"] == "test-env:0.1.0"
        assert lock_data["build"]["version"] == "0.1.0"
        assert lock_data["build"]["baseImage"] == "python:3.11"
        assert lock_data["build"]["platform"] == "linux/amd64"
        assert lock_data["environment"]["toolCount"] == 2
        assert lock_data["environment"]["runtime"]["python"] == "3.11.6"
        assert len(lock_data["tools"]) == 2

    @mock.patch("hud.cli.build.build_docker_image")
    @mock.patch("hud.cli.build.collect_runtime_metadata")
    @mock.patch("hud.cli.build.analyze_mcp_environment")
    @mock.patch("hud.cli.build.save_to_registry")
    @mock.patch("hud.cli.build.get_docker_image_id")
    @mock.patch("subprocess.run")
    def test_build_environment_internal_tools(
        self,
        mock_run,
        mock_get_id,
        mock_save_registry,
        mock_analyze,
        mock_collect_runtime,
        mock_build_docker,
        tmp_path,
    ):
        """Dispatcher tools should include internalTools in lock, with count."""
        env_dir = tmp_path / "env-int"
        env_dir.mkdir()
        (env_dir / "pyproject.toml").write_text("""
[tool.hud]
image = "test/env:dev"
""")
        dockerfile = env_dir / "Dockerfile"
        dockerfile.write_text("""
FROM python:3.11
""")

        mock_build_docker.return_value = True
        mock_analyze.return_value = {
            "success": True,
            "toolCount": 1,
            "internalToolCount": 2,
            "initializeMs": 500,
            "tools": [
                {
                    "name": "setup",
                    "description": "setup dispatcher",
                    "inputSchema": {"type": "object"},
                    "internalTools": ["board", "seed"],
                }
            ],
        }
        mock_get_id.return_value = "sha256:fff111"
        mock_collect_runtime.return_value = {
            "python": "3.11.6",
            "cuda": None,
            "cudnn": None,
            "pytorch": None,
        }

        mock_result = mock.Mock()
        mock_result.returncode = 0
        mock_run.return_value = mock_result

        build_environment(str(env_dir), "env-int:latest")

        lock_file = env_dir / "hud.lock.yaml"
        with open(lock_file) as f:
            data = yaml.safe_load(f)
        assert data["environment"]["internalToolCount"] == 2
        assert data["tools"][0]["name"] == "setup"
        assert data["tools"][0]["internalTools"] == ["board", "seed"]

    def test_build_environment_no_directory(self):
        """Test build when directory doesn't exist."""
        with pytest.raises(typer.Exit):
            build_environment("/nonexistent/path")

    def test_build_environment_no_pyproject(self, tmp_path):
        """Test build when pyproject.toml missing."""
        with pytest.raises(typer.Exit):
            build_environment(str(tmp_path))

    @mock.patch("hud.cli.build.build_docker_image")
    def test_build_environment_docker_failure(self, mock_build, tmp_path):
        """Test when Docker build fails."""
        env_dir = tmp_path / "test-env"
        env_dir.mkdir()
        (env_dir / "pyproject.toml").write_text("[tool.hud]")
        (env_dir / "Dockerfile").write_text("FROM python:3.11")

        mock_build.return_value = False

        with pytest.raises(typer.Exit):
            build_environment(str(env_dir))
