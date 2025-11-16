"""Tests for the convert command."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest
import typer

from hud.cli.flows.tasks import convert_tasks_to_remote
from hud.types import Task


class TestConvertCommand:
    """Test the convert command functionality."""

    @pytest.fixture
    def temp_tasks_file(self, tmp_path):
        """Create a temporary tasks file."""
        tasks = [
            {
                "prompt": "Test task 1",
                "mcp_config": {
                    "local": {
                        "command": "docker",
                        "args": ["run", "--rm", "-i", "test-image:latest"],
                    }
                },
            }
        ]
        tasks_file = tmp_path / "tasks.json"
        tasks_file.write_text(json.dumps(tasks))
        return tasks_file

    @pytest.fixture
    def mock_env_dir(self, tmp_path):
        """Create a mock environment directory with lock file."""
        env_dir = tmp_path / "env"
        env_dir.mkdir()

        # Create lock file
        lock_data = {
            "images": {
                "remote": "registry.hud.ai/test-org/test-env:v1.0.0",
                "local": "test-env:latest",
            }
        }
        lock_file = env_dir / "hud.lock.yaml"
        import yaml

        lock_file.write_text(yaml.dump(lock_data))

        return env_dir

    @patch("hud.cli.flows.tasks._derive_remote_image")
    @patch("hud.cli.flows.tasks._ensure_pushed")
    @patch("hud.cli.flows.tasks.find_environment_dir")
    @patch("hud.cli.flows.tasks.load_tasks")
    @patch("hud.settings.settings")
    def test_convert_tasks_basic(
        self,
        mock_settings,
        mock_load_tasks,
        mock_find_env,
        mock_ensure_pushed,
        mock_derive_remote,
        temp_tasks_file,
        mock_env_dir,
    ):
        """Test basic task conversion from local to remote."""
        # Setup mocks
        mock_settings.api_key = "test-api-key"
        mock_find_env.return_value = mock_env_dir

        # Mock the push check to return updated lock data
        mock_ensure_pushed.return_value = {
            "images": {
                "remote": "registry.hud.ai/test-org/test-env:v1.0.0",
                "local": "test-env:v1.0.0",
            }
        }

        # Mock derive remote image
        mock_derive_remote.return_value = "registry.hud.ai/test-org/test-env:v1.0.0"

        task = Task(
            prompt="Test task",
            mcp_config={
                "local": {"command": "docker", "args": ["run", "--rm", "-i", "test-image:latest"]}
            },
        )
        raw_task = {
            "prompt": "Test task",
            "mcp_config": {
                "local": {"command": "docker", "args": ["run", "--rm", "-i", "test-image:latest"]}
            },
        }

        mock_load_tasks.side_effect = [[task], [raw_task]]

        # Run conversion
        result_path = convert_tasks_to_remote(str(temp_tasks_file))

        # Check result
        assert result_path.endswith("remote_tasks.json")
        assert Path(result_path).exists()

        # Verify converted content
        with open(result_path) as f:
            converted_tasks = json.load(f)

        assert len(converted_tasks) == 1
        assert "hud" in converted_tasks[0]["mcp_config"]
        assert converted_tasks[0]["mcp_config"]["hud"]["url"] == "https://mcp.hud.ai/v3/mcp"

    @patch("hud.settings.settings")
    def test_convert_missing_api_key(self, mock_settings, temp_tasks_file):
        """Test that conversion fails without API key."""
        mock_settings.api_key = ""

        with pytest.raises(typer.Exit):
            convert_tasks_to_remote(str(temp_tasks_file))

    @patch("hud.cli.flows.tasks.find_environment_dir")
    @patch("hud.cli.flows.tasks.load_tasks")
    @patch("hud.settings.settings")
    def test_convert_already_remote(
        self, mock_settings, mock_load_tasks, mock_find_env, temp_tasks_file
    ):
        """Test that already remote tasks are not converted again."""
        mock_settings.api_key = "test-api-key"
        mock_find_env.return_value = None  # No env dir needed for remote tasks

        # Create task that's already remote
        task = Task(
            prompt="Test task",
            mcp_config={
                "remote": {
                    "url": "https://mcp.hud.ai",
                    "headers": {"Mcp-Image": "registry.hud.ai/test/image:v1"},
                }
            },
        )

        mock_load_tasks.return_value = [task]

        # Should return original path without modification
        result_path = convert_tasks_to_remote(str(temp_tasks_file))
        assert result_path == str(temp_tasks_file)

    @patch("hud.cli.flows.tasks.find_environment_dir")
    @patch("hud.cli.flows.tasks.load_tasks")
    @patch("hud.settings.settings")
    def test_convert_no_environment(
        self, mock_settings, mock_load_tasks, mock_find_env, temp_tasks_file
    ):
        """Test that conversion fails when no environment is found."""
        mock_settings.api_key = "test-api-key"
        mock_find_env.return_value = None

        task = Task(
            prompt="Test task",
            mcp_config={
                "local": {"command": "docker", "args": ["run", "--rm", "-i", "test-image:latest"]}
            },
        )

        mock_load_tasks.return_value = [task]

        with pytest.raises(typer.Exit):
            convert_tasks_to_remote(str(temp_tasks_file))

    @patch("hud.utils.hud_console.hud_console.confirm")
    @patch("hud.cli.flows.tasks._derive_remote_image")
    @patch("hud.cli.flows.tasks._ensure_pushed")
    @patch("hud.cli.flows.tasks.find_environment_dir")
    @patch("hud.cli.flows.tasks.load_tasks")
    @patch("hud.settings.settings")
    def test_convert_with_env_vars(
        self,
        mock_settings,
        mock_load_tasks,
        mock_find_env,
        mock_ensure_pushed,
        mock_derive_remote,
        mock_confirm,
        temp_tasks_file,
        mock_env_dir,
    ):
        """Test conversion includes environment variables as headers."""
        mock_settings.api_key = "test-api-key"
        mock_find_env.return_value = mock_env_dir
        mock_confirm.return_value = True  # Always confirm in tests

        # Mock the push check to return updated lock data
        mock_ensure_pushed.return_value = {
            "images": {
                "remote": "registry.hud.ai/test-org/test-env:v1.0.0",
                "local": "test-env:v1.0.0",
            }
        }

        # Mock derive remote image
        mock_derive_remote.return_value = "registry.hud.ai/test-org/test-env:v1.0.0"

        # Add .env file with API keys
        env_file = mock_env_dir / ".env"
        env_file.write_text("OPENAI_API_KEY=sk-test123\nANTHROPIC_API_KEY=sk-ant456")

        task = Task(
            prompt="Test task",
            mcp_config={
                "local": {
                    "command": "docker",
                    "args": ["run", "--rm", "-i", "-e", "OPENAI_API_KEY", "test-image:latest"],
                }
            },
        )
        raw_task = task.model_dump()

        mock_load_tasks.side_effect = [[task], [raw_task]]

        # Run conversion
        result_path = convert_tasks_to_remote(str(temp_tasks_file))

        # Verify headers include env vars
        with open(result_path) as f:
            converted_tasks = json.load(f)

        headers = converted_tasks[0]["mcp_config"]["hud"]["headers"]
        assert "Env-Openai-Api-Key" in headers
        assert headers["Env-Openai-Api-Key"] == "${OPENAI_API_KEY}"


class TestConvertHelperFunctions:
    """Test helper functions used by convert command."""

    def test_env_var_to_header_key(self):
        """Test environment variable name conversion to header format."""
        from hud.cli.flows.tasks import _env_var_to_header_key

        assert _env_var_to_header_key("OPENAI_API_KEY") == "Env-Openai-Api-Key"
        assert _env_var_to_header_key("ANTHROPIC_API_KEY") == "Env-Anthropic-Api-Key"
        assert _env_var_to_header_key("SIMPLE") == "Env-Simple"
        assert _env_var_to_header_key("MULTIPLE_WORD_VAR") == "Env-Multiple-Word-Var"

    def test_extract_dotenv_api_key_vars(self):
        """Test extraction of API-like variables from .env file."""
        # Create test env directory with .env file
        import tempfile

        from hud.cli.flows.tasks import _extract_dotenv_api_key_vars

        with tempfile.TemporaryDirectory() as tmpdir:
            env_dir = Path(tmpdir)
            env_file = env_dir / ".env"
            env_file.write_text("""
# Test .env file
OPENAI_API_KEY=sk-test123
ANTHROPIC_API_KEY=sk-ant456
SOME_TOKEN=abc123
CLIENT_SECRET=secret789
USER_PASSWORD=pass123
REGULAR_VAR=not_included
HUD_API_URL=https://api.hud.ai
""")

            result = _extract_dotenv_api_key_vars(env_dir)

            # Should include only API-like variables
            assert "OPENAI_API_KEY" in result
            assert "ANTHROPIC_API_KEY" in result
            assert "SOME_TOKEN" in result
            assert "CLIENT_SECRET" in result
            assert "USER_PASSWORD" in result
            assert "REGULAR_VAR" not in result
            assert "HUD_API_URL" in result  # API in name, so it's included

    def test_is_remote_url(self):
        """Test remote URL detection."""
        from hud.cli.flows.tasks import _is_remote_url

        # This function matches URLs with domain names (not localhost or IPs)
        assert _is_remote_url("https://mcp.hud.ai")
        assert _is_remote_url("http://mcp.hud.ai")
        assert _is_remote_url("https://mcp.hud.ai/some/path")
        assert _is_remote_url("https://example.com")  # Also matches
        assert not _is_remote_url("http://localhost:8000")  # localhost doesn't match
        assert not _is_remote_url("file:///path/to/file")  # file:// doesn't match

    def test_extract_env_vars_from_docker_args(self):
        """Test extraction of environment variables from docker arguments."""
        from hud.cli.flows.tasks import _extract_env_vars_from_docker_args

        # Test with various docker arg formats
        args = [
            "run",
            "--rm",
            "-i",
            "-e",
            "VAR1",
            "-e",
            "VAR2=value",
            "--env",
            "VAR3",
            "--env=VAR4",
            # Note: -eFOO compact form is not supported by the implementation
            "--env-file",
            ".env",
            "-p",
            "8080:80",
        ]

        result = _extract_env_vars_from_docker_args(args)

        assert "VAR1" in result
        assert "VAR2" in result
        assert "VAR3" in result
        assert "VAR4" in result
        # FOO is not extracted because -eFOO compact form is not supported
        assert len(result) == 4

    def test_derive_remote_image(self):
        """Test deriving remote image from lock data."""
        from hud.cli.flows.tasks import _derive_remote_image

        # The function derives remote image from images.local, not images.remote
        lock_data = {"images": {"local": "test-env:v1.0.0"}}
        result = _derive_remote_image(lock_data)
        assert result == "test-env:v1.0.0"

        # Test fallback to legacy format
        lock_data = {
            "image": "test-org/test-env:v1.0.0",
        }
        result = _derive_remote_image(lock_data)
        assert result == "test-org/test-env:v1.0.0"

    def test_extract_vars_from_task_configs(self):
        """Test extraction of env vars from task configurations."""
        from hud.cli.flows.tasks import _extract_vars_from_task_configs

        raw_tasks = [
            {
                "prompt": "Task 1",
                "mcp_config": {
                    "local": {"command": "docker", "args": ["run", "-e", "API_KEY1", "image1"]}
                },
            },
            {
                "prompt": "Task 2",
                "mcp_config": {
                    "local": {
                        "command": "docker",
                        "args": ["run", "-e", "API_KEY2", "--env", "API_KEY3", "image2"],
                    }
                },
            },
            {"prompt": "Task 3", "mcp_config": {"remote": {"url": "https://mcp.hud.ai"}}},
        ]

        result = _extract_vars_from_task_configs(raw_tasks)

        assert "API_KEY1" in result
        assert "API_KEY2" in result
        assert "API_KEY3" in result
        assert len(result) == 3
