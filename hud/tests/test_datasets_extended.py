"""Extended tests for dataset utilities to improve coverage."""

from __future__ import annotations

from typing import cast
from unittest.mock import AsyncMock, patch

import pytest

from hud.datasets import run_dataset
from hud.types import LegacyTask, MCPToolCall


class TestTaskExtended:
    """Extended tests for LegacyTask functionality."""

    def test_taskconfig_with_all_fields(self):
        """Test LegacyTask with all possible fields."""
        setup_tool = MCPToolCall(name="setup", arguments={"board_size": 4})
        evaluate_tool = MCPToolCall(name="evaluate", arguments={"metric": "score"})

        task = LegacyTask(
            id="test-123",
            prompt="Play the game",
            mcp_config={
                "server": {"url": "http://localhost:8080"},
                "auth": {"token": "test-token"},
            },
            setup_tool=setup_tool,
            evaluate_tool=evaluate_tool,
            metadata={"experiment": "test1", "version": 2},
        )

        assert task.id == "test-123"
        assert task.prompt == "Play the game"
        assert task.setup_tool == setup_tool
        assert task.evaluate_tool == evaluate_tool
        assert task.metadata["experiment"] == "test1"
        assert task.metadata["version"] == 2

    def test_taskconfig_list_tools(self):
        """Test LegacyTask with list of tools."""
        setup_tools = [
            MCPToolCall(name="init", arguments={}),
            MCPToolCall(name="configure", arguments={"mode": "test"}),
        ]

        task = LegacyTask(
            prompt="Multi-setup task", mcp_config={"test": True}, setup_tool=setup_tools
        )

        assert isinstance(task.setup_tool, list)
        assert len(task.setup_tool) == 2
        # Type narrowing for pyright - we know it's a list with 2 items
        # Cast to list to satisfy type checker
        setup_tools = cast("list[MCPToolCall]", task.setup_tool)
        assert setup_tools[0].name == "init"
        assert setup_tools[1].arguments is not None
        assert setup_tools[1].arguments["mode"] == "test"

    def test_env_var_complex_resolution(self, monkeypatch):
        """Test complex environment variable scenarios."""
        # Set environment variables
        monkeypatch.setenv("HUD_API_KEY", "sk-12345")
        monkeypatch.setenv("HUD_TELEMETRY_URL", "https://api.example.com")
        monkeypatch.setenv("EMPTY_VAR", "")
        monkeypatch.setenv("RUN_ID", "run-789")

        # Mock settings in the shared env utility where resolve_env_vars is implemented
        with patch("hud.utils.env.settings") as mock_settings:
            mock_settings.api_key = "sk-12345"
            mock_settings.hud_telemetry_url = "https://api.example.com"
            mock_settings.model_dump.return_value = {
                "api_key": "sk-12345",
                "hud_telemetry_url": "https://api.example.com",
            }

            task = LegacyTask(
                prompt="Complex env test",
                mcp_config={
                    "auth": {
                        "bearer": "Bearer ${HUD_API_KEY}",
                        "empty": "${EMPTY_VAR}",
                        "missing": "${MISSING_VAR}",
                    },
                    "endpoints": [
                        "${HUD_TELEMETRY_URL}/v1",
                        "${HUD_TELEMETRY_URL}/v2",
                        "${MISSING_URL}",
                    ],
                    "metadata": {"run_id": "${RUN_ID}", "combined": "${HUD_API_KEY}-${RUN_ID}"},
                },
            )

        assert task.mcp_config["auth"]["bearer"] == "Bearer sk-12345"
        assert task.mcp_config["auth"]["empty"] == ""
        assert task.mcp_config["auth"]["missing"] == ""
        assert task.mcp_config["endpoints"][0] == "https://api.example.com/v1"
        assert task.mcp_config["endpoints"][1] == "https://api.example.com/v2"
        assert task.mcp_config["endpoints"][2] == ""
        assert task.mcp_config["metadata"]["combined"] == "sk-12345-run-789"

    def test_non_string_values_preserved(self):
        """Test that non-string values are preserved during env resolution."""
        task = LegacyTask(
            prompt="Test non-strings",
            mcp_config={
                "string": "${MISSING}",
                "number": 42,
                "boolean": True,
                "null": None,
                "nested": {"list": [1, 2, "${VAR}", 4], "dict": {"key": "${KEY}", "num": 123}},
            },
        )

        assert task.mcp_config["string"] == ""
        assert task.mcp_config["number"] == 42
        assert task.mcp_config["boolean"] is True
        assert task.mcp_config["null"] is None
        assert task.mcp_config["nested"]["list"] == [1, 2, "", 4]
        assert task.mcp_config["nested"]["dict"]["num"] == 123


class TestRunDatasetExtended:
    """Extended tests for run_dataset functionality."""

    @pytest.mark.asyncio
    async def test_run_dataset_empty(self):
        """Test running empty dataset raises ValueError."""
        from hud.types import AgentType

        # Empty task list should raise ValueError
        with pytest.raises(ValueError, match="No tasks to run"):
            await run_dataset([], agent_type=AgentType.CLAUDE)

    @pytest.mark.asyncio
    async def test_run_dataset_with_task_list(self):
        """Test run_dataset with Task objects."""
        from hud.eval.task import Task
        from hud.types import Trace

        # Create mock tasks with env as dict (to avoid real connections)
        mock_env = {"name": "test"}

        tasks = [
            Task(env=mock_env, scenario="test1"),
            Task(env=mock_env, scenario="test2"),
        ]

        # Mock hud.eval to avoid real eval context
        mock_ctx = AsyncMock()
        mock_ctx.results = None
        mock_ctx.reward = None

        # Create mock agent class and instance
        mock_agent_instance = AsyncMock()
        mock_agent_instance.run.return_value = Trace(reward=1.0, done=True)
        mock_agent_cls = AsyncMock()
        mock_agent_cls.create.return_value = mock_agent_instance

        with (
            patch("hud.datasets.runner.hud.eval") as mock_eval,
            patch("hud.agents.claude.ClaudeAgent", mock_agent_cls),
        ):
            mock_eval.return_value.__aenter__ = AsyncMock(return_value=mock_ctx)
            mock_eval.return_value.__aexit__ = AsyncMock(return_value=None)

            results = await run_dataset(tasks, agent_type="claude", max_steps=5)

            # Should return list with ctx
            assert len(results) == 1
            mock_agent_instance.run.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_dataset_from_source_string(self):
        """Test run_dataset with source string calls load_tasks."""
        from hud.eval.task import Task
        from hud.types import Trace

        mock_env = {"name": "test"}
        mock_tasks = [Task(env=mock_env, scenario="loaded")]  # type: ignore[arg-type]

        mock_ctx = AsyncMock()
        mock_ctx.results = None

        # Create mock agent class and instance
        mock_agent_instance = AsyncMock()
        mock_agent_instance.run.return_value = Trace(reward=1.0, done=True)
        mock_agent_cls = AsyncMock()
        mock_agent_cls.create.return_value = mock_agent_instance

        with (
            patch("hud.datasets.loader.load_tasks", return_value=mock_tasks) as mock_load,
            patch("hud.datasets.runner.hud.eval") as mock_eval,
            patch("hud.agents.openai.OpenAIAgent", mock_agent_cls),
        ):
            mock_eval.return_value.__aenter__ = AsyncMock(return_value=mock_ctx)
            mock_eval.return_value.__aexit__ = AsyncMock(return_value=None)

            await run_dataset("test-org/dataset", agent_type="openai")

            # Should call load_dataset with the source string
            mock_load.assert_called_once_with("test-org/dataset")

    @pytest.mark.asyncio
    async def test_run_dataset_passes_parameters(self):
        """Test that run_dataset passes parameters correctly to hud.eval."""
        from hud.eval.task import Task
        from hud.types import AgentType, Trace

        mock_env = {"name": "test"}
        tasks = [Task(env=mock_env, scenario="test")]

        mock_ctx = AsyncMock()
        mock_ctx.results = None

        # Create mock agent class and instance
        mock_agent_instance = AsyncMock()
        mock_agent_instance.run.return_value = Trace(reward=1.0, done=True)
        mock_agent_cls = AsyncMock()
        mock_agent_cls.create.return_value = mock_agent_instance

        with (
            patch("hud.datasets.runner.hud.eval") as mock_eval,
            patch("hud.agents.claude.ClaudeAgent", mock_agent_cls),
        ):
            mock_eval.return_value.__aenter__ = AsyncMock(return_value=mock_ctx)
            mock_eval.return_value.__aexit__ = AsyncMock(return_value=None)

            await run_dataset(
                tasks, agent_type=AgentType.CLAUDE, max_steps=25, max_concurrent=10, group_size=3
            )

            # Verify hud.eval was called with correct params
            mock_eval.assert_called_once_with(
                tasks,
                group=3,
                max_concurrent=10,
            )
