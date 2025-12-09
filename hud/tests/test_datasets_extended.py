"""Extended tests for dataset utilities to improve coverage."""

from __future__ import annotations

from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from hud.datasets import (
    Task,
    run_dataset,
)
from hud.types import MCPToolCall
from hud.utils.tasks import save_tasks


class TestTaskExtended:
    """Extended tests for Task functionality."""

    def test_taskconfig_with_all_fields(self):
        """Test Task with all possible fields."""
        setup_tool = MCPToolCall(name="setup", arguments={"board_size": 4})
        evaluate_tool = MCPToolCall(name="evaluate", arguments={"metric": "score"})

        task = Task(
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
        """Test Task with list of tools."""
        setup_tools = [
            MCPToolCall(name="init", arguments={}),
            MCPToolCall(name="configure", arguments={"mode": "test"}),
        ]

        task = Task(prompt="Multi-setup task", mcp_config={"test": True}, setup_tool=setup_tools)

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

        # Mock settings to return our test values
        with patch("hud.types.settings") as mock_settings:
            mock_settings.api_key = "sk-12345"
            mock_settings.hud_telemetry_url = "https://api.example.com"
            mock_settings.model_dump.return_value = {
                "api_key": "sk-12345",
                "hud_telemetry_url": "https://api.example.com",
            }

            task = Task(
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
        task = Task(
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


class TestDatasetOperations:
    """Test dataset conversion and operations."""

    def test_save_taskconfigs_empty_list(self):
        """Test saving empty task list."""
        with patch("datasets.Dataset") as MockDataset:
            mock_instance = MagicMock()
            MockDataset.from_list.return_value = mock_instance
            mock_instance.push_to_hub.return_value = None

            save_tasks([], "test-org/empty-dataset")

            MockDataset.from_list.assert_called_once_with([])
            mock_instance.push_to_hub.assert_called_once_with("test-org/empty-dataset")

    def test_save_taskconfigs_mixed_rejection(self):
        """Test that mixing dicts and Task objects is rejected."""
        valid_dict = {"prompt": "Dict task", "mcp_config": {"test": True}}

        task_object = Task(prompt="Object task", mcp_config={"resolved": "${SOME_VAR}"})

        # First item is dict, second is object
        with pytest.raises(ValueError, match="Item 1 is a Task object"):
            save_tasks([valid_dict, task_object], "test-org/mixed")  # type: ignore


class TestRunDatasetExtended:
    """Extended tests for run_dataset functionality."""

    @pytest.mark.asyncio
    async def test_run_dataset_empty(self):
        """Test running empty dataset."""
        with (
            patch("hud.clients.MCPClient"),
            patch("hud.eval.display.print_link"),
            patch("hud.eval.display.print_complete"),
        ):
            # Create a mock agent class with proper type
            from hud.agents import MCPAgent

            mock_agent_class = type("MockAgent", (MCPAgent,), {})

            results = await run_dataset(
                "empty_run",
                [],  # Empty task list
                mock_agent_class,
            )

            assert results == []

    @pytest.mark.asyncio
    async def test_run_dataset_with_metadata(self):
        """Test run_dataset with custom metadata."""
        from hud.agents import MCPAgent
        from hud.types import Trace

        # Create a proper mock agent class
        mock_agent_instance = AsyncMock()
        mock_agent_instance.run.return_value = Trace(reward=1.0, done=True)

        mock_agent_class = type(
            "MockAgent",
            (MCPAgent,),
            {
                "__init__": lambda self, **kwargs: None,
                "__new__": lambda cls, **kwargs: mock_agent_instance,
            },
        )

        tasks = [{"prompt": "Task 1", "mcp_config": {"url": "test1"}}]

        # Mock EvalContext to avoid actual MCP connections
        mock_ctx = AsyncMock()
        mock_ctx.__aenter__ = AsyncMock(return_value=mock_ctx)
        mock_ctx.__aexit__ = AsyncMock(return_value=None)

        with (
            patch("hud.clients.MCPClient"),
            patch("hud.eval.context.EvalContext.from_task", return_value=mock_ctx),
            patch("hud.eval.display.print_link"),
            patch("hud.eval.display.print_complete"),
        ):
            # Should run without error
            await run_dataset(
                "metadata_run",
                tasks,
                mock_agent_class,  # type: ignore
                {"verbose": True},
            )

    @pytest.mark.asyncio
    async def test_run_dataset_exception_handling(self):
        """Test exception handling during task execution."""
        from hud.types import Trace

        # Track execution by task index
        executed_task_indices: set[int] = set()

        # Create a mock agent class where behavior depends on the task being run
        def create_mock_agent(**kwargs):
            agent = AsyncMock()

            async def mock_run(task, **run_kwargs):
                # Extract task index from prompt "Task {i}"
                task_idx = int(task.prompt.split()[-1])
                executed_task_indices.add(task_idx)

                if task_idx == 1:  # Second task (index 1) should fail
                    raise RuntimeError("Task 2 failed")
                return Trace(reward=1.0, done=True, content=f"success-{task_idx + 1}")

            agent.run = mock_run
            return agent

        # Mock the agent class itself - runner calls agent_class.create()
        mock_agent_class = MagicMock()
        mock_agent_class.create = MagicMock(side_effect=create_mock_agent)
        mock_agent_class.__name__ = "MockAgent"

        tasks = [{"prompt": f"Task {i}", "mcp_config": {"url": f"test{i}"}} for i in range(3)]

        # Create mock contexts for each task
        def create_mock_ctx(*args, **kwargs):
            ctx = AsyncMock()
            ctx.__aenter__ = AsyncMock(return_value=ctx)
            ctx.__aexit__ = AsyncMock(return_value=None)
            ctx._suppress_link = False
            return ctx

        with (
            patch("hud.clients.MCPClient"),
            patch("hud.eval.context.EvalContext.from_task", side_effect=create_mock_ctx),
            patch("hud.eval.display.print_link"),
            patch("hud.eval.display.print_complete"),
        ):
            # Should complete without raising
            results = await run_dataset("error_run", tasks, mock_agent_class)  # type: ignore

            # All tasks should be attempted
            assert len(executed_task_indices) == 3
            assert executed_task_indices == {0, 1, 2}

            # Second result should be None due to exception
            assert results[1] is None

    @pytest.mark.asyncio
    async def test_run_dataset_client_cleanup(self):
        """Test that run_dataset completes successfully."""
        from hud.agents import MCPAgent
        from hud.types import Trace

        mock_agent_instance = AsyncMock()
        mock_agent_instance.run.return_value = Trace(reward=1.0, done=True)

        mock_agent_class = type(
            "MockAgent",
            (MCPAgent,),
            {
                "__init__": lambda self, **kwargs: None,
                "__new__": lambda cls, **kwargs: mock_agent_instance,
            },
        )

        tasks = [{"prompt": f"Task {i}", "mcp_config": {"url": f"test{i}"}} for i in range(3)]

        # Create mock contexts
        def create_mock_ctx(*args, **kwargs):
            ctx = AsyncMock()
            ctx.__aenter__ = AsyncMock(return_value=ctx)
            ctx.__aexit__ = AsyncMock(return_value=None)
            ctx._suppress_link = False
            return ctx

        with (
            patch("hud.clients.MCPClient"),
            patch("hud.eval.context.EvalContext.from_task", side_effect=create_mock_ctx),
            patch("hud.eval.display.print_link"),
            patch("hud.eval.display.print_complete"),
        ):
            results = await run_dataset("cleanup_run", tasks, mock_agent_class)  # type: ignore

            # Verify results were returned
            assert len(results) == 3

    @pytest.mark.asyncio
    async def test_run_dataset_validation_error(self):
        """Test that tasks without required fields cause validation errors."""
        from pydantic import ValidationError

        from hud.agents import MCPAgent

        # Create a task without mcp_config (required field)
        task: dict[str, Any] = {
            "prompt": "Test task",
            # No mcp_config - should cause validation error during Task(**task_dict)
        }

        mock_agent_class = type("MockAgent", (MCPAgent,), {})

        # Validation errors should be raised immediately when Task objects are created
        with pytest.raises(ValidationError):
            await run_dataset(
                "validation_run",
                [task],  # Pass the task directly
                mock_agent_class,  # type: ignore
            )
