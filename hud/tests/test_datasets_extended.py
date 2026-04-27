"""Extended tests for dataset utilities to improve coverage."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from hud.datasets import run_dataset


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

        # Create mock agent class and instance (use MagicMock since create() is sync)
        mock_agent_instance = AsyncMock()
        mock_agent_instance.run.return_value = Trace(reward=1.0, done=True)
        mock_agent_cls = MagicMock()
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

        # Create mock agent class and instance (use MagicMock since create() is sync)
        mock_agent_instance = AsyncMock()
        mock_agent_instance.run.return_value = Trace(reward=1.0, done=True)
        mock_agent_cls = MagicMock()
        mock_agent_cls.create.return_value = mock_agent_instance

        with (
            patch("hud.datasets.loader.load_tasks", return_value=mock_tasks) as mock_load,
            patch("hud.datasets.runner.hud.eval") as mock_eval,
            patch("hud.agents.OpenAIAgent", mock_agent_cls),
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

        # Create mock agent class and instance (use MagicMock since create() is sync)
        mock_agent_instance = AsyncMock()
        mock_agent_instance.run.return_value = Trace(reward=1.0, done=True)
        mock_agent_cls = MagicMock()
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
                quiet=True,
                job_id=None,
                taskset_id=None,
            )
