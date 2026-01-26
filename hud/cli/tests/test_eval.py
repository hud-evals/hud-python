"""Tests for hud.cli.eval module and run_dataset function."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from mcp import types

from hud.environment.router import ToolRouter
from hud.eval.context import EvalContext
from hud.types import AgentType, MCPToolResult, Trace


class MockEvalContext(EvalContext):
    """Mock EvalContext for testing."""

    def __init__(
        self,
        prompt: str = "Test prompt",
        tools: list[types.Tool] | None = None,
    ) -> None:
        # Core attributes
        self.prompt = prompt
        self._tools = tools or []
        self._submitted: str | None = None
        self.reward: float | None = None
        self.results: list[EvalContext] = []

        # Environment attributes
        self._router = ToolRouter()
        self._agent_include: list[str] | None = None
        self._agent_exclude: list[str] | None = None

        # EvalContext attributes
        self._task = None
        self.trace_id = "test-trace-id"
        self.eval_name = "test-eval"
        self.job_id: str | None = None
        self.group_id: str | None = None
        self.index = 0
        self.variants: dict[str, Any] = {}
        self.answer: str | None = None
        self.system_prompt: str | None = None
        self.error: BaseException | None = None
        self.metadata: dict[str, Any] = {}
        self._is_summary = False

    def as_tools(self) -> list[types.Tool]:
        return self._tools

    @property
    def has_scenario(self) -> bool:
        return False

    async def list_tools(self) -> list[types.Tool]:
        return self._tools

    async def call_tool(self, call: Any, /, **kwargs: Any) -> MCPToolResult:
        return MCPToolResult(
            content=[types.TextContent(type="text", text="ok")],
            isError=False,
        )

    async def submit(self, answer: str) -> None:
        self._submitted = answer


def _create_mock_agent_cls() -> tuple[MagicMock, MagicMock]:
    """Create a mock agent class and instance for testing."""
    mock_agent_instance = MagicMock()
    mock_agent_instance.run = AsyncMock(return_value=Trace(reward=1.0, done=True))
    mock_agent_cls = MagicMock()
    mock_agent_cls.create.return_value = mock_agent_instance
    return mock_agent_cls, mock_agent_instance


class TestRunDataset:
    """Test the new run_dataset function."""

    @pytest.mark.asyncio
    async def test_run_dataset_with_task_list(self) -> None:
        """Test run_dataset with a list of tasks."""
        from hud.eval.task import Task

        tasks = [
            Task(env={"name": "test"}, id="task1", scenario="test"),
            Task(env={"name": "test"}, id="task2", scenario="test"),
        ]
        mock_agent_cls, mock_agent_instance = _create_mock_agent_cls()

        # Mock hud.eval to return our mock context
        mock_ctx = MockEvalContext()

        with (
            patch("hud.datasets.runner.hud.eval") as mock_eval,
            patch("hud.agents.claude.ClaudeAgent", mock_agent_cls),
        ):
            # Set up the async context manager
            mock_eval.return_value.__aenter__ = AsyncMock(return_value=mock_ctx)
            mock_eval.return_value.__aexit__ = AsyncMock(return_value=None)

            from hud.datasets.runner import run_dataset

            await run_dataset(tasks, agent_type="claude", max_steps=5)

            # Verify hud.eval was called with correct params
            mock_eval.assert_called_once()
            call_kwargs = mock_eval.call_args[1]
            assert call_kwargs["group"] == 1
            assert call_kwargs["max_concurrent"] == 30

            # Agent should have run
            mock_agent_instance.run.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_dataset_with_string_source(self) -> None:
        """Test run_dataset with a string source (loads via load_dataset)."""
        from hud.eval.task import Task

        mock_tasks = [Task(env={"name": "test"}, id="loaded_task", scenario="loaded")]
        mock_agent_cls, _ = _create_mock_agent_cls()
        mock_ctx = MockEvalContext()

        with (
            patch("hud.datasets.loader.load_tasks", return_value=mock_tasks) as mock_load,
            patch("hud.datasets.runner.hud.eval") as mock_eval,
            patch("hud.agents.OpenAIAgent", mock_agent_cls),
        ):
            mock_eval.return_value.__aenter__ = AsyncMock(return_value=mock_ctx)
            mock_eval.return_value.__aexit__ = AsyncMock(return_value=None)

            from hud.datasets.runner import run_dataset

            await run_dataset("my-tasks.json", agent_type="openai")

            # Verify load_dataset was called
            mock_load.assert_called_once_with("my-tasks.json")

    @pytest.mark.asyncio
    async def test_run_dataset_empty_tasks_raises(self) -> None:
        """Test run_dataset raises ValueError for empty tasks."""
        with patch("hud.datasets.loader.load_dataset", return_value=[]):
            from hud.datasets.runner import run_dataset

            with pytest.raises(ValueError, match="No tasks to run"):
                await run_dataset([], agent_type=AgentType.CLAUDE)

    @pytest.mark.asyncio
    async def test_run_dataset_with_group_size(self) -> None:
        """Test run_dataset passes group_size to hud.eval."""
        from hud.eval.task import Task

        tasks = [Task(env={"name": "test"}, id="task1", scenario="test")]
        mock_agent_cls, _ = _create_mock_agent_cls()
        mock_ctx = MockEvalContext()

        with (
            patch("hud.datasets.runner.hud.eval") as mock_eval,
            patch("hud.agents.claude.ClaudeAgent", mock_agent_cls),
        ):
            mock_eval.return_value.__aenter__ = AsyncMock(return_value=mock_ctx)
            mock_eval.return_value.__aexit__ = AsyncMock(return_value=None)

            from hud.datasets.runner import run_dataset

            await run_dataset(tasks, agent_type="claude", group_size=3)

            call_kwargs = mock_eval.call_args[1]
            assert call_kwargs["group"] == 3

    @pytest.mark.asyncio
    async def test_run_dataset_with_max_concurrent(self) -> None:
        """Test run_dataset passes max_concurrent to hud.eval."""
        from hud.eval.task import Task

        tasks = [Task(env={"name": "test"}, id="task1", scenario="test")]
        mock_agent_cls, _ = _create_mock_agent_cls()
        mock_ctx = MockEvalContext()

        with (
            patch("hud.datasets.runner.hud.eval") as mock_eval,
            patch("hud.agents.claude.ClaudeAgent", mock_agent_cls),
        ):
            mock_eval.return_value.__aenter__ = AsyncMock(return_value=mock_ctx)
            mock_eval.return_value.__aexit__ = AsyncMock(return_value=None)

            from hud.datasets.runner import run_dataset

            await run_dataset(tasks, agent_type="claude", max_concurrent=10)

            call_kwargs = mock_eval.call_args[1]
            assert call_kwargs["max_concurrent"] == 10

    @pytest.mark.asyncio
    async def test_run_dataset_returns_results(self) -> None:
        """Test run_dataset returns EvalContext results."""
        from hud.eval.task import Task

        tasks = [Task(env={"name": "test"}, id="task1", scenario="test")]
        mock_agent_cls, _ = _create_mock_agent_cls()
        mock_ctx = MockEvalContext()

        with (
            patch("hud.datasets.runner.hud.eval") as mock_eval,
            patch("hud.agents.claude.ClaudeAgent", mock_agent_cls),
        ):
            mock_eval.return_value.__aenter__ = AsyncMock(return_value=mock_ctx)
            mock_eval.return_value.__aexit__ = AsyncMock(return_value=None)

            from hud.datasets.runner import run_dataset

            results = await run_dataset(tasks, agent_type="claude")

            # Should return list with the context
            assert len(results) == 1
            assert results[0] is mock_ctx

    @pytest.mark.asyncio
    async def test_run_dataset_parallel_results(self) -> None:
        """Test run_dataset returns ctx.results for parallel execution."""
        from hud.eval.task import Task

        tasks = [Task(env={"name": "test"}, id="task1", scenario="test")]
        mock_agent_cls, _ = _create_mock_agent_cls()

        # Create mock context with results (parallel execution)
        mock_result1 = MockEvalContext(prompt="result1")
        mock_result1.reward = 0.8
        mock_result2 = MockEvalContext(prompt="result2")
        mock_result2.reward = 0.9

        mock_ctx = MockEvalContext()
        mock_ctx.results = [mock_result1, mock_result2]

        with (
            patch("hud.datasets.runner.hud.eval") as mock_eval,
            patch("hud.agents.claude.ClaudeAgent", mock_agent_cls),
        ):
            mock_eval.return_value.__aenter__ = AsyncMock(return_value=mock_ctx)
            mock_eval.return_value.__aexit__ = AsyncMock(return_value=None)

            from hud.datasets.runner import run_dataset

            results = await run_dataset(tasks, agent_type="claude")

            # Should return the parallel results
            assert len(results) == 2
            assert results[0].reward == 0.8
            assert results[1].reward == 0.9
