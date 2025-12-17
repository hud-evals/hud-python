"""Tests for hud.datasets.utils module."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from hud.datasets.utils import (
    BatchRequest,
    SingleTaskRequest,
    cancel_all_jobs,
    cancel_job,
    cancel_task,
    submit_rollouts,
)
from hud.eval.display import display_results
from hud.types import AgentType, LegacyTask, Trace


class TestSingleTaskRequest:
    """Tests for SingleTaskRequest schema."""

    def test_valid_request(self):
        """Test creating a valid SingleTaskRequest with v5 task."""
        request = SingleTaskRequest(
            task={"env": {"name": "browser"}, "scenario": "checkout"},
            agent_type=AgentType.CLAUDE,
            agent_params={"checkpoint_name": "claude-sonnet-4-5"},
            max_steps=10,
            job_id="job-123",
            task_id="task-1",
            trace_name="Test trace",
        )
        assert request.task_id == "task-1"
        assert request.agent_type == AgentType.CLAUDE

    def test_empty_job_id_rejected(self):
        """Test that empty job_id is rejected."""
        with pytest.raises(ValueError, match="job_id must be a non-empty string"):
            SingleTaskRequest(
                task={"prompt": "test", "mcp_config": {}},
                agent_type=AgentType.CLAUDE,
                job_id="",
                task_id="task-1",
                trace_name="Test",
            )

    def test_invalid_task_rejected(self):
        """Test that invalid task payload is rejected (neither v4 nor v5)."""
        with pytest.raises(ValueError, match="Task must have 'env'"):
            SingleTaskRequest(
                task={"invalid_field": "test"},  # Missing required fields
                agent_type=AgentType.CLAUDE,
                job_id="job-123",
                task_id="task-1",
                trace_name="Test",
            )

    def test_incomplete_v4_task_rejected(self):
        """Test that incomplete v4 task (missing evaluate_tool) is rejected."""
        # When prompt + mcp_config is present but evaluate_tool is missing,
        # it's detected as v4 format but fails validation
        with pytest.raises(ValueError, match="v4 task missing required fields"):
            SingleTaskRequest(
                task={
                    "prompt": "test",
                    "mcp_config": {"server": {"url": "http://localhost"}},
                    # Missing evaluate_tool
                },
                agent_type=AgentType.CLAUDE,
                job_id="job-123",
                task_id="task-1",
                trace_name="Test",
            )

    def test_valid_v4_task_accepted(self):
        """Test that complete v4 task is accepted."""
        request = SingleTaskRequest(
            task={
                "prompt": "test",
                "mcp_config": {"server": {"url": "http://localhost"}},
                "evaluate_tool": {"name": "check", "arguments": {}},
            },
            agent_type=AgentType.CLAUDE,
            job_id="job-123",
            task_id="task-1",
            trace_name="Test",
        )
        assert request.task_id == "task-1"

    def test_valid_v5_task_accepted(self):
        """Test that v5 task with env is accepted."""
        request = SingleTaskRequest(
            task={"env": {"name": "browser"}, "scenario": "login"},
            agent_type=AgentType.CLAUDE,
            job_id="job-123",
            task_id="task-1",
            trace_name="Test",
        )
        assert request.task_id == "task-1"


class TestBatchRequest:
    """Tests for BatchRequest schema."""

    def test_valid_batch(self):
        """Test creating a valid batch request."""
        requests = [
            SingleTaskRequest(
                task={"env": {"name": "browser"}, "scenario": "test"},
                agent_type=AgentType.CLAUDE,
                job_id="job-123",
                task_id=f"task-{i}",
                trace_name=f"Trace {i}",
            )
            for i in range(3)
        ]
        batch = BatchRequest(requests=requests)
        assert len(batch.requests) == 3


class TestCancellationFunctions:
    """Tests for cancellation functions."""

    @pytest.mark.asyncio
    async def test_cancel_task(self):
        """Test cancel_task makes correct API call."""
        with patch("hud.datasets.utils.httpx.AsyncClient") as mock_client_cls:
            mock_response = MagicMock()
            mock_response.json.return_value = {"cancelled": True, "task_id": "task-1"}
            mock_response.raise_for_status = MagicMock()

            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client_cls.return_value = mock_client

            with patch("hud.datasets.utils.settings") as mock_settings:
                mock_settings.hud_api_url = "https://api.hud.ai"
                mock_settings.api_key = "test-key"

                result = await cancel_task("job-123", "task-1")

            assert result["cancelled"] is True
            mock_client.post.assert_called_once()
            call_args = mock_client.post.call_args
            assert "cancel" in call_args[0][0]
            assert call_args[1]["json"]["job_id"] == "job-123"
            assert call_args[1]["json"]["task_id"] == "task-1"

    @pytest.mark.asyncio
    async def test_cancel_job(self):
        """Test cancel_job makes correct API call."""
        with patch("hud.datasets.utils.httpx.AsyncClient") as mock_client_cls:
            mock_response = MagicMock()
            mock_response.json.return_value = {"cancelled": 5, "job_id": "job-123"}
            mock_response.raise_for_status = MagicMock()

            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client_cls.return_value = mock_client

            with patch("hud.datasets.utils.settings") as mock_settings:
                mock_settings.hud_api_url = "https://api.hud.ai"
                mock_settings.api_key = "test-key"

                result = await cancel_job("job-123")

            assert result["cancelled"] == 5
            mock_client.post.assert_called_once()

    @pytest.mark.asyncio
    async def test_cancel_all_jobs(self):
        """Test cancel_all_jobs makes correct API call."""
        with patch("hud.datasets.utils.httpx.AsyncClient") as mock_client_cls:
            mock_response = MagicMock()
            mock_response.json.return_value = {"jobs_cancelled": 3, "total_tasks_cancelled": 10}
            mock_response.raise_for_status = MagicMock()

            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client_cls.return_value = mock_client

            with patch("hud.datasets.utils.settings") as mock_settings:
                mock_settings.hud_api_url = "https://api.hud.ai"
                mock_settings.api_key = "test-key"

                result = await cancel_all_jobs()

            assert result["jobs_cancelled"] == 3
            assert result["total_tasks_cancelled"] == 10


class TestDisplayResults:
    """Tests for display_results function."""

    def test_display_with_traces(self):
        """Test displaying single-run trace results."""
        tasks = [
            LegacyTask(id="t1", prompt="Test task 1", mcp_config={}),
            LegacyTask(id="t2", prompt="Test task 2", mcp_config={}),
        ]
        results = [
            Trace(reward=0.9, done=True),
            Trace(reward=0.5, done=True),
        ]

        # Should not raise
        display_results(results, tasks=tasks)

    def test_display_with_group_stats(self):
        """Test displaying group statistics."""
        tasks = [
            LegacyTask(id="t1", prompt="Test task 1", mcp_config={}),
        ]
        results = [
            {
                "task_id": "t1",
                "prompt": "Test task 1",
                "mean_reward": 0.85,
                "std_reward": 0.1,
                "min_reward": 0.7,
                "max_reward": 1.0,
                "success_rate": 0.9,
                "group_size": 3,
                "rewards": [0.8, 0.85, 0.9],
            }
        ]

        # Should not raise
        display_results(results, tasks=tasks)

    def test_display_empty_results(self):
        """Test displaying when no valid results."""
        tasks = [LegacyTask(prompt="Test", mcp_config={})]
        results: list[Trace | None] = [None]

        # Should not raise
        display_results(results, tasks=tasks)


class TestSubmitRollouts:
    """Tests for submit_rollouts function."""

    @pytest.mark.asyncio
    async def test_submit_single_task(self):
        """Test submitting a single task (v5 format)."""
        from hud.eval.task import Task

        tasks = [Task(env={"name": "browser"}, scenario="test", id="task-1")]

        with patch("hud.datasets.utils.httpx.AsyncClient") as mock_client_cls:
            mock_response = MagicMock()
            mock_response.json.return_value = {"accepted": 1, "rejected": 0}
            mock_response.raise_for_status = MagicMock()

            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client_cls.return_value = mock_client

            with patch("hud.datasets.utils.settings") as mock_settings:
                mock_settings.hud_api_url = "https://api.hud.ai"
                mock_settings.api_key = "test-key"

                # submit_rollouts doesn't return a value
                await submit_rollouts(
                    tasks=tasks,
                    agent_type=AgentType.CLAUDE,
                    job_id="job-123",
                )

            mock_client.post.assert_called_once()

    @pytest.mark.asyncio
    async def test_submit_with_group_size(self):
        """Test submitting with group_size > 1 creates multiple requests per task."""
        from hud.eval.task import Task

        tasks = [Task(env={"name": "browser"}, scenario="test", id="task-1")]

        with patch("hud.datasets.utils.httpx.AsyncClient") as mock_client_cls:
            mock_response = MagicMock()
            mock_response.json.return_value = {"accepted": 3, "rejected": 0}
            mock_response.raise_for_status = MagicMock()

            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client_cls.return_value = mock_client

            with patch("hud.datasets.utils.settings") as mock_settings:
                mock_settings.hud_api_url = "https://api.hud.ai"
                mock_settings.api_key = "test-key"

                await submit_rollouts(
                    tasks=tasks,
                    agent_type=AgentType.CLAUDE,
                    job_id="job-123",
                    group_size=3,
                )

            # Verify batch request contains 3 requests (1 task x 3 group_size)
            call_args = mock_client.post.call_args
            assert call_args is not None
            batch_data = call_args.kwargs["json"]
            assert len(batch_data["requests"]) == 3
