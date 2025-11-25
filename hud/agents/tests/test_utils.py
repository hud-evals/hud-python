"""Tests for OpenAI MCP Agent utility functions."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from hud.agents.utils import log_agent_metadata_to_status, log_task_config_to_current_trace
from hud.types import Task


class TestAgentUtils:
    """Test agent utility functions."""

    @pytest.mark.asyncio
    async def test_log_task_config_no_task_run_id(self):
        """Test log_task_config when there's no current task run id."""
        task = Task(prompt="Test task", mcp_config={"server": "test"})

        # Should not raise, just return silently
        with patch("hud.agents.utils.get_current_task_run_id", return_value=None):
            await log_task_config_to_current_trace(task)

    @pytest.mark.asyncio
    async def test_log_task_config_with_task_run_id(self):
        """Test log_task_config when there's a current task run id."""
        task = Task(prompt="Test task", mcp_config={"server": "test"})

        with (
            patch("hud.agents.utils.get_current_task_run_id", return_value="task-123"),
            patch(
                "hud.agents.utils._update_task_status_async", new_callable=AsyncMock
            ) as mock_update,
        ):
            await log_task_config_to_current_trace(task)

            mock_update.assert_called_once()
            call_kwargs = mock_update.call_args
            assert call_kwargs[0][0] == "task-123"
            assert call_kwargs[0][1] == "running"
            assert "task_config" in call_kwargs[1]["extra_metadata"]

    @pytest.mark.asyncio
    async def test_log_task_config_exception_suppressed(self):
        """Test that exceptions are suppressed in log_task_config."""
        task = Task(prompt="Test task", mcp_config={"server": "test"})

        with (
            patch("hud.agents.utils.get_current_task_run_id", return_value="task-123"),
            patch(
                "hud.agents.utils._update_task_status_async",
                side_effect=Exception("API error"),
            ),
        ):
            # Should not raise
            await log_task_config_to_current_trace(task)

    @pytest.mark.asyncio
    async def test_log_agent_metadata_no_task_run_id(self):
        """Test log_agent_metadata when there's no current task run id."""
        with patch("hud.agents.utils.get_current_task_run_id", return_value=None):
            await log_agent_metadata_to_status(model_name="test-model")

    @pytest.mark.asyncio
    async def test_log_agent_metadata_no_metadata(self):
        """Test log_agent_metadata when no metadata is provided."""
        with patch("hud.agents.utils.get_current_task_run_id", return_value="task-123"):
            # Should return early without calling update
            await log_agent_metadata_to_status()

    @pytest.mark.asyncio
    async def test_log_agent_metadata_with_model_name(self):
        """Test log_agent_metadata with model_name."""
        with (
            patch("hud.agents.utils.get_current_task_run_id", return_value="task-123"),
            patch(
                "hud.agents.utils._update_task_status_async", new_callable=AsyncMock
            ) as mock_update,
        ):
            await log_agent_metadata_to_status(model_name="gpt-4")

            mock_update.assert_called_once()
            call_kwargs = mock_update.call_args
            assert call_kwargs[1]["extra_metadata"]["agent"]["model_name"] == "gpt-4"
            assert "checkpoint_name" not in call_kwargs[1]["extra_metadata"]["agent"]

    @pytest.mark.asyncio
    async def test_log_agent_metadata_with_checkpoint_name(self):
        """Test log_agent_metadata with checkpoint_name."""
        with (
            patch("hud.agents.utils.get_current_task_run_id", return_value="task-123"),
            patch(
                "hud.agents.utils._update_task_status_async", new_callable=AsyncMock
            ) as mock_update,
        ):
            await log_agent_metadata_to_status(checkpoint_name="v1.0")

            mock_update.assert_called_once()
            call_kwargs = mock_update.call_args
            assert call_kwargs[1]["extra_metadata"]["agent"]["checkpoint_name"] == "v1.0"
            assert "model_name" not in call_kwargs[1]["extra_metadata"]["agent"]

    @pytest.mark.asyncio
    async def test_log_agent_metadata_with_both(self):
        """Test log_agent_metadata with both model_name and checkpoint_name."""
        with (
            patch("hud.agents.utils.get_current_task_run_id", return_value="task-123"),
            patch(
                "hud.agents.utils._update_task_status_async", new_callable=AsyncMock
            ) as mock_update,
        ):
            await log_agent_metadata_to_status(model_name="gpt-4", checkpoint_name="v1.0")

            mock_update.assert_called_once()
            call_kwargs = mock_update.call_args
            agent_meta = call_kwargs[1]["extra_metadata"]["agent"]
            assert agent_meta["model_name"] == "gpt-4"
            assert agent_meta["checkpoint_name"] == "v1.0"

    @pytest.mark.asyncio
    async def test_log_agent_metadata_exception_suppressed(self):
        """Test that exceptions are suppressed in log_agent_metadata."""
        with (
            patch("hud.agents.utils.get_current_task_run_id", return_value="task-123"),
            patch(
                "hud.agents.utils._update_task_status_async",
                side_effect=Exception("API error"),
            ),
        ):
            # Should not raise
            await log_agent_metadata_to_status(model_name="test-model")
