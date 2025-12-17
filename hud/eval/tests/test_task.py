"""Tests for hud.eval.task module."""

from __future__ import annotations

import pytest

from hud.eval.task import Task, TaskAgentConfig


class TestTaskSerialization:
    """Tests for Task serialization and roundtrip."""

    def test_v5_task_roundtrip(self) -> None:
        """v5 Task serializes and deserializes correctly."""
        task = Task(
            env={"name": "browser", "include": ["navigate", "click"]},
            scenario="checkout",
            id="task-1",
            args={"user_id": "alice"},
        )

        # Serialize
        data = task.model_dump(mode="json")

        # Should have v5 format
        assert "env" in data
        assert data["env"]["name"] == "browser"
        assert data["scenario"] == "checkout"
        assert data["id"] == "task-1"

        # Recreate from serialized data
        task2 = Task(**data)

        # Serialize again
        data2 = task2.model_dump(mode="json")

        # Should be identical
        assert data == data2

    def test_v4_task_roundtrip(self) -> None:
        """v4 Task serializes (flattens) and deserializes correctly."""
        v4_dict = {
            "prompt": "Go to google.com and search for cats",
            "mcp_config": {
                "browser": {"url": "http://localhost:8080"},
            },
            "evaluate_tool": {"name": "check_url", "arguments": {"contains": "google"}},
            "setup_tool": {"name": "navigate", "arguments": {"url": "about:blank"}},
            "id": "v4-task-1",
            "agent_config": {"system_prompt": "You are a helpful assistant"},
            "metadata": {"category": "navigation"},
        }

        # Create Task from v4 dict
        task = Task.from_v4(v4_dict)

        # Serialize (should flatten to v4 format)
        data = task.model_dump(mode="json")

        # Should have v4 format (flat, not nested env)
        assert "prompt" in data
        assert "mcp_config" in data
        assert "evaluate_tool" in data
        assert data["prompt"] == "Go to google.com and search for cats"
        assert data["id"] == "v4-task-1"

        # Recreate from serialized data
        task2 = Task(**data)

        # Serialize again
        data2 = task2.model_dump(mode="json")

        # Should be identical
        assert data == data2

    def test_v4_preserves_agent_config(self) -> None:
        """v4 Task preserves agent_config through roundtrip."""
        v4_dict = {
            "prompt": "Test prompt",
            "mcp_config": {"server": {"url": "http://localhost"}},
            "evaluate_tool": {"name": "check", "arguments": {}},
            "agent_config": {"system_prompt": "Custom system prompt"},
        }

        task = Task.from_v4(v4_dict)
        data = task.model_dump(mode="json")

        assert data.get("agent_config") == {"system_prompt": "Custom system prompt"}

        # Roundtrip
        task2 = Task(**data)
        assert task2.agent_config is not None
        assert isinstance(task2.agent_config, TaskAgentConfig)
        assert task2.agent_config.system_prompt == "Custom system prompt"

    def test_v4_preserves_metadata(self) -> None:
        """v4 Task preserves metadata through roundtrip."""
        v4_dict = {
            "prompt": "Test prompt",
            "mcp_config": {"server": {"url": "http://localhost"}},
            "evaluate_tool": {"name": "check", "arguments": {}},
            "metadata": {"key1": "value1", "key2": 42},
        }

        task = Task.from_v4(v4_dict)
        data = task.model_dump(mode="json")

        assert data.get("metadata") == {"key1": "value1", "key2": 42}

        # Roundtrip
        task2 = Task(**data)
        assert task2.metadata == {"key1": "value1", "key2": 42}


class TestTaskValidation:
    """Tests for Task validation."""

    def test_v5_allows_none_env(self) -> None:
        """v5 Task allows None env (for blank evals)."""
        task = Task(scenario="test")  # env=None is valid
        assert task.env is None
        assert task.scenario == "test"

    def test_v4_requires_evaluate_tool(self) -> None:
        """v4 Task requires evaluate_tool for validation."""
        from hud.eval.utils import validate_v4_task

        with pytest.raises(ValueError, match="evaluate_tool"):
            validate_v4_task(
                {
                    "prompt": "test",
                    "mcp_config": {"server": {}},
                    # Missing evaluate_tool
                }
            )

    def test_agent_config_accepts_dict(self) -> None:
        """agent_config can be provided as dict and gets converted."""
        task = Task(
            env={"name": "browser"},
            agent_config={"system_prompt": "Hello"},
        )

        assert isinstance(task.agent_config, TaskAgentConfig)
        assert task.agent_config.system_prompt == "Hello"
