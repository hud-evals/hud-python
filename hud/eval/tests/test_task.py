"""Tests for hud.eval.task module."""

from __future__ import annotations

import pytest

from hud.eval.task import Task, TaskAgentConfig


class TestTaskSerialization:
    """Tests for Task serialization and roundtrip."""

    def test_task_roundtrip(self) -> None:
        """Task serializes and deserializes correctly."""
        task = Task(
            env={"name": "browser", "include": ["navigate", "click"]},
            scenario="checkout",
            id="task-1",
            args={"user_id": "alice"},
        )

        # Serialize
        data = task.model_dump(mode="json")

        # Should have the current task format
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


class TestTaskValidation:
    """Tests for Task validation."""

    def test_allows_none_env(self) -> None:
        """Task allows None env (for blank evals)."""
        task = Task(scenario="test")  # env=None is valid
        assert task.env is None
        assert task.scenario == "test"

    def test_rejects_legacy_task_fields(self) -> None:
        """Task rejects legacy task dictionaries."""
        with pytest.raises(ValueError, match="Legacy task fields are no longer supported"):
            Task.model_validate(
                {
                    "prompt": "test",
                    "mcp_config": {"server": {}},
                    "evaluate_tool": {"name": "check", "arguments": {}},
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

    def test_agent_config_rejects_legacy_fields(self) -> None:
        """agent_config rejects removed compatibility fields."""
        with pytest.raises(ValueError, match="append_setup_output"):
            Task(
                env={"name": "browser"},
                agent_config={"append_setup_output": True},
            )


class TestValidationAnnotation:
    """Tests that annotation is preserved through validation sequences (golden traces)."""

    def test_validation_preserves_annotation_from_mcp_tool_call(self) -> None:
        """Annotation set on MCPToolCall objects survives Task construction."""
        from hud.types import MCPToolCall

        task = Task(
            env={"name": "browser"},
            scenario="checkout",
            validation=[
                MCPToolCall(name="click", arguments={"x": 1}, annotation="Open the cart"),
                MCPToolCall(name="submit", arguments={}, annotation="Confirm purchase"),
            ],
        )

        assert task.validation is not None
        assert task.validation[0].annotation == "Open the cart"
        assert task.validation[1].annotation == "Confirm purchase"

    def test_validation_preserves_annotation_from_dict(self) -> None:
        """Annotation in raw dicts is preserved through convert_validation."""
        task = Task(
            env={"name": "browser"},
            scenario="checkout",
            validation=[  # type: ignore[arg-type]
                {"name": "click", "arguments": {"x": 1}, "annotation": "Open the cart"},
                {"name": "submit", "arguments": {}},
            ],
        )

        assert task.validation is not None
        assert task.validation[0].annotation == "Open the cart"
        assert task.validation[1].annotation is None

    def test_validation_annotation_roundtrip(self) -> None:
        """Annotation survives full Task serialize -> deserialize roundtrip."""
        from hud.types import MCPToolCall

        task = Task(
            env={"name": "browser"},
            scenario="checkout",
            validation=[
                MCPToolCall(name="click", arguments={"x": 1}, annotation="Step 1"),
            ],
        )

        data = task.model_dump(mode="json")
        restored = Task(**data)

        assert restored.validation is not None
        assert restored.validation[0].annotation == "Step 1"
        assert restored.validation[0].name == "click"
        assert restored.validation[0].arguments == {"x": 1}
