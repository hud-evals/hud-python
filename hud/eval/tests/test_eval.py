"""Tests for hud.eval.task module (Task class)."""

from __future__ import annotations

import pytest

from hud.eval.task import Task


class TestTaskDataclass:
    """Tests for Task as a Pydantic model."""

    def test_init_defaults(self) -> None:
        """Task initializes with sensible defaults."""
        task = Task()

        assert task.env is None
        assert task.scenario is None
        assert task.args is None  # None = template, {} = runnable with no args

    def test_init_with_env_dict(self) -> None:
        """Task auto-converts env dict to Environment via validator."""
        from hud.environment import Environment

        task = Task(
            env={"name": "browser", "include": ["navigate"]},
            scenario="checkout",
            args={"user_id": "alice"},
        )

        # env dict is auto-converted to Environment
        assert isinstance(task.env, Environment)
        assert task.scenario == "checkout"
        assert task.args == {"user_id": "alice"}

    def test_copy_creates_new_instance(self) -> None:
        """copy() creates a new Task instance."""
        original = Task(
            id="task-123",
            slug="demo-slug",
            env={"name": "test"},
            scenario="checkout",
            args={"user_id": "alice"},
        )
        copied = original.copy()

        assert copied is not original
        assert copied.env is original.env  # Env reference is shared (intentional)
        assert copied.id is None
        assert copied.slug is None
        assert copied.scenario == original.scenario
        assert copied.args == original.args
        assert copied.args is not original.args  # Args are deep copied

    def test_copy_with_deep_true_preserves_env_ref_and_deep_copies_args(self) -> None:
        """copy(deep=True) keeps env shared but deep-copies mutable task data."""
        original = Task(
            env={"name": "test"},
            scenario="checkout",
            args={"user": {"id": "alice"}},
        )

        copied = original.copy(deep=True)
        assert copied.env is original.env
        assert copied.args is not original.args
        assert copied.args == original.args

        assert copied.args is not None
        assert original.args is not None
        copied.args["user"]["id"] = "bob"
        assert original.args["user"]["id"] == "alice"

    def test_copy_with_update_validates_payload(self) -> None:
        """copy(update=...) re-validates updates through Task validators."""
        from pydantic import ValidationError

        original = Task(
            env={"name": "test"},
            scenario="checkout",
            args={"user_id": "alice"},
        )

        with pytest.raises(ValidationError):
            original.copy(update={"env": {"include": ["navigate"]}})


class TestEnvironmentCall:
    """Tests for Environment.__call__ returning Task."""

    def test_call_returns_task(self) -> None:
        """Environment() returns a Task object."""
        from hud.environment import Environment

        env = Environment("test-env")
        task = env()

        assert isinstance(task, Task)

    def test_call_with_scenario_sets_scenario(self) -> None:
        """Environment(scenario) sets scenario name."""
        from hud.environment import Environment

        env = Environment("test-env")
        task = env("checkout")

        assert task.scenario == "checkout"

    def test_call_with_args_sets_args(self) -> None:
        """Environment(scenario, **args) sets args."""
        from hud.environment import Environment

        env = Environment("test-env")
        task = env("checkout", user_id="alice", amount=100)

        assert task.args == {"user_id": "alice", "amount": 100}

    def test_call_returns_task_with_env(self) -> None:
        """Environment() returns Task with env reference."""
        from hud.environment import Environment

        env = Environment("test-env")
        task = env()

        # Task has reference to the Environment
        assert task.env is env
