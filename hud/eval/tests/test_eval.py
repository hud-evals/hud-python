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
        assert task.args == {}

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
            env={"name": "test"},
            scenario="checkout",
            args={"user_id": "alice"},
        )
        copied = original.copy()

        assert copied is not original
        assert copied.env is original.env  # Env reference is shared (intentional)
        assert copied.scenario == original.scenario
        assert copied.args == original.args
        assert copied.args is not original.args  # Args are deep copied


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

        # With setup_tool (v4 legacy)
        env2 = Environment("test-env").setup_tool("navigate", url="https://example.com")
        task2 = env2()
        assert task2.env is env2
        assert len(task2.env._setup_calls) == 1


class TestTaskFromV4:
    """Tests for Task.from_v4() migration helper."""

    def test_from_v4_with_legacy_task(self) -> None:
        """Task.from_v4() accepts LegacyTask object."""
        import warnings

        # Suppress the deprecation warning from LegacyTask
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            from hud.types import LegacyTask

            legacy = LegacyTask(
                prompt="Navigate to google.com",
                mcp_config={"hud": {"url": "https://mcp.hud.ai"}},
                evaluate_tool={"name": "check", "arguments": {}},
            )

        task = Task.from_v4(legacy)

        assert isinstance(task, Task)
        assert task.env is not None
        assert task.env.prompt == "Navigate to google.com"
        assert task.scenario is None  # Uses setup/evaluate_tool, not scenarios

    def test_from_v4_with_dict(self) -> None:
        """Task.from_v4() accepts dict with LegacyTask fields."""
        task = Task.from_v4(
            {
                "prompt": "Navigate to google.com",
                "mcp_config": {"hud": {"url": "https://mcp.hud.ai"}},
                "evaluate_tool": {"name": "check", "arguments": {}},
            }
        )

        assert isinstance(task, Task)
        assert task.env is not None
        assert task.env.prompt == "Navigate to google.com"

    def test_from_v4_with_json_string(self) -> None:
        """Task.from_v4() accepts JSON string."""
        import json

        data = {
            "prompt": "Navigate to google.com",
            "mcp_config": {"hud": {"url": "https://mcp.hud.ai"}},
            "evaluate_tool": {"name": "check", "arguments": {}},
        }
        task = Task.from_v4(json.dumps(data))

        assert isinstance(task, Task)
        assert task.env is not None
        assert task.env.prompt == "Navigate to google.com"

    def test_from_v4_with_setup_tool(self) -> None:
        """Task.from_v4() preserves setup_tool via env._setup_calls."""
        task = Task.from_v4(
            {
                "prompt": "Check URL",
                "mcp_config": {"hud": {"url": "https://mcp.hud.ai"}},
                "setup_tool": {"name": "navigate", "arguments": {"url": "https://google.com"}},
                "evaluate_tool": {"name": "check", "arguments": {}},
            }
        )

        # setup_tool is converted to env._setup_calls
        assert len(task.env._setup_calls) == 1
        assert task.env._setup_calls[0] == ("navigate", {"url": "https://google.com"})

    def test_from_v4_with_evaluate_tool(self) -> None:
        """Task.from_v4() preserves evaluate_tool via env._evaluate_calls."""
        task = Task.from_v4(
            {
                "prompt": "Check URL",
                "mcp_config": {"hud": {"url": "https://mcp.hud.ai"}},
                "evaluate_tool": {"name": "check_url", "arguments": {"expected": "google"}},
            }
        )

        # evaluate_tool is converted to env._evaluate_calls
        assert len(task.env._evaluate_calls) == 1
        assert task.env._evaluate_calls[0] == ("check_url", {"expected": "google"})

    def test_from_v4_with_invalid_type_raises(self) -> None:
        """Task.from_v4() raises TypeError for invalid input."""
        with pytest.raises(TypeError):
            Task.from_v4(12345)  # type: ignore[arg-type]

    def test_from_v4_with_invalid_json_raises(self) -> None:
        """Task.from_v4() raises JSONDecodeError for invalid JSON."""
        import json

        with pytest.raises(json.JSONDecodeError):
            Task.from_v4("not valid json")

    def test_from_v4_does_not_warn_on_use(self) -> None:
        """Task.from_v4() suppresses LegacyTask deprecation warning."""
        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            Task.from_v4(
                {
                    "prompt": "test",
                    "mcp_config": {"hud": {}},
                    "evaluate_tool": {"name": "check", "arguments": {}},
                }
            )

        # Should not trigger deprecation warning since we're migrating
        legacy_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
        assert len(legacy_warnings) == 0
