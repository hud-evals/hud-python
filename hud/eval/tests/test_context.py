"""Tests for hud.eval.context module."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from hud.eval.context import (
    EvalContext,
    get_current_trace_headers,
    get_current_trace_id,
    set_trace_context,
)


class TestEvalContext:
    """Tests for EvalContext."""

    def test_init_generates_trace_id(self) -> None:
        """EvalContext generates trace_id if not provided."""
        ctx = EvalContext(name="test-task", quiet=True)

        assert ctx.trace_id is not None
        assert len(ctx.trace_id) == 36  # UUID format

    def test_init_uses_provided_trace_id(self) -> None:
        """EvalContext uses provided trace_id."""
        ctx = EvalContext(name="test-task", trace_id="custom-id", quiet=True)

        assert ctx.trace_id == "custom-id"

    def test_headers_contains_trace_id(self) -> None:
        """headers property returns dict with trace ID."""
        ctx = EvalContext(name="test-task", trace_id="test-123", quiet=True)

        assert ctx.headers == {"Trace-Id": "test-123"}

    def test_success_true_when_no_error(self) -> None:
        """success property returns True when no error."""
        ctx = EvalContext(name="test-task", quiet=True)

        assert ctx.success is True

    def test_success_false_when_error(self) -> None:
        """success property returns False when error is set."""
        ctx = EvalContext(name="test-task", quiet=True)
        ctx.error = ValueError("test error")

        assert ctx.success is False

    def test_variants_empty_by_default(self) -> None:
        """variants is empty dict by default."""
        ctx = EvalContext(name="test-task", quiet=True)

        assert ctx.variants == {}

    def test_variants_set_from_init(self) -> None:
        """variants set from parameter."""
        ctx = EvalContext(
            name="test-task",
            variants={"model": "gpt-4o", "temp": 0.7},
            quiet=True,
        )

        assert ctx.variants == {"model": "gpt-4o", "temp": 0.7}

    @pytest.mark.asyncio
    async def test_context_manager_sets_headers(self) -> None:
        """Context manager sets trace headers in contextvar."""
        ctx = EvalContext(name="test-task", trace_id="test-123", quiet=True)

        # Mock telemetry calls
        with (
            patch.object(ctx, "_eval_enter", new_callable=AsyncMock),
            patch.object(ctx, "_eval_exit", new_callable=AsyncMock),
            patch.object(EvalContext, "__aenter__", return_value=ctx),
            patch.object(EvalContext, "__aexit__", return_value=None),
        ):
            assert get_current_trace_headers() is None

            # Manually set token for test
            from hud.eval.context import _current_trace_headers

            token = _current_trace_headers.set(ctx.headers)
            try:
                headers = get_current_trace_headers()
                assert headers is not None
                assert headers["Trace-Id"] == "test-123"
            finally:
                _current_trace_headers.reset(token)

            assert get_current_trace_headers() is None

    def test_set_trace_context(self) -> None:
        """set_trace_context sets and resets Trace-Id."""
        assert get_current_trace_id() is None

        with set_trace_context("test-trace-123"):
            assert get_current_trace_id() == "test-trace-123"

        assert get_current_trace_id() is None

    def test_repr(self) -> None:
        """__repr__ shows useful info."""
        ctx = EvalContext(
            name="test-task",
            trace_id="abc12345-6789-0000-0000-000000000000",
            quiet=True,
        )
        ctx.reward = 0.95

        repr_str = repr(ctx)
        assert "abc12345" in repr_str
        assert "test-task" in repr_str
        assert "0.95" in repr_str


class TestScenarioErrorPropagation:
    """Tests for scenario evaluate errors being captured on EvalContext."""

    @pytest.mark.asyncio
    async def test_scenario_evaluate_error_sets_context_error(self) -> None:
        """Scenario evaluate failure sets self.error on EvalContext."""
        ctx = EvalContext(name="test-task", quiet=True)
        # Simulate a task with a scenario
        mock_task = MagicMock()
        mock_task.scenario = "test-scenario"
        ctx._task = mock_task

        async def failing_evaluate(name: str):
            raise RuntimeError("Command '['git', 'apply']' returned non-zero exit status 1.")

        ctx.run_scenario_evaluate = failing_evaluate  # type: ignore[method-assign]

        await ctx._run_task_scenario_evaluate()

        assert ctx.error is not None
        assert "git" in str(ctx.error)
        assert ctx.success is False
        assert ctx.reward is None

    @pytest.mark.asyncio
    async def test_scenario_evaluate_success_sets_reward(self) -> None:
        """Successful scenario evaluate sets reward and evaluation_result."""
        from hud.tools.types import EvaluationResult

        ctx = EvalContext(name="test-task", quiet=True)
        mock_task = MagicMock()
        mock_task.scenario = "test-scenario"
        ctx._task = mock_task

        async def successful_evaluate(name: str):
            return EvaluationResult(reward=0.85, done=True)

        ctx.run_scenario_evaluate = successful_evaluate  # type: ignore[method-assign]

        await ctx._run_task_scenario_evaluate()

        assert ctx.error is None
        assert ctx.success is True
        assert ctx.reward == 0.85
        assert ctx.evaluation_result is not None
        assert ctx.evaluation_result.reward == 0.85


class TestEvalContextPrompt:
    """Tests for EvalContext.prompt feature."""

    def test_prompt_can_be_set(self) -> None:
        """EvalContext.prompt can be set."""
        ctx = EvalContext(name="test-task", quiet=True)
        ctx.prompt = "Test prompt"

        assert ctx.prompt == "Test prompt"

    def test_prompt_included_in_payload(self) -> None:
        """Prompt is included in eval payload."""
        ctx = EvalContext(name="test-task", quiet=True)
        ctx.prompt = "Test prompt"

        payload = ctx._build_base_payload()
        assert payload.prompt == "Test prompt"


class TestEvalContextFromEnvironment:
    """Tests for EvalContext.from_environment factory."""

    def test_copies_connections(self) -> None:
        """from_environment copies connections from parent (deep copy)."""
        from hud.environment import Environment

        parent = Environment("parent-env")
        # Add a mock connection with copy method
        mock_conn = MagicMock()
        mock_conn_copy = MagicMock()
        mock_conn.copy.return_value = mock_conn_copy
        parent._connections["test-conn"] = mock_conn

        ctx = EvalContext.from_environment(parent, name="test-task")

        # Verify connection was copied (not same object)
        assert "test-conn" in ctx._connections
        mock_conn.copy.assert_called_once()
        assert ctx._connections["test-conn"] is mock_conn_copy

    def test_copies_prompt(self) -> None:
        """from_environment copies prompt from parent."""
        from hud.environment import Environment

        parent = Environment("parent-env")
        parent.prompt = "Parent prompt"

        ctx = EvalContext.from_environment(parent, name="test-task")

        assert ctx.prompt == "Parent prompt"

    def test_sets_eval_properties(self) -> None:
        """from_environment sets eval-specific properties."""
        from hud.environment import Environment

        parent = Environment("parent-env")

        ctx = EvalContext.from_environment(
            parent,
            name="test-task",
            trace_id="custom-trace",
            variants={"model": "gpt-4o"},
            group_id="group-123",
            index=5,
        )

        assert ctx.eval_name == "test-task"
        assert ctx.trace_id == "custom-trace"
        assert ctx.variants == {"model": "gpt-4o"}
        assert ctx.group_id == "group-123"
        assert ctx.index == 5

    def test_assigns_hud_environment_headers_per_context(self) -> None:
        """Each EvalContext gets its own HUD environment id."""
        from hud.environment import Environment
        from hud.environment.connection import ConnectionConfig, ConnectionType, Connector

        parent = Environment("parent-env")
        parent_headers = {
            "Environment-Name": "browser",
            "Environment-Id": "parent-env-id",
            "mcp-session-id": "parent-session-id",
        }
        parent._connections["hud"] = Connector(
            transport=SimpleNamespace(url="https://mcp.hud.so/jsonrpc", headers=parent_headers),
            config=ConnectionConfig(),
            name="hud",
            connection_type=ConnectionType.REMOTE,
        )

        ctx_a = EvalContext.from_environment(parent, name="task-a", trace_id="trace-a")
        ctx_b = EvalContext.from_environment(parent, name="task-b", trace_id="trace-b")

        headers_a = ctx_a._connections["hud"]._transport.headers
        headers_b = ctx_b._connections["hud"]._transport.headers

        assert headers_a["Environment-Name"] == "browser"
        assert headers_b["Environment-Name"] == "browser"
        assert headers_a["Environment-Id"] != "parent-env-id"
        assert headers_b["Environment-Id"] != "parent-env-id"
        assert headers_a["Environment-Id"] != headers_b["Environment-Id"]

        assert headers_a is not headers_b
        assert parent_headers["Environment-Id"] == "parent-env-id"
        assert parent_headers["mcp-session-id"] == "parent-session-id"

    def test_does_not_rewrite_non_hud_headers(self) -> None:
        """Non-HUD MCP connectors keep their existing env/session headers."""
        from hud.environment import Environment
        from hud.environment.connection import ConnectionConfig, ConnectionType, Connector

        parent = Environment("parent-env")
        original_headers = {
            "Environment-Name": "browser",
            "Environment-Id": "existing-env-id",
            "mcp-session-id": "existing-session-id",
        }
        parent._connections["external"] = Connector(
            transport=SimpleNamespace(url="https://example.com/mcp", headers=original_headers),
            config=ConnectionConfig(),
            name="external",
            connection_type=ConnectionType.REMOTE,
        )

        ctx = EvalContext.from_environment(parent, name="task-a", trace_id="trace-a")
        copied_headers = ctx._connections["external"]._transport.headers

        assert copied_headers["Environment-Id"] == "existing-env-id"
        assert copied_headers["mcp-session-id"] == "existing-session-id"


class TestEvalContextFromTask:
    """Tests for EvalContext.from_task factory."""

    def test_v5_validation_populates_integration_calls(self) -> None:
        """Task.validation is mapped to integration test calls for replay."""
        from hud.environment import Environment
        from hud.eval.task import Task
        from hud.types import MCPToolCall

        env = Environment("test-env")
        validation_calls = [
            MCPToolCall(name="tool_a", arguments={"x": 1}),
            MCPToolCall(name="tool_b", arguments={"y": "ok"}),
        ]
        task = Task(
            env=env,
            scenario="demo",
            args={},
            validation=validation_calls,
        )

        ctx = EvalContext.from_task(task)
        assert ctx._integration_test_calls == [
            ("tool_a", {"x": 1}),
            ("tool_b", {"y": "ok"}),
        ]

    def test_v5_validation_overrides_environment_integration_calls(self) -> None:
        """Task.validation takes precedence over env-level integration calls."""
        from hud.environment import Environment
        from hud.eval.task import Task
        from hud.types import MCPToolCall

        env = Environment("test-env")
        env._integration_test_calls = [("old_tool", {"stale": True})]

        task = Task(
            env=env,
            scenario="demo",
            args={},
            validation=[MCPToolCall(name="new_tool", arguments={"fresh": True})],
        )

        ctx = EvalContext.from_task(task)
        assert ctx._integration_test_calls == [("new_tool", {"fresh": True})]

    def test_v5_empty_validation_clears_environment_integration_calls(self) -> None:
        """Task.validation=[] still overrides env-level integration calls."""
        from hud.environment import Environment
        from hud.eval.task import Task

        env = Environment("test-env")
        env._integration_test_calls = [("old_tool", {"stale": True})]

        task = Task(
            env=env,
            scenario="demo",
            args={},
            validation=[],
        )

        ctx = EvalContext.from_task(task)

        assert ctx._integration_test_calls == []

    def test_v4_integration_test_tool_remains_supported(self) -> None:
        """Legacy integration_test_tool still populates integration calls."""
        from hud.eval.task import Task

        task = Task.from_v4(
            {
                "prompt": "test",
                "mcp_config": {"server": {"url": "http://localhost"}},
                "evaluate_tool": {"name": "check", "arguments": {}},
                "integration_test_tool": [
                    {"name": "legacy_tool", "arguments": {"v": 1}},
                ],
            }
        )

        ctx = EvalContext.from_task(task)
        assert ctx._integration_test_calls == [("legacy_tool", {"v": 1})]

    def test_v5_validation_replays_with_integration_runner(self) -> None:
        """IntegrationTestRunner executes v5 Task.validation calls via EvalContext.from_task."""
        import asyncio

        from mcp import types as mcp_types

        from hud.agents.misc import IntegrationTestRunner
        from hud.environment import Environment
        from hud.eval.task import Task
        from hud.types import MCPToolCall, MCPToolResult

        executed_calls: list[tuple[str, dict[str, object]]] = []

        async def _run() -> None:
            env = Environment("test-env")
            validation_calls = [
                MCPToolCall(name="tool_a", arguments={"x": 1}),
                MCPToolCall(name="tool_b", arguments={"y": "ok"}),
            ]
            task = Task(
                env=env,
                scenario="demo",
                args={},
                validation=validation_calls,
            )

            ctx = EvalContext.from_task(task)

            async def fake_call_tool(call, /, **kwargs):
                if isinstance(call, tuple):
                    name = str(call[0])
                    arguments = dict(call[1]) if len(call) > 1 else {}
                else:
                    name = str(call)
                    arguments = {}
                executed_calls.append((name, arguments))
                return MCPToolResult(
                    content=[mcp_types.TextContent(type="text", text="ok")],
                    isError=False,
                )

            ctx.call_tool = fake_call_tool  # type: ignore[method-assign]

            runner = IntegrationTestRunner.create()
            result = await runner.run(ctx)
            assert result.done is True

        asyncio.run(_run())

        assert executed_calls == [
            ("tool_a", {"x": 1}),
            ("tool_b", {"y": "ok"}),
        ]
