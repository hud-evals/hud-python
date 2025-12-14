"""Tests for hud.eval.context module."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from hud.eval.context import (
    EvalContext,
    get_current_trace_headers,
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
