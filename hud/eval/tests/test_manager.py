"""Tests for hud.eval.manager module (hud.eval() function)."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from hud.eval.context import EvalContext, get_current_trace_headers
from hud.eval.manager import run_eval


class TestRunEvalNoArgs:
    """Tests for hud.eval() with no arguments (blank eval)."""

    @pytest.mark.asyncio
    async def test_blank_eval_creates_context(self) -> None:
        """hud.eval() with no args creates an EvalContext."""
        with (
            patch.object(EvalContext, "_eval_enter", new_callable=AsyncMock),
            patch.object(EvalContext, "_eval_exit", new_callable=AsyncMock),
        ):
            async with run_eval(quiet=True) as ctx:
                assert isinstance(ctx, EvalContext)
                assert ctx.eval_name == "eval"

    @pytest.mark.asyncio
    async def test_blank_eval_generates_trace_id(self) -> None:
        """hud.eval() with no args generates a trace_id."""
        with (
            patch.object(EvalContext, "_eval_enter", new_callable=AsyncMock),
            patch.object(EvalContext, "_eval_exit", new_callable=AsyncMock),
        ):
            async with run_eval(quiet=True) as ctx:
                assert ctx.trace_id is not None
                assert len(ctx.trace_id) == 36  # UUID format

    @pytest.mark.asyncio
    async def test_blank_eval_sets_trace_headers(self) -> None:
        """hud.eval() sets trace headers in contextvar during context."""
        with (
            patch.object(EvalContext, "_eval_enter", new_callable=AsyncMock),
            patch.object(EvalContext, "_eval_exit", new_callable=AsyncMock),
        ):
            # Before context, no headers
            assert get_current_trace_headers() is None

            async with run_eval(quiet=True) as ctx:
                # Inside context, headers are set
                headers = get_current_trace_headers()
                assert headers is not None
                assert headers["Trace-Id"] == ctx.trace_id

            # After context, headers are cleared
            assert get_current_trace_headers() is None

    @pytest.mark.asyncio
    async def test_blank_eval_reward_can_be_set(self) -> None:
        """hud.eval() allows setting reward on context."""
        with (
            patch.object(EvalContext, "_eval_enter", new_callable=AsyncMock),
            patch.object(EvalContext, "_eval_exit", new_callable=AsyncMock),
        ):
            async with run_eval(quiet=True) as ctx:
                assert ctx.reward is None
                ctx.reward = 0.95

            assert ctx.reward == 0.95

    @pytest.mark.asyncio
    async def test_blank_eval_reports_reward_on_exit(self) -> None:
        """hud.eval() reports reward to backend on exit."""
        with (
            patch.object(EvalContext, "_eval_enter", new_callable=AsyncMock),
            patch.object(EvalContext, "_eval_exit", new_callable=AsyncMock) as mock_exit,
        ):
            async with run_eval(quiet=True) as ctx:
                ctx.reward = 0.85

            # _eval_exit should have been called (with no error)
            mock_exit.assert_called_once_with(None)

    @pytest.mark.asyncio
    async def test_blank_eval_empty_variants(self) -> None:
        """hud.eval() with no args has empty variants dict."""
        with (
            patch.object(EvalContext, "_eval_enter", new_callable=AsyncMock),
            patch.object(EvalContext, "_eval_exit", new_callable=AsyncMock),
        ):
            async with run_eval(quiet=True) as ctx:
                assert ctx.variants == {}

    @pytest.mark.asyncio
    async def test_blank_eval_has_headers_property(self) -> None:
        """hud.eval() context has headers property for gateway integration."""
        with (
            patch.object(EvalContext, "_eval_enter", new_callable=AsyncMock),
            patch.object(EvalContext, "_eval_exit", new_callable=AsyncMock),
        ):
            async with run_eval(quiet=True) as ctx:
                headers = ctx.headers
                assert "Trace-Id" in headers
                assert headers["Trace-Id"] == ctx.trace_id


class TestRunEvalWithApiKey:
    """Tests for hud.eval() with api_key parameter."""

    @pytest.mark.asyncio
    async def test_api_key_passed_to_context(self) -> None:
        """hud.eval(api_key=...) passes api_key to context."""
        with (
            patch.object(EvalContext, "_eval_enter", new_callable=AsyncMock),
            patch.object(EvalContext, "_eval_exit", new_callable=AsyncMock),
        ):
            async with run_eval(api_key="test-key", quiet=True) as ctx:
                assert ctx._eval_api_key == "test-key"


class TestRunEvalWithJobId:
    """Tests for hud.eval() with job_id parameter."""

    @pytest.mark.asyncio
    async def test_job_id_passed_to_context(self) -> None:
        """hud.eval(job_id=...) passes job_id to context."""
        with (
            patch.object(EvalContext, "_eval_enter", new_callable=AsyncMock),
            patch.object(EvalContext, "_eval_exit", new_callable=AsyncMock),
        ):
            async with run_eval(job_id="job-123", quiet=True) as ctx:
                assert ctx.job_id == "job-123"


class TestRunEvalErrorHandling:
    """Tests for hud.eval() error handling."""

    @pytest.mark.asyncio
    async def test_error_tracked_on_exception(self) -> None:
        """hud.eval() tracks error when exception occurs."""
        with (
            patch.object(EvalContext, "_eval_enter", new_callable=AsyncMock),
            patch.object(EvalContext, "_eval_exit", new_callable=AsyncMock) as mock_exit,
        ):
            with pytest.raises(ValueError):
                async with run_eval(quiet=True):
                    raise ValueError("test error")

            # _eval_exit should have been called with error message
            mock_exit.assert_called_once()
            error_msg = mock_exit.call_args[0][0]
            assert error_msg is not None
            assert "test error" in error_msg
