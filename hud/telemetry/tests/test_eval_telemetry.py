"""Tests for EvalContext telemetry integration with mock backend."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import patch

import pytest

import hud
from hud.environment import Environment
from hud.eval import Task
from hud.telemetry.exporter import _pending_futures, _pending_spans


@pytest.fixture(autouse=True)
def clear_pending_state():
    """Clear pending spans and futures before and after each test."""
    _pending_spans.clear()
    _pending_futures.clear()
    yield
    _pending_spans.clear()
    _pending_futures.clear()


class TestEvalContextTelemetry:
    """Tests for EvalContext telemetry integration."""

    @pytest.mark.asyncio
    async def test_call_tool_records_span(self):
        """Test that call_tool records a span with correct format."""
        uploaded_spans: list[dict[str, Any]] = []

        def capture_upload(
            task_run_id: str,
            spans: list[dict[str, Any]],
            telemetry_url: str,
            api_key: str,
        ) -> bool:
            uploaded_spans.extend(spans)
            return True

        # Create environment with a simple tool
        env = Environment("test-env")

        @env.tool
        async def greet(name: str) -> str:
            """Say hello."""
            return f"Hello, {name}!"

        # Create task from environment
        task = Task(env=env)

        with (
            patch("hud.settings.settings") as mock_settings,
            patch("hud.telemetry.exporter._do_upload", side_effect=capture_upload),
            patch("hud.eval.context.make_request"),  # Don't send eval enter/exit
        ):
            mock_settings.api_key = "test-key"
            mock_settings.telemetry_enabled = True
            mock_settings.hud_telemetry_url = "https://api.hud.ai"
            mock_settings.hud_api_url = "https://api.hud.ai"

            async with hud.eval(task) as ctx:
                result = await ctx.call_tool("greet", name="World")
                # call_tool returns MCPToolResult with formatted content
                assert "Hello, World!" in str(result)
                trace_id = ctx.trace_id

            # Wait for thread pool
            await asyncio.sleep(0.2)

        # Verify span was recorded
        assert len(uploaded_spans) >= 1
        span = uploaded_spans[0]

        # Check span structure
        assert "name" in span
        assert "trace_id" in span
        assert "span_id" in span
        assert "start_time" in span
        assert "end_time" in span
        assert "status_code" in span
        assert "attributes" in span

        # Check attributes
        attrs = span["attributes"]
        assert attrs["task_run_id"] == trace_id
        assert attrs["category"] == "mcp"

    @pytest.mark.asyncio
    async def test_call_tool_records_error_span(self):
        """Test that failed call_tool records error span."""
        uploaded_spans: list[dict[str, Any]] = []

        def capture_upload(
            task_run_id: str,
            spans: list[dict[str, Any]],
            telemetry_url: str,
            api_key: str,
        ) -> bool:
            uploaded_spans.extend(spans)
            return True

        env = Environment("test-env")

        @env.tool
        async def failing_tool() -> str:
            """Always fails."""
            raise ValueError("Tool error")

        task = Task(env=env)

        with (
            patch("hud.settings.settings") as mock_settings,
            patch("hud.telemetry.exporter._do_upload", side_effect=capture_upload),
            patch("hud.eval.context.make_request"),
        ):
            mock_settings.api_key = "test-key"
            mock_settings.telemetry_enabled = True
            mock_settings.hud_telemetry_url = "https://api.hud.ai"
            mock_settings.hud_api_url = "https://api.hud.ai"

            async with hud.eval(task) as ctx:
                # Tool errors are wrapped in ToolError
                with pytest.raises(Exception, match="Tool error"):
                    await ctx.call_tool("failing_tool")

            await asyncio.sleep(0.2)

        # Should have recorded span with ERROR status
        assert len(uploaded_spans) >= 1
        span = uploaded_spans[0]
        assert span["status_code"] == "ERROR"
        # Error message contains the original error
        assert "Tool error" in (span.get("status_message") or "")

    @pytest.mark.asyncio
    async def test_multiple_call_tools_record_spans(self):
        """Test that multiple call_tool calls each record a span."""
        uploaded_spans: list[dict[str, Any]] = []

        def capture_upload(
            task_run_id: str,
            spans: list[dict[str, Any]],
            telemetry_url: str,
            api_key: str,
        ) -> bool:
            uploaded_spans.extend(spans)
            return True

        env = Environment("test-env")

        @env.tool
        async def add(a: int, b: int) -> int:
            """Add two numbers."""
            return a + b

        @env.tool
        async def multiply(a: int, b: int) -> int:
            """Multiply two numbers."""
            return a * b

        task = Task(env=env)

        with (
            patch("hud.settings.settings") as mock_settings,
            patch("hud.telemetry.exporter._do_upload", side_effect=capture_upload),
            patch("hud.eval.context.make_request"),
        ):
            mock_settings.api_key = "test-key"
            mock_settings.telemetry_enabled = True
            mock_settings.hud_telemetry_url = "https://api.hud.ai"
            mock_settings.hud_api_url = "https://api.hud.ai"

            async with hud.eval(task) as ctx:
                r1 = await ctx.call_tool("add", a=2, b=3)
                r2 = await ctx.call_tool("multiply", a=4, b=5)
                # Results are MCPToolResult objects
                assert "5" in str(r1)
                assert "20" in str(r2)

            await asyncio.sleep(0.2)

        # Should have 2 spans
        assert len(uploaded_spans) >= 2

    @pytest.mark.asyncio
    async def test_flush_called_on_context_exit(self):
        """Test that flush is called when context exits."""
        env = Environment("test-env")

        @env.tool
        async def simple_tool() -> str:
            return "done"

        task = Task(env=env)

        with (
            patch("hud.eval.context.flush") as mock_flush,
            patch("hud.settings.settings") as mock_settings,
            patch("hud.eval.context.make_request"),
        ):
            mock_settings.api_key = "test-key"
            mock_settings.telemetry_enabled = True
            mock_settings.hud_api_url = "https://api.hud.ai"

            async with hud.eval(task) as ctx:
                await ctx.call_tool("simple_tool")
                trace_id = ctx.trace_id

            # Verify flush was called with the trace_id
            mock_flush.assert_called_once_with(trace_id)

    @pytest.mark.asyncio
    async def test_telemetry_disabled_no_upload(self):
        """Test that no upload happens when telemetry is disabled."""
        upload_called = False

        def should_not_be_called(*args: Any, **kwargs: Any) -> bool:
            nonlocal upload_called
            upload_called = True
            return True

        env = Environment("test-env")

        @env.tool
        async def test_tool() -> str:
            return "ok"

        task = Task(env=env)

        with (
            patch("hud.settings.settings") as mock_settings,
            patch("hud.telemetry.exporter._do_upload", side_effect=should_not_be_called),
            patch("hud.eval.context.make_request"),
        ):
            mock_settings.api_key = "test-key"
            mock_settings.telemetry_enabled = False  # Disabled!
            mock_settings.hud_telemetry_url = "https://api.hud.ai"
            mock_settings.hud_api_url = "https://api.hud.ai"

            async with hud.eval(task) as ctx:
                await ctx.call_tool("test_tool")

            await asyncio.sleep(0.1)

        assert upload_called is False


class TestSpanFormat:
    """Tests for the format of recorded spans."""

    @pytest.mark.asyncio
    async def test_span_has_required_fields(self):
        """Test that spans have all required HudSpan fields."""
        uploaded_spans: list[dict[str, Any]] = []

        def capture_upload(
            task_run_id: str,
            spans: list[dict[str, Any]],
            telemetry_url: str,
            api_key: str,
        ) -> bool:
            uploaded_spans.extend(spans)
            return True

        env = Environment("test-env")

        @env.tool
        async def echo(message: str) -> str:
            return message

        task = Task(env=env)

        with (
            patch("hud.settings.settings") as mock_settings,
            patch("hud.telemetry.exporter._do_upload", side_effect=capture_upload),
            patch("hud.eval.context.make_request"),
        ):
            mock_settings.api_key = "test-key"
            mock_settings.telemetry_enabled = True
            mock_settings.hud_telemetry_url = "https://api.hud.ai"
            mock_settings.hud_api_url = "https://api.hud.ai"

            async with hud.eval(task) as ctx:
                await ctx.call_tool("echo", message="test")

            await asyncio.sleep(0.2)

        assert len(uploaded_spans) >= 1
        span = uploaded_spans[0]

        # Required fields from HudSpan
        assert "name" in span
        assert "trace_id" in span
        assert len(span["trace_id"]) == 32  # 32-char hex
        assert "span_id" in span
        assert len(span["span_id"]) == 16  # 16-char hex
        assert "start_time" in span
        assert "end_time" in span
        assert "status_code" in span
        assert span["status_code"] in ("OK", "ERROR", "UNSET")

        # Attributes
        assert "attributes" in span
        attrs = span["attributes"]
        assert "task_run_id" in attrs
        assert "category" in attrs

    @pytest.mark.asyncio
    async def test_span_timestamps_are_iso(self):
        """Test that span timestamps are in ISO format."""
        uploaded_spans: list[dict[str, Any]] = []

        def capture_upload(
            task_run_id: str,
            spans: list[dict[str, Any]],
            telemetry_url: str,
            api_key: str,
        ) -> bool:
            uploaded_spans.extend(spans)
            return True

        env = Environment("test-env")

        @env.tool
        async def noop() -> None:
            pass

        task = Task(env=env)

        with (
            patch("hud.settings.settings") as mock_settings,
            patch("hud.telemetry.exporter._do_upload", side_effect=capture_upload),
            patch("hud.eval.context.make_request"),
        ):
            mock_settings.api_key = "test-key"
            mock_settings.telemetry_enabled = True
            mock_settings.hud_telemetry_url = "https://api.hud.ai"
            mock_settings.hud_api_url = "https://api.hud.ai"

            async with hud.eval(task) as ctx:
                await ctx.call_tool("noop")

            await asyncio.sleep(0.2)

        span = uploaded_spans[0]

        # ISO format: YYYY-MM-DDTHH:MM:SS.ssssssZ
        import re

        iso_pattern = r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}"
        assert re.match(iso_pattern, span["start_time"])
        assert re.match(iso_pattern, span["end_time"])
