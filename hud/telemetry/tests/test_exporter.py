"""Tests for telemetry exporter with mock backend."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import patch

import pytest

from hud.telemetry.exporter import (
    _do_upload,
    _pending_futures,
    _pending_spans,
    flush,
    queue_span,
    shutdown,
)


@pytest.fixture(autouse=True)
def clear_pending_state():
    """Clear pending spans and futures before and after each test."""
    _pending_spans.clear()
    _pending_futures.clear()
    yield
    _pending_spans.clear()
    _pending_futures.clear()


class TestDoUpload:
    """Tests for _do_upload function."""

    def test_upload_success(self):
        """Test successful upload."""
        with patch("hud.telemetry.exporter.make_request_sync") as mock_request:
            result = _do_upload(
                task_run_id="test-task-123",
                spans=[{"name": "test.span", "attributes": {"task_run_id": "test-task-123"}}],
                telemetry_url="https://api.hud.ai",
                api_key="test-key",
            )

            assert result is True
            mock_request.assert_called_once()
            call_kwargs = mock_request.call_args.kwargs
            assert call_kwargs["method"] == "POST"
            assert "test-task-123" in call_kwargs["url"]
            assert call_kwargs["api_key"] == "test-key"
            assert "telemetry" in call_kwargs["json"]

    def test_upload_failure(self):
        """Test upload failure handling."""
        with patch("hud.telemetry.exporter.make_request_sync") as mock_request:
            mock_request.side_effect = Exception("Network error")

            result = _do_upload(
                task_run_id="test-task-123",
                spans=[{"name": "test.span"}],
                telemetry_url="https://api.hud.ai",
                api_key="test-key",
            )

            assert result is False


class TestQueueSpan:
    """Tests for queue_span function."""

    def test_queue_span_without_api_key(self):
        """Test that spans are not queued without API key."""
        with patch("hud.settings.settings") as mock_settings:
            mock_settings.api_key = None
            mock_settings.telemetry_enabled = True

            queue_span({"name": "test", "attributes": {"task_run_id": "123"}})

            assert len(_pending_spans) == 0

    def test_queue_span_without_telemetry_enabled(self):
        """Test that spans are not queued when telemetry disabled."""
        with patch("hud.settings.settings") as mock_settings:
            mock_settings.api_key = "test-key"
            mock_settings.telemetry_enabled = False

            queue_span({"name": "test", "attributes": {"task_run_id": "123"}})

            assert len(_pending_spans) == 0

    def test_queue_span_without_task_run_id(self):
        """Test that spans without task_run_id are ignored."""
        with patch("hud.settings.settings") as mock_settings:
            mock_settings.api_key = "test-key"
            mock_settings.telemetry_enabled = True

            queue_span({"name": "test", "attributes": {}})

            assert len(_pending_spans) == 0

    def test_queue_span_adds_to_pending(self):
        """Test that spans are added to pending list."""
        # Don't mock _do_upload so spans stay in pending
        with patch("hud.settings.settings") as mock_settings:
            mock_settings.api_key = "test-key"
            mock_settings.telemetry_enabled = True
            mock_settings.hud_telemetry_url = "https://api.hud.ai"

            # Use a sync context (no event loop) so upload happens sync
            # But we'll make it fail so span stays in pending
            with patch("hud.telemetry.exporter._do_upload", return_value=False):
                span = {"name": "test", "attributes": {"task_run_id": "task-123"}}
                queue_span(span)

                # Span should be in pending (upload failed so not removed)
                assert "task-123" in _pending_spans
                assert span in _pending_spans["task-123"]

    @pytest.mark.asyncio
    async def test_queue_span_uploads_async(self):
        """Test that spans are uploaded via thread pool in async context."""
        uploaded_spans: list[dict[str, Any]] = []

        def mock_upload(
            task_run_id: str,
            spans: list[dict[str, Any]],
            telemetry_url: str,
            api_key: str,
        ) -> bool:
            uploaded_spans.extend(spans)
            return True

        with (
            patch("hud.settings.settings") as mock_settings,
            patch("hud.telemetry.exporter._do_upload", side_effect=mock_upload),
        ):
            mock_settings.api_key = "test-key"
            mock_settings.telemetry_enabled = True
            mock_settings.hud_telemetry_url = "https://api.hud.ai"

            span = {"name": "test.async", "attributes": {"task_run_id": "async-task"}}
            queue_span(span)

            # Wait for thread pool to complete
            await asyncio.sleep(0.1)

            assert len(uploaded_spans) == 1
            assert uploaded_spans[0]["name"] == "test.async"


class TestFlush:
    """Tests for flush function."""

    def test_flush_specific_task(self):
        """Test flushing spans for specific task."""
        uploaded: list[tuple[str, list[dict[str, Any]]]] = []

        def mock_upload(
            task_run_id: str,
            spans: list[dict[str, Any]],
            telemetry_url: str,
            api_key: str,
        ) -> bool:
            uploaded.append((task_run_id, spans))
            return True

        with (
            patch("hud.settings.settings") as mock_settings,
            patch("hud.telemetry.exporter._do_upload", side_effect=mock_upload),
        ):
            mock_settings.api_key = "test-key"
            mock_settings.telemetry_enabled = True
            mock_settings.hud_telemetry_url = "https://api.hud.ai"

            # Add spans for two tasks
            _pending_spans["task-1"].append({"name": "span1"})
            _pending_spans["task-2"].append({"name": "span2"})

            # Flush only task-1
            flush("task-1")

            assert len(uploaded) == 1
            assert uploaded[0][0] == "task-1"
            assert "task-1" not in _pending_spans
            assert "task-2" in _pending_spans

    def test_flush_all_tasks(self):
        """Test flushing all pending spans."""
        uploaded: list[tuple[str, list[dict[str, Any]]]] = []

        def mock_upload(
            task_run_id: str,
            spans: list[dict[str, Any]],
            telemetry_url: str,
            api_key: str,
        ) -> bool:
            uploaded.append((task_run_id, spans))
            return True

        with (
            patch("hud.settings.settings") as mock_settings,
            patch("hud.telemetry.exporter._do_upload", side_effect=mock_upload),
        ):
            mock_settings.api_key = "test-key"
            mock_settings.telemetry_enabled = True
            mock_settings.hud_telemetry_url = "https://api.hud.ai"

            _pending_spans["task-1"].append({"name": "span1"})
            _pending_spans["task-2"].append({"name": "span2"})

            flush()

            assert len(uploaded) == 2
            assert len(_pending_spans) == 0

    def test_flush_clears_without_api_key(self):
        """Test that flush clears spans when no API key."""
        with patch("hud.settings.settings") as mock_settings:
            mock_settings.api_key = None
            mock_settings.telemetry_enabled = True

            _pending_spans["task-1"].append({"name": "span1"})

            flush()

            assert len(_pending_spans) == 0


class TestShutdown:
    """Tests for shutdown function."""

    def test_shutdown_flushes_pending(self):
        """Test that shutdown flushes pending spans."""
        uploaded: list[str] = []

        def mock_upload(
            task_run_id: str,
            spans: list[dict[str, Any]],
            telemetry_url: str,
            api_key: str,
        ) -> bool:
            uploaded.append(task_run_id)
            return True

        with (
            patch("hud.settings.settings") as mock_settings,
            patch("hud.telemetry.exporter._do_upload", side_effect=mock_upload),
            patch("hud.telemetry.exporter._get_api_key", return_value="test-key"),
        ):
            mock_settings.api_key = "test-key"
            mock_settings.telemetry_enabled = True
            mock_settings.hud_telemetry_url = "https://api.hud.ai"

            _pending_spans["shutdown-task"].append({"name": "final-span"})

            result = shutdown(timeout=1.0)

            assert result is True
            assert "shutdown-task" in uploaded
