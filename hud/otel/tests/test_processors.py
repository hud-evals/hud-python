"""Tests for OpenTelemetry processors."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from hud.otel.processors import HudEnrichmentProcessor


class TestHudEnrichmentProcessor:
    """Test HudEnrichmentProcessor."""

    def test_on_start_with_run_id(self):
        """Test on_start with current task run ID."""

        processor = HudEnrichmentProcessor()

        # Mock span
        span = MagicMock()
        span.set_attribute = MagicMock()
        span.is_recording.return_value = True

        # Mock context with run ID
        with patch("hud.otel.processors.get_current_task_run_id", return_value="test-run-123"):
            processor.on_start(span, parent_context=None)

        # Verify attribute was set
        span.set_attribute.assert_called_with("hud.task_run_id", "test-run-123")

    def test_on_start_no_run_id(self):
        """Test on_start without current task run ID."""

        processor = HudEnrichmentProcessor()

        # Mock span
        span = MagicMock()
        span.set_attribute = MagicMock()
        span.is_recording.return_value = True

        # Mock context without run ID
        with patch("hud.otel.processors.get_current_task_run_id", return_value=None):
            processor.on_start(span, parent_context=None)

        # Verify no task run ID attribute was set
        span.set_attribute.assert_not_called()

    def test_on_end(self):
        """Test on_end does nothing."""

        processor = HudEnrichmentProcessor()
        span = MagicMock()

        # Should not raise
        processor.on_end(span)

    def test_shutdown(self):
        """Test shutdown does nothing."""

        processor = HudEnrichmentProcessor()

        # Should not raise
        processor.shutdown()

    def test_force_flush(self):
        """Test force_flush returns True."""

        processor = HudEnrichmentProcessor()

        # Should return True
        result = processor.force_flush()
        assert result is True

    def test_on_start_with_job_id(self):
        """Test on_start with job ID in baggage."""

        processor = HudEnrichmentProcessor()

        # Mock span
        span = MagicMock()
        span.set_attribute = MagicMock()
        span.is_recording.return_value = True

        # Mock baggage with job ID
        with patch("hud.otel.processors.baggage.get_baggage", return_value="job-123"):
            with patch("hud.otel.processors.get_current_task_run_id", return_value=None):
                processor.on_start(span, parent_context=None)

        # Verify job ID attribute was set
        span.set_attribute.assert_called_with("hud.job_id", "job-123")

    def test_on_start_exception_handling(self):
        """Test on_start handles exceptions gracefully."""

        processor = HudEnrichmentProcessor()

        # Mock span that raises exception
        span = MagicMock()
        span.is_recording.side_effect = Exception("Test error")

        # Should not raise
        processor.on_start(span, parent_context=None)
