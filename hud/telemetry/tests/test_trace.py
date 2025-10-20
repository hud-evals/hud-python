"""Tests for telemetry trace functionality."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from hud.telemetry.trace import Trace, trace


class TestTraceAPI:
    """Tests for trace API function."""

    def test_trace_with_disabled_telemetry_and_no_api_key(self):
        """Test trace behavior when telemetry is disabled and no API key."""
        # Mock settings to disable telemetry and remove API key
        mock_settings = type("Settings", (), {"telemetry_enabled": False, "api_key": None})()

        with (
            patch("hud.settings.get_settings", return_value=mock_settings),
            patch("hud.telemetry.trace.OtelTrace") as mock_otel_trace,
        ):
            mock_otel_trace.return_value.__enter__.return_value = "1234567890"

            with trace("test-trace") as task_run_id:
                # Should use placeholder ID for custom backends
                assert len(task_run_id.id) == 36

    def test_trace_with_enabled_telemetry_and_api_key(self):
        """Test trace behavior when telemetry is enabled with API key."""
        mock_settings = type("Settings", (), {"telemetry_enabled": True, "api_key": "test-key"})()

        with (
            patch("hud.settings.get_settings", return_value=mock_settings),
            patch("hud.telemetry.trace.OtelTrace") as mock_otel_trace,
            patch("hud.telemetry.trace.uuid.uuid4") as mock_uuid,
        ):
            mock_uuid.return_value = "mock-uuid-123"
            mock_otel_trace.return_value.__enter__.return_value = "mock-uuid-123"

            with trace("test-trace") as task_run_id:
                # Should use generated UUID
                assert task_run_id.id == "mock-uuid-123"

    def test_trace_with_no_api_key(self):
        """Test trace behavior with no API key (custom backend scenario)."""
        mock_settings = type(
            "Settings",
            (),
            {
                "telemetry_enabled": True,  # Enabled but no API key
                "api_key": None,
            },
        )()

        with (
            patch("hud.settings.get_settings", return_value=mock_settings),
            patch("hud.telemetry.trace.OtelTrace") as mock_otel_trace,
        ):
            mock_otel_trace.return_value.__enter__.return_value = "custom-otlp-trace"

            with trace("test-trace") as task_run_id:
                # In absence of HUD API key, ID should still be a string
                assert isinstance(task_run_id.id, str)

    def test_trace_with_job_id(self):
        """Test trace with job_id parameter."""
        mock_settings = type("Settings", (), {"telemetry_enabled": True, "api_key": "test-key"})()

        with (
            patch("hud.settings.get_settings", return_value=mock_settings),
            patch("hud.telemetry.trace.OtelTrace") as mock_otel_trace,
            trace("test-trace", job_id="job-123") as trace_obj,
        ):
            assert trace_obj.job_id == "job-123"

            # Check OtelTrace was called with job_id
            call_kwargs = mock_otel_trace.call_args[1]
            assert call_kwargs["job_id"] == "job-123"

    def test_trace_with_task_id(self):
        """Test trace with task_id parameter."""
        mock_settings = type("Settings", (), {"telemetry_enabled": True, "api_key": "test-key"})()

        with (
            patch("hud.settings.get_settings", return_value=mock_settings),
            patch("hud.telemetry.trace.OtelTrace"),
            trace("test-trace", task_id="task-456") as trace_obj,
        ):
            assert trace_obj.task_id == "task-456"

    def test_trace_with_attributes(self):
        """Test trace with custom attributes."""
        mock_settings = type("Settings", (), {"telemetry_enabled": True, "api_key": "test-key"})()

        with (
            patch("hud.settings.get_settings", return_value=mock_settings),
            patch("hud.telemetry.trace.OtelTrace") as mock_otel_trace,
            trace("test-trace", attrs={"custom": "value"}),
        ):
            # Check OtelTrace was called with attributes
            call_kwargs = mock_otel_trace.call_args[1]
            assert call_kwargs["attributes"] == {"custom": "value"}

    def test_trace_non_root(self):
        """Test trace with root=False."""
        mock_settings = type("Settings", (), {"telemetry_enabled": True, "api_key": "test-key"})()

        with (
            patch("hud.settings.get_settings", return_value=mock_settings),
            patch("hud.telemetry.trace.OtelTrace") as mock_otel_trace,
            trace("test-trace", root=False),
        ):
            # Check OtelTrace was called with is_root=False
            call_kwargs = mock_otel_trace.call_args[1]
            assert call_kwargs["is_root"] is False


class TestTraceClass:
    """Tests for Trace class."""

    def test_trace_initialization(self):
        """Test Trace initialization."""
        trace_obj = Trace(
            trace_id="test-id",
            name="Test Trace",
            job_id="job-123",
            task_id="task-456",
        )

        assert trace_obj.id == "test-id"
        assert trace_obj.name == "Test Trace"
        assert trace_obj.job_id == "job-123"
        assert trace_obj.task_id == "task-456"
        assert trace_obj.created_at is not None

    @pytest.mark.asyncio
    async def test_trace_log(self):
        """Test Trace async log method."""
        trace_obj = Trace("test-id", "Test")

        with (
            patch("hud.telemetry.trace.settings") as mock_settings,
            patch("hud.telemetry.trace.make_request", new_callable=AsyncMock) as mock_request,
        ):
            mock_settings.telemetry_enabled = True
            mock_settings.api_key = "test-key"
            mock_settings.hud_telemetry_url = "https://test.com"

            await trace_obj.log({"metric": 1.0})

            mock_request.assert_called_once()
            call_kwargs = mock_request.call_args[1]
            assert call_kwargs["json"]["metrics"] == {"metric": 1.0}

    @pytest.mark.asyncio
    async def test_trace_log_telemetry_disabled(self):
        """Test Trace log when telemetry is disabled."""
        trace_obj = Trace("test-id", "Test")

        with (
            patch("hud.telemetry.trace.settings") as mock_settings,
            patch("hud.telemetry.trace.make_request", new_callable=AsyncMock) as mock_request,
        ):
            mock_settings.telemetry_enabled = False

            await trace_obj.log({"metric": 1.0})

            mock_request.assert_not_called()

    @pytest.mark.asyncio
    async def test_trace_log_error(self):
        """Test Trace log handles errors gracefully."""
        trace_obj = Trace("test-id", "Test")

        with (
            patch("hud.telemetry.trace.settings") as mock_settings,
            patch("hud.telemetry.trace.make_request", new_callable=AsyncMock) as mock_request,
        ):
            mock_settings.telemetry_enabled = True
            mock_settings.api_key = "test-key"
            mock_settings.hud_telemetry_url = "https://test.com"
            mock_request.side_effect = Exception("Network error")

            # Should not raise
            await trace_obj.log({"metric": 1.0})

    def test_trace_log_sync(self):
        """Test Trace sync log method."""
        trace_obj = Trace("test-id", "Test")

        with (
            patch("hud.telemetry.trace.settings") as mock_settings,
            patch("hud.telemetry.trace.make_request_sync") as mock_request,
        ):
            mock_settings.telemetry_enabled = True
            mock_settings.api_key = "test-key"
            mock_settings.hud_telemetry_url = "https://test.com"

            trace_obj.log_sync({"metric": 1.0})

            mock_request.assert_called_once()

    def test_trace_log_sync_telemetry_disabled(self):
        """Test Trace sync log when telemetry is disabled."""
        trace_obj = Trace("test-id", "Test")

        with (
            patch("hud.telemetry.trace.settings") as mock_settings,
            patch("hud.telemetry.trace.make_request_sync") as mock_request,
        ):
            mock_settings.telemetry_enabled = False

            trace_obj.log_sync({"metric": 1.0})

            mock_request.assert_not_called()

    def test_trace_log_sync_error(self):
        """Test Trace sync log handles errors gracefully."""
        trace_obj = Trace("test-id", "Test")

        with (
            patch("hud.telemetry.trace.settings") as mock_settings,
            patch("hud.telemetry.trace.make_request_sync") as mock_request,
        ):
            mock_settings.telemetry_enabled = True
            mock_settings.api_key = "test-key"
            mock_settings.hud_telemetry_url = "https://test.com"
            mock_request.side_effect = Exception("Network error")

            # Should not raise
            trace_obj.log_sync({"metric": 1.0})

    def test_trace_repr(self):
        """Test Trace __repr__."""
        trace_obj = Trace("test-id", "Test Trace")

        repr_str = repr(trace_obj)
        assert "test-id" in repr_str
        assert "Test Trace" in repr_str
