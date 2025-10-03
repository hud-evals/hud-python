"""Tests for telemetry trace functionality."""

from __future__ import annotations

from unittest.mock import patch

from hud.telemetry.trace import trace


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
            mock_otel_trace.return_value.__enter__.return_value = "custom-otlp-trace"

            with trace("test-trace") as task_run_id:
                # In custom backend mode, the Otel trace id is returned from context
                assert isinstance(task_run_id.id, str)

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
