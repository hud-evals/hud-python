from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from hud.datasets.runner import _flush_telemetry


@pytest.mark.asyncio
async def test_flush_telemetry():
    """Test _flush_telemetry function."""
    with (
        patch("hud.otel.config.is_telemetry_configured", return_value=True),
        patch("hud.utils.hud_console.hud_console"),
        patch("hud.utils.task_tracking.wait_all_tasks", new_callable=AsyncMock) as mock_wait,
        patch("opentelemetry.trace.get_tracer_provider") as mock_get_provider,
    ):
        from opentelemetry.sdk.trace import TracerProvider

        mock_provider = MagicMock(spec=TracerProvider)
        mock_provider.force_flush.return_value = True
        mock_get_provider.return_value = mock_provider

        mock_wait.return_value = 5

        await _flush_telemetry()

        mock_wait.assert_called_once()
        mock_provider.force_flush.assert_called_once_with(timeout_millis=20000)


@pytest.mark.asyncio
async def test_flush_telemetry_no_telemetry():
    """Test _flush_telemetry when telemetry is not configured."""
    with (
        patch("hud.otel.config.is_telemetry_configured", return_value=False),
        patch("hud.utils.hud_console.hud_console"),
        patch("hud.utils.task_tracking.wait_all_tasks", new_callable=AsyncMock) as mock_wait,
        patch("opentelemetry.trace.get_tracer_provider"),
    ):
        mock_wait.return_value = 0

        await _flush_telemetry()

        mock_wait.assert_called_once()


@pytest.mark.asyncio
async def test_flush_telemetry_exception():
    """Test _flush_telemetry handles exceptions gracefully."""
    with (
        patch("hud.otel.config.is_telemetry_configured", return_value=True),
        patch("hud.utils.hud_console.hud_console"),
        patch("hud.utils.task_tracking.wait_all_tasks", new_callable=AsyncMock) as mock_wait,
        patch("opentelemetry.trace.get_tracer_provider") as mock_get_provider,
    ):
        from opentelemetry.sdk.trace import TracerProvider

        mock_provider = MagicMock(spec=TracerProvider)
        mock_provider.force_flush.side_effect = Exception("Flush failed")
        mock_get_provider.return_value = mock_provider

        mock_wait.return_value = 3

        # Should not raise
        await _flush_telemetry()


@pytest.mark.asyncio
async def test_flush_telemetry_no_completed_tasks():
    """Test _flush_telemetry when no tasks were completed."""
    with (
        patch("hud.otel.config.is_telemetry_configured", return_value=True),
        patch("hud.utils.hud_console.hud_console"),
        patch("hud.utils.task_tracking.wait_all_tasks", new_callable=AsyncMock) as mock_wait,
        patch("opentelemetry.trace.get_tracer_provider") as mock_get_provider,
    ):
        from opentelemetry.sdk.trace import TracerProvider

        mock_provider = MagicMock(spec=TracerProvider)
        mock_get_provider.return_value = mock_provider

        mock_wait.return_value = 0

        await _flush_telemetry()

        mock_provider.force_flush.assert_called_once()


@pytest.mark.asyncio
async def test_flush_telemetry_non_sdk_provider():
    """Test _flush_telemetry with non-SDK TracerProvider."""
    with (
        patch("hud.otel.config.is_telemetry_configured", return_value=True),
        patch("hud.utils.hud_console.hud_console"),
        patch("hud.utils.task_tracking.wait_all_tasks", new_callable=AsyncMock) as mock_wait,
        patch("opentelemetry.trace.get_tracer_provider") as mock_get_provider,
    ):
        # Return a non-TracerProvider object
        mock_get_provider.return_value = MagicMock(spec=object)

        mock_wait.return_value = 2

        # Should not raise
        await _flush_telemetry()
