from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from hud.telemetry.utils import flush_telemetry


@pytest.mark.asyncio
async def test_flush_telemetry():
    """Test flush_telemetry function."""
    with (
        patch("hud.otel.config.is_telemetry_configured", return_value=True),
        patch("hud.utils.hud_console.hud_console"),
        patch("opentelemetry.trace.get_tracer_provider") as mock_get_provider,
    ):
        from opentelemetry.sdk.trace import TracerProvider

        mock_provider = MagicMock(spec=TracerProvider)
        mock_provider.force_flush.return_value = True
        mock_get_provider.return_value = mock_provider

        await flush_telemetry()

        mock_provider.force_flush.assert_called_once_with(timeout_millis=5000)


@pytest.mark.asyncio
async def test_flush_telemetry_not_configured():
    """Test flush_telemetry when telemetry is not configured."""
    with patch("hud.otel.config.is_telemetry_configured", return_value=False):
        await flush_telemetry()


@pytest.mark.asyncio
async def test_flush_telemetry_exception():
    """Test flush_telemetry handles exceptions gracefully."""
    with (
        patch("hud.otel.config.is_telemetry_configured", return_value=True),
        patch("hud.utils.hud_console.hud_console"),
        patch("opentelemetry.trace.get_tracer_provider") as mock_get_provider,
    ):
        from opentelemetry.sdk.trace import TracerProvider

        mock_provider = MagicMock(spec=TracerProvider)
        mock_provider.force_flush.side_effect = Exception("Flush failed")
        mock_get_provider.return_value = mock_provider

        await flush_telemetry()


@pytest.mark.asyncio
async def test_flush_telemetry_timeout():
    """Test flush_telemetry when force_flush times out."""
    with (
        patch("hud.otel.config.is_telemetry_configured", return_value=True),
        patch("hud.utils.hud_console.hud_console"),
        patch("opentelemetry.trace.get_tracer_provider") as mock_get_provider,
    ):
        from opentelemetry.sdk.trace import TracerProvider

        mock_provider = MagicMock(spec=TracerProvider)
        mock_provider.force_flush.return_value = False
        mock_get_provider.return_value = mock_provider

        await flush_telemetry()
