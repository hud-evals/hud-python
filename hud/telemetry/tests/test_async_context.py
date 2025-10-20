from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from hud.telemetry.async_context import async_job, async_trace


@pytest.mark.asyncio
async def test_async_trace_basic():
    """Test basic AsyncTrace usage."""
    with (
        patch("hud.telemetry.async_context.OtelTrace") as mock_otel,
        patch("hud.telemetry.async_context.track_task"),
        patch("hud.telemetry.async_context._print_trace_url"),
        patch("hud.telemetry.async_context._print_trace_complete_url"),
    ):
        mock_otel_instance = MagicMock()
        mock_otel.return_value = mock_otel_instance

        async with async_trace("Test Task") as trace_obj:
            assert trace_obj.name == "Test Task"
            assert trace_obj.id is not None


@pytest.mark.asyncio
async def test_async_trace_with_job_id():
    """Test AsyncTrace with job_id parameter."""
    with (
        patch("hud.telemetry.async_context.OtelTrace") as mock_otel,
        patch("hud.telemetry.async_context.track_task"),
    ):
        mock_otel_instance = MagicMock()
        mock_otel.return_value = mock_otel_instance

        async with async_trace("Test", job_id="job-123") as trace_obj:
            assert trace_obj.job_id == "job-123"


@pytest.mark.asyncio
async def test_async_trace_with_task_id():
    """Test AsyncTrace with task_id parameter."""
    with (
        patch("hud.telemetry.async_context.OtelTrace") as mock_otel,
        patch("hud.telemetry.async_context.track_task"),
    ):
        mock_otel_instance = MagicMock()
        mock_otel.return_value = mock_otel_instance

        async with async_trace("Test", task_id="task-456") as trace_obj:
            assert trace_obj.task_id == "task-456"


@pytest.mark.asyncio
async def test_async_trace_prints_url_without_job():
    """Test AsyncTrace prints URL when not part of a job."""
    with (
        patch("hud.telemetry.async_context.settings") as mock_settings,
        patch("hud.telemetry.async_context.OtelTrace") as mock_otel,
        patch("hud.telemetry.async_context.track_task"),
        patch("hud.telemetry.async_context._print_trace_url") as mock_print_url,
    ):
        mock_settings.telemetry_enabled = True
        mock_settings.api_key = "test-key"
        mock_otel_instance = MagicMock()
        mock_otel.return_value = mock_otel_instance

        async with async_trace("Test", job_id=None):
            pass

        # Should print trace URL
        mock_print_url.assert_called_once()


@pytest.mark.asyncio
async def test_async_trace_no_print_url_with_job():
    """Test AsyncTrace doesn't print URL when part of a job."""
    with (
        patch("hud.telemetry.async_context.settings") as mock_settings,
        patch("hud.telemetry.async_context.OtelTrace") as mock_otel,
        patch("hud.telemetry.async_context.track_task"),
        patch("hud.telemetry.async_context._print_trace_url") as mock_print_url,
    ):
        mock_settings.telemetry_enabled = True
        mock_settings.api_key = "test-key"
        mock_otel_instance = MagicMock()
        mock_otel.return_value = mock_otel_instance

        async with async_trace("Test", job_id="job-123"):
            pass

        # Should NOT print trace URL when job_id is set
        mock_print_url.assert_not_called()


@pytest.mark.asyncio
async def test_async_trace_with_exception():
    """Test AsyncTrace handles exceptions."""
    with (
        patch("hud.telemetry.async_context.settings") as mock_settings,
        patch("hud.telemetry.async_context.OtelTrace") as mock_otel,
        patch("hud.telemetry.async_context.track_task"),
        patch("hud.telemetry.async_context._print_trace_complete_url") as mock_print,
    ):
        # Enable telemetry for this test
        mock_settings.telemetry_enabled = True
        mock_settings.api_key = "test-key"

        mock_otel_instance = MagicMock()
        mock_otel.return_value = mock_otel_instance

        with pytest.raises(ValueError):
            async with async_trace("Test"):
                raise ValueError("Test error")

        # Should have been called with error_occurred keyword arg
        mock_print.assert_called_once()
        call_kwargs = mock_print.call_args[1]
        assert call_kwargs["error_occurred"] is True


@pytest.mark.asyncio
async def test_async_job_basic():
    """Test basic AsyncJob usage."""
    with (
        patch("hud.telemetry.async_context.track_task"),
        patch("hud.telemetry.async_context._print_job_url"),
        patch("hud.telemetry.async_context._print_job_complete_url"),
    ):
        async with async_job("Test Job") as job_obj:
            assert job_obj.name == "Test Job"
            assert job_obj.id is not None


@pytest.mark.asyncio
async def test_async_job_with_metadata():
    """Test AsyncJob with metadata."""
    with (
        patch("hud.telemetry.async_context.track_task"),
        patch("hud.telemetry.async_context._print_job_url"),
        patch("hud.telemetry.async_context._print_job_complete_url"),
    ):
        async with async_job("Test", metadata={"key": "value"}) as job_obj:
            assert job_obj.metadata == {"key": "value"}


@pytest.mark.asyncio
async def test_async_job_with_dataset_link():
    """Test AsyncJob with dataset_link."""
    with (
        patch("hud.telemetry.async_context.track_task"),
        patch("hud.telemetry.async_context._print_job_url"),
        patch("hud.telemetry.async_context._print_job_complete_url"),
    ):
        async with async_job("Test", dataset_link="test/dataset") as job_obj:
            assert job_obj.dataset_link == "test/dataset"


@pytest.mark.asyncio
async def test_async_job_with_custom_job_id():
    """Test AsyncJob with custom job_id."""
    with (
        patch("hud.telemetry.async_context.track_task"),
        patch("hud.telemetry.async_context._print_job_url"),
        patch("hud.telemetry.async_context._print_job_complete_url"),
    ):
        async with async_job("Test", job_id="custom-id") as job_obj:
            assert job_obj.id == "custom-id"


@pytest.mark.asyncio
async def test_async_job_with_exception():
    """Test AsyncJob handles exceptions."""
    with (
        patch("hud.telemetry.async_context.track_task"),
        patch("hud.telemetry.async_context._print_job_url"),
        patch("hud.telemetry.async_context._print_job_complete_url") as mock_print,
    ):
        with pytest.raises(ValueError):
            async with async_job("Test"):
                raise ValueError("Job error")

        # Should print with error_occurred keyword arg
        mock_print.assert_called_once()
        call_kwargs = mock_print.call_args[1]
        assert call_kwargs["error_occurred"] is True


@pytest.mark.asyncio
async def test_async_job_status_updates():
    """Test AsyncJob sends status updates."""
    with (
        patch("hud.telemetry.async_context.settings") as mock_settings,
        patch("hud.telemetry.async_context.track_task") as mock_track,
        patch("hud.telemetry.async_context._print_job_url"),
        patch("hud.telemetry.async_context._print_job_complete_url"),
    ):
        mock_settings.telemetry_enabled = True
        mock_settings.api_key = "test-key"
        mock_settings.hud_telemetry_url = "https://test.com"

        async with async_job("Test"):
            pass

        # Should have called track_task twice (running and completed)
        assert mock_track.call_count == 2


@pytest.mark.asyncio
async def test_async_job_includes_dataset_link_in_status():
    """Test AsyncJob includes dataset_link in status updates."""
    with (
        patch("hud.telemetry.async_context.settings") as mock_settings,
        patch("hud.telemetry.async_context.track_task"),
        patch("hud.telemetry.async_context.make_request", new_callable=AsyncMock),
        patch("hud.telemetry.async_context._print_job_url"),
        patch("hud.telemetry.async_context._print_job_complete_url"),
    ):
        mock_settings.telemetry_enabled = True
        mock_settings.api_key = "test-key"
        mock_settings.hud_telemetry_url = "https://test.com"

        async with async_job("Test", dataset_link="test/dataset"):
            pass


@pytest.mark.asyncio
async def test_async_trace_non_root():
    """Test AsyncTrace with root=False."""
    with (
        patch("hud.telemetry.async_context.OtelTrace") as mock_otel,
        patch("hud.telemetry.async_context.track_task") as mock_track,
    ):
        mock_otel_instance = MagicMock()
        mock_otel.return_value = mock_otel_instance

        async with async_trace("Test", root=False):
            pass

        # Should not track status updates for non-root traces
        mock_track.assert_not_called()
