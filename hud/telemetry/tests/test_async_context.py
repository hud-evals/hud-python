from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from hud.telemetry.async_context import async_job, async_trace


@pytest.mark.asyncio
async def test_async_trace_basic():
    """Test basic AsyncTrace usage."""
    with (
        patch("hud.telemetry.async_context.OtelTrace") as mock_otel,
        patch("hud.telemetry.async_context._update_task_status_async", new_callable=AsyncMock),
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
        patch("hud.telemetry.async_context._update_task_status_async", new_callable=AsyncMock),
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
        patch("hud.telemetry.async_context._update_task_status_async", new_callable=AsyncMock),
    ):
        mock_otel_instance = MagicMock()
        mock_otel.return_value = mock_otel_instance

        async with async_trace("Test", task_id="task-456") as trace_obj:
            assert trace_obj.task_id == "task-456"


@pytest.mark.asyncio
async def test_async_trace_status_updates():
    """Test AsyncTrace sends and awaits status updates."""
    with (
        patch("hud.telemetry.async_context.settings") as mock_settings,
        patch("hud.telemetry.async_context.OtelTrace") as mock_otel,
        patch(
            "hud.telemetry.async_context._update_task_status_async",
            new_callable=AsyncMock,
        ) as mock_update,
        patch("hud.telemetry.utils.flush_telemetry", new_callable=AsyncMock),
    ):
        mock_settings.telemetry_enabled = True
        mock_settings.api_key = "test-key"
        mock_otel_instance = MagicMock()
        mock_otel.return_value = mock_otel_instance

        async with async_trace("Test", job_id=None):
            pass

        assert mock_update.call_count == 2


@pytest.mark.asyncio
async def test_async_trace_with_exception():
    """Test AsyncTrace handles exceptions."""
    with (
        patch("hud.telemetry.async_context.settings") as mock_settings,
        patch("hud.telemetry.async_context.OtelTrace") as mock_otel,
        patch(
            "hud.telemetry.async_context._update_task_status_async",
            new_callable=AsyncMock,
        ) as mock_update,
        patch("hud.telemetry.utils.flush_telemetry", new_callable=AsyncMock),
    ):
        mock_settings.telemetry_enabled = True
        mock_settings.api_key = "test-key"
        mock_otel_instance = MagicMock()
        mock_otel.return_value = mock_otel_instance

        with pytest.raises(ValueError):
            async with async_trace("Test"):
                raise ValueError("Test error")

        assert mock_update.call_count == 2
        final_call = mock_update.call_args_list[1]
        assert final_call[0][1] == "error"


@pytest.mark.asyncio
async def test_async_trace_non_root():
    """Test AsyncTrace with root=False."""
    with (
        patch("hud.telemetry.async_context.OtelTrace") as mock_otel,
        patch(
            "hud.telemetry.async_context._update_task_status_async",
            new_callable=AsyncMock,
        ) as mock_update,
    ):
        mock_otel_instance = MagicMock()
        mock_otel.return_value = mock_otel_instance

        async with async_trace("Test", root=False):
            pass

        mock_update.assert_not_called()


@pytest.mark.asyncio
async def test_async_trace_flushes_when_standalone():
    """Test AsyncTrace flushes spans when not part of a job."""
    with (
        patch("hud.telemetry.async_context.settings") as mock_settings,
        patch("hud.telemetry.async_context.OtelTrace") as mock_otel,
        patch(
            "hud.telemetry.async_context._update_task_status_async",
            new_callable=AsyncMock,
        ),
        patch("hud.telemetry.utils.flush_telemetry", new_callable=AsyncMock) as mock_flush,
    ):
        mock_settings.telemetry_enabled = True
        mock_settings.api_key = "test-key"
        mock_otel_instance = MagicMock()
        mock_otel.return_value = mock_otel_instance

        async with async_trace("Test", job_id=None):
            pass

        mock_flush.assert_called_once()


@pytest.mark.asyncio
async def test_async_trace_no_flush_when_in_job():
    """Test AsyncTrace doesn't flush when part of a job."""
    with (
        patch("hud.telemetry.async_context.settings") as mock_settings,
        patch("hud.telemetry.async_context.OtelTrace") as mock_otel,
        patch(
            "hud.telemetry.async_context._update_task_status_async",
            new_callable=AsyncMock,
        ),
        patch("hud.telemetry.utils.flush_telemetry", new_callable=AsyncMock) as mock_flush,
    ):
        mock_settings.telemetry_enabled = True
        mock_settings.api_key = "test-key"
        mock_otel_instance = MagicMock()
        mock_otel.return_value = mock_otel_instance

        async with async_trace("Test", job_id="job-123"):
            pass

        mock_flush.assert_not_called()


@pytest.mark.asyncio
async def test_async_job_basic():
    """Test basic AsyncJob usage."""
    with (
        patch("hud.telemetry.async_context.make_request", new_callable=AsyncMock),
        patch("hud.telemetry.utils.flush_telemetry", new_callable=AsyncMock),
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
        patch("hud.telemetry.async_context.make_request", new_callable=AsyncMock),
        patch("hud.telemetry.utils.flush_telemetry", new_callable=AsyncMock),
        patch("hud.telemetry.async_context._print_job_url"),
        patch("hud.telemetry.async_context._print_job_complete_url"),
    ):
        async with async_job("Test", metadata={"key": "value"}) as job_obj:
            assert job_obj.metadata == {"key": "value"}


@pytest.mark.asyncio
async def test_async_job_with_dataset_link():
    """Test AsyncJob with dataset_link."""
    with (
        patch("hud.telemetry.async_context.make_request", new_callable=AsyncMock),
        patch("hud.telemetry.utils.flush_telemetry", new_callable=AsyncMock),
        patch("hud.telemetry.async_context._print_job_url"),
        patch("hud.telemetry.async_context._print_job_complete_url"),
    ):
        async with async_job("Test", dataset_link="test/dataset") as job_obj:
            assert job_obj.dataset_link == "test/dataset"


@pytest.mark.asyncio
async def test_async_job_with_custom_job_id():
    """Test AsyncJob with custom job_id."""
    with (
        patch("hud.telemetry.async_context.make_request", new_callable=AsyncMock),
        patch("hud.telemetry.utils.flush_telemetry", new_callable=AsyncMock),
        patch("hud.telemetry.async_context._print_job_url"),
        patch("hud.telemetry.async_context._print_job_complete_url"),
    ):
        async with async_job("Test", job_id="custom-id") as job_obj:
            assert job_obj.id == "custom-id"


@pytest.mark.asyncio
async def test_async_job_with_exception():
    """Test AsyncJob handles exceptions."""
    with (
        patch("hud.telemetry.async_context.make_request", new_callable=AsyncMock),
        patch("hud.telemetry.utils.flush_telemetry", new_callable=AsyncMock),
        patch("hud.telemetry.async_context._print_job_url"),
        patch("hud.telemetry.async_context._print_job_complete_url") as mock_print,
    ):
        with pytest.raises(ValueError):
            async with async_job("Test"):
                raise ValueError("Job error")

        mock_print.assert_called_once()
        call_kwargs = mock_print.call_args[1]
        assert call_kwargs["error_occurred"] is True


@pytest.mark.asyncio
async def test_async_job_status_updates():
    """Test AsyncJob sends status updates."""
    with (
        patch("hud.telemetry.async_context.settings") as mock_settings,
        patch("hud.telemetry.async_context.make_request", new_callable=AsyncMock) as mock_request,
        patch("hud.telemetry.utils.flush_telemetry", new_callable=AsyncMock),
        patch("hud.telemetry.async_context._print_job_url"),
        patch("hud.telemetry.async_context._print_job_complete_url"),
    ):
        mock_settings.telemetry_enabled = True
        mock_settings.api_key = "test-key"
        mock_settings.hud_telemetry_url = "https://test.com"

        async with async_job("Test"):
            pass

        assert mock_request.call_count == 2


@pytest.mark.asyncio
async def test_async_job_flushes_on_exit():
    """Test AsyncJob flushes telemetry on exit."""
    with (
        patch("hud.telemetry.async_context.make_request", new_callable=AsyncMock),
        patch("hud.telemetry.utils.flush_telemetry", new_callable=AsyncMock) as mock_flush,
        patch("hud.telemetry.async_context._print_job_url"),
        patch("hud.telemetry.async_context._print_job_complete_url"),
    ):
        async with async_job("Test"):
            pass

        mock_flush.assert_called_once()


@pytest.mark.asyncio
async def test_async_trace_nested_contexts():
    """Test nested AsyncTrace contexts work correctly."""
    with (
        patch("hud.telemetry.async_context.OtelTrace") as mock_otel,
        patch("hud.telemetry.async_context._update_task_status_async", new_callable=AsyncMock),
    ):
        mock_otel_instance = MagicMock()
        mock_otel.return_value = mock_otel_instance

        async with async_trace("Outer Task") as outer:
            assert outer.name == "Outer Task"

            async with async_trace("Inner Task", root=False) as inner:
                assert inner.name == "Inner Task"
                assert inner.id != outer.id


@pytest.mark.asyncio
async def test_async_trace_concurrent_traces():
    """Test multiple concurrent AsyncTrace contexts."""
    import asyncio

    with (
        patch("hud.telemetry.async_context.OtelTrace") as mock_otel,
        patch("hud.telemetry.async_context._update_task_status_async", new_callable=AsyncMock),
    ):
        mock_otel_instance = MagicMock()
        mock_otel.return_value = mock_otel_instance

        async def run_trace(name: str):
            async with async_trace(name) as trace_obj:
                await asyncio.sleep(0.01)
                return trace_obj.id

        # Run multiple traces concurrently
        ids = await asyncio.gather(
            run_trace("Trace 1"),
            run_trace("Trace 2"),
            run_trace("Trace 3"),
        )

        # All traces should have unique IDs
        assert len(set(ids)) == 3


@pytest.mark.asyncio
async def test_async_trace_with_attrs():
    """Test AsyncTrace with attrs parameter passed to OtelTrace."""
    with (
        patch("hud.telemetry.async_context.OtelTrace") as mock_otel,
        patch("hud.telemetry.async_context._update_task_status_async", new_callable=AsyncMock),
    ):
        mock_otel_instance = MagicMock()
        mock_otel.return_value = mock_otel_instance

        attrs = {"key": "value", "count": 42}
        async with async_trace("Test", attrs=attrs):
            # attrs are passed to OtelTrace, not exposed on Trace object
            mock_otel.assert_called_once()
            call_kwargs = mock_otel.call_args[1]
            assert call_kwargs["attributes"] == attrs


@pytest.mark.asyncio
async def test_async_trace_exception_types():
    """Test AsyncTrace handles different exception types correctly."""
    with (
        patch("hud.telemetry.async_context.settings") as mock_settings,
        patch("hud.telemetry.async_context.OtelTrace") as mock_otel,
        patch(
            "hud.telemetry.async_context._update_task_status_async",
            new_callable=AsyncMock,
        ) as mock_update,
        patch("hud.telemetry.utils.flush_telemetry", new_callable=AsyncMock),
    ):
        mock_settings.telemetry_enabled = True
        mock_settings.api_key = "test-key"
        mock_otel_instance = MagicMock()
        mock_otel.return_value = mock_otel_instance

        # Test KeyError
        with pytest.raises(KeyError):
            async with async_trace("Test"):
                raise KeyError("Missing key")

        # Test RuntimeError
        with pytest.raises(RuntimeError):
            async with async_trace("Test"):
                raise RuntimeError("Runtime issue")

        # Both should have resulted in error status
        assert mock_update.call_count >= 4  # 2 calls per trace


@pytest.mark.asyncio
async def test_async_job_nested_with_trace():
    """Test AsyncJob with nested AsyncTrace contexts."""
    with (
        patch("hud.telemetry.async_context.make_request", new_callable=AsyncMock),
        patch("hud.telemetry.utils.flush_telemetry", new_callable=AsyncMock),
        patch("hud.telemetry.async_context._print_job_url"),
        patch("hud.telemetry.async_context._print_job_complete_url"),
        patch("hud.telemetry.async_context.OtelTrace") as mock_otel,
        patch("hud.telemetry.async_context._update_task_status_async", new_callable=AsyncMock),
    ):
        mock_otel_instance = MagicMock()
        mock_otel.return_value = mock_otel_instance

        async with async_job("Test Job") as job_obj:
            async with async_trace("Task 1", job_id=job_obj.id) as trace1:
                assert trace1.job_id == job_obj.id

            async with async_trace("Task 2", job_id=job_obj.id) as trace2:
                assert trace2.job_id == job_obj.id
                assert trace2.id != trace1.id


@pytest.mark.asyncio
async def test_async_job_concurrent_jobs():
    """Test multiple concurrent AsyncJob contexts."""
    import asyncio

    with (
        patch("hud.telemetry.async_context.make_request", new_callable=AsyncMock),
        patch("hud.telemetry.utils.flush_telemetry", new_callable=AsyncMock),
        patch("hud.telemetry.async_context._print_job_url"),
        patch("hud.telemetry.async_context._print_job_complete_url"),
    ):

        async def run_job(name: str):
            async with async_job(name) as job_obj:
                await asyncio.sleep(0.01)
                return job_obj.id

        # Run multiple jobs concurrently
        ids = await asyncio.gather(
            run_job("Job 1"),
            run_job("Job 2"),
            run_job("Job 3"),
        )

        # All jobs should have unique IDs
        assert len(set(ids)) == 3


@pytest.mark.asyncio
async def test_async_job_with_multiple_exceptions():
    """Test AsyncJob handles multiple exceptions in nested contexts."""
    with (
        patch("hud.telemetry.async_context.make_request", new_callable=AsyncMock),
        patch("hud.telemetry.utils.flush_telemetry", new_callable=AsyncMock),
        patch("hud.telemetry.async_context._print_job_url"),
        patch("hud.telemetry.async_context._print_job_complete_url") as mock_print,
    ):
        with pytest.raises(ValueError):
            async with async_job("Test"):
                try:
                    raise RuntimeError("First error")
                except RuntimeError:
                    # Catch and re-raise different error
                    raise ValueError("Second error")

        mock_print.assert_called_once()
        call_kwargs = mock_print.call_args[1]
        assert call_kwargs["error_occurred"] is True


@pytest.mark.asyncio
async def test_async_trace_telemetry_disabled():
    """Test AsyncTrace behavior when telemetry is disabled."""
    with (
        patch("hud.telemetry.async_context.settings") as mock_settings,
        patch("hud.telemetry.async_context.OtelTrace") as mock_otel,
        patch(
            "hud.telemetry.async_context._update_task_status_async",
            new_callable=AsyncMock,
        ),
    ):
        mock_settings.telemetry_enabled = False
        mock_otel_instance = MagicMock()
        mock_otel.return_value = mock_otel_instance

        async with async_trace("Test"):
            pass

        # Should still create trace but not send updates
        mock_otel.assert_called_once()
        # Status updates might still be called depending on implementation


@pytest.mark.asyncio
async def test_async_job_empty_metadata():
    """Test AsyncJob with empty metadata dict."""
    with (
        patch("hud.telemetry.async_context.make_request", new_callable=AsyncMock),
        patch("hud.telemetry.utils.flush_telemetry", new_callable=AsyncMock),
        patch("hud.telemetry.async_context._print_job_url"),
        patch("hud.telemetry.async_context._print_job_complete_url"),
    ):
        async with async_job("Test", metadata={}) as job_obj:
            assert job_obj.metadata == {}


@pytest.mark.asyncio
async def test_async_trace_with_all_parameters():
    """Test AsyncTrace with all parameters specified."""
    with (
        patch("hud.telemetry.async_context.OtelTrace") as mock_otel,
        patch("hud.telemetry.async_context._update_task_status_async", new_callable=AsyncMock),
    ):
        mock_otel_instance = MagicMock()
        mock_otel.return_value = mock_otel_instance

        async with async_trace(
            "Test",
            job_id="job-123",
            task_id="task-456",
            group_id="group-789",
            attrs={"key": "value"},
            root=True,
        ) as trace_obj:
            assert trace_obj.name == "Test"
            assert trace_obj.job_id == "job-123"
            assert trace_obj.task_id == "task-456"
            assert trace_obj.group_id == "group-789"
            # Verify attrs were passed to OtelTrace
            call_kwargs = mock_otel.call_args[1]
            assert call_kwargs["attributes"] == {"key": "value"}


@pytest.mark.asyncio
async def test_async_trace_with_group_id():
    """Test AsyncTrace with group_id parameter."""
    with (
        patch("hud.telemetry.async_context.OtelTrace") as mock_otel,
        patch("hud.telemetry.async_context._update_task_status_async", new_callable=AsyncMock),
    ):
        mock_otel_instance = MagicMock()
        mock_otel.return_value = mock_otel_instance

        async with async_trace("Test", group_id="group-999") as trace_obj:
            assert trace_obj.group_id == "group-999"
