from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock, patch

import pytest

from hud.telemetry.job import (
    Job,
    _print_job_complete_url,
    _print_job_url,
    create_job,
    get_current_job,
    job,
    job_decorator,
)


def test_job_initialization():
    """Test Job initialization with all parameters."""
    job_obj = Job(
        job_id="test-id",
        name="Test Job",
        metadata={"key": "value"},
        dataset_link="test/dataset",
    )

    assert job_obj.id == "test-id"
    assert job_obj.name == "Test Job"
    assert job_obj.metadata == {"key": "value"}
    assert job_obj.dataset_link == "test/dataset"
    assert job_obj.status == "created"
    assert isinstance(job_obj.created_at, datetime)
    assert job_obj.tasks == []


def test_job_initialization_minimal():
    """Test Job initialization with minimal parameters."""
    job_obj = Job(job_id="test-id", name="Test")

    assert job_obj.metadata == {}
    assert job_obj.dataset_link is None


def test_job_add_task():
    """Test adding tasks to a job."""
    job_obj = Job(job_id="test-id", name="Test")

    job_obj.add_task("task1")
    job_obj.add_task("task2")

    assert job_obj.tasks == ["task1", "task2"]


@pytest.mark.asyncio
async def test_job_update_status_async():
    """Test async status update."""
    job_obj = Job(job_id="test-id", name="Test")

    with (
        patch("hud.telemetry.job.settings") as mock_settings,
        patch("hud.telemetry.job.make_request", new_callable=AsyncMock) as mock_request,
    ):
        mock_settings.telemetry_enabled = True
        mock_settings.api_key = "test_key"
        mock_settings.hud_telemetry_url = "https://test.com"

        await job_obj.update_status("running")

        assert job_obj.status == "running"
        mock_request.assert_called_once()
        call_kwargs = mock_request.call_args[1]
        assert call_kwargs["method"] == "POST"
        assert "test-id" in call_kwargs["url"]
        assert call_kwargs["json"]["status"] == "running"


@pytest.mark.asyncio
async def test_job_update_status_async_with_dataset():
    """Test async status update includes dataset link."""
    job_obj = Job(job_id="test-id", name="Test", dataset_link="test/dataset")

    with (
        patch("hud.telemetry.job.settings") as mock_settings,
        patch("hud.telemetry.job.make_request", new_callable=AsyncMock) as mock_request,
    ):
        mock_settings.telemetry_enabled = True
        mock_settings.api_key = "test_key"
        mock_settings.hud_telemetry_url = "https://test.com"

        await job_obj.update_status("running")

        call_kwargs = mock_request.call_args[1]
        assert call_kwargs["json"]["dataset_link"] == "test/dataset"


@pytest.mark.asyncio
async def test_job_update_status_async_telemetry_disabled():
    """Test async status update when telemetry is disabled."""
    job_obj = Job(job_id="test-id", name="Test")

    with (
        patch("hud.telemetry.job.settings") as mock_settings,
        patch("hud.telemetry.job.make_request", new_callable=AsyncMock) as mock_request,
    ):
        mock_settings.telemetry_enabled = False

        await job_obj.update_status("running")

        assert job_obj.status == "running"
        mock_request.assert_not_called()


@pytest.mark.asyncio
async def test_job_update_status_async_error():
    """Test async status update handles errors gracefully."""
    job_obj = Job(job_id="test-id", name="Test")

    with (
        patch("hud.telemetry.job.settings") as mock_settings,
        patch("hud.telemetry.job.make_request", new_callable=AsyncMock) as mock_request,
    ):
        mock_settings.telemetry_enabled = True
        mock_settings.api_key = "test_key"
        mock_settings.hud_telemetry_url = "https://test.com"
        mock_request.side_effect = Exception("Network error")

        # Should not raise
        await job_obj.update_status("running")
        assert job_obj.status == "running"


def test_job_update_status_sync():
    """Test sync status update."""
    job_obj = Job(job_id="test-id", name="Test")

    with (
        patch("hud.telemetry.job.settings") as mock_settings,
        patch("hud.telemetry.job.make_request_sync") as mock_request,
    ):
        mock_settings.telemetry_enabled = True
        mock_settings.api_key = "test_key"
        mock_settings.hud_telemetry_url = "https://test.com"

        job_obj.update_status_sync("completed")

        assert job_obj.status == "completed"
        mock_request.assert_called_once()


def test_job_update_status_sync_with_dataset():
    """Test sync status update includes dataset link."""
    job_obj = Job(job_id="test-id", name="Test", dataset_link="test/dataset")

    with (
        patch("hud.telemetry.job.settings") as mock_settings,
        patch("hud.telemetry.job.make_request_sync") as mock_request,
    ):
        mock_settings.telemetry_enabled = True
        mock_settings.api_key = "test_key"
        mock_settings.hud_telemetry_url = "https://test.com"

        job_obj.update_status_sync("completed")

        call_kwargs = mock_request.call_args[1]
        assert call_kwargs["json"]["dataset_link"] == "test/dataset"


def test_job_update_status_sync_telemetry_disabled():
    """Test sync status update when telemetry is disabled."""
    job_obj = Job(job_id="test-id", name="Test")

    with (
        patch("hud.telemetry.job.settings") as mock_settings,
        patch("hud.telemetry.job.make_request_sync") as mock_request,
    ):
        mock_settings.telemetry_enabled = False

        job_obj.update_status_sync("completed")

        mock_request.assert_not_called()


def test_job_update_status_sync_error():
    """Test sync status update handles errors gracefully."""
    job_obj = Job(job_id="test-id", name="Test")

    with (
        patch("hud.telemetry.job.settings") as mock_settings,
        patch("hud.telemetry.job.make_request_sync") as mock_request,
    ):
        mock_settings.telemetry_enabled = True
        mock_settings.api_key = "test_key"
        mock_settings.hud_telemetry_url = "https://test.com"
        mock_request.side_effect = Exception("Network error")

        # Should not raise
        job_obj.update_status_sync("completed")


@pytest.mark.asyncio
async def test_job_log():
    """Test async log method."""
    job_obj = Job(job_id="test-id", name="Test")

    with (
        patch("hud.telemetry.job.settings") as mock_settings,
        patch("hud.telemetry.job.make_request", new_callable=AsyncMock) as mock_request,
    ):
        mock_settings.telemetry_enabled = True
        mock_settings.api_key = "test_key"
        mock_settings.hud_telemetry_url = "https://test.com"

        await job_obj.log({"loss": 0.5, "accuracy": 0.95})

        mock_request.assert_called_once()
        call_kwargs = mock_request.call_args[1]
        assert call_kwargs["json"]["metrics"] == {"loss": 0.5, "accuracy": 0.95}
        assert "timestamp" in call_kwargs["json"]


@pytest.mark.asyncio
async def test_job_log_telemetry_disabled():
    """Test async log when telemetry is disabled."""
    job_obj = Job(job_id="test-id", name="Test")

    with (
        patch("hud.telemetry.job.settings") as mock_settings,
        patch("hud.telemetry.job.make_request", new_callable=AsyncMock) as mock_request,
    ):
        mock_settings.telemetry_enabled = False

        await job_obj.log({"loss": 0.5})

        mock_request.assert_not_called()


@pytest.mark.asyncio
async def test_job_log_error():
    """Test async log handles errors gracefully."""
    job_obj = Job(job_id="test-id", name="Test")

    with (
        patch("hud.telemetry.job.settings") as mock_settings,
        patch("hud.telemetry.job.make_request", new_callable=AsyncMock) as mock_request,
    ):
        mock_settings.telemetry_enabled = True
        mock_settings.api_key = "test_key"
        mock_settings.hud_telemetry_url = "https://test.com"
        mock_request.side_effect = Exception("Network error")

        # Should not raise
        await job_obj.log({"loss": 0.5})


def test_job_log_sync():
    """Test sync log method."""
    job_obj = Job(job_id="test-id", name="Test")

    with (
        patch("hud.telemetry.job.settings") as mock_settings,
        patch("hud.telemetry.job.make_request_sync") as mock_request,
    ):
        mock_settings.telemetry_enabled = True
        mock_settings.api_key = "test_key"
        mock_settings.hud_telemetry_url = "https://test.com"

        job_obj.log_sync({"loss": 0.5, "accuracy": 0.95})

        mock_request.assert_called_once()
        call_kwargs = mock_request.call_args[1]
        assert call_kwargs["json"]["metrics"] == {"loss": 0.5, "accuracy": 0.95}


def test_job_log_sync_telemetry_disabled():
    """Test sync log when telemetry is disabled."""
    job_obj = Job(job_id="test-id", name="Test")

    with (
        patch("hud.telemetry.job.settings") as mock_settings,
        patch("hud.telemetry.job.make_request_sync") as mock_request,
    ):
        mock_settings.telemetry_enabled = False

        job_obj.log_sync({"loss": 0.5})

        mock_request.assert_not_called()


def test_job_log_sync_error():
    """Test sync log handles errors gracefully."""
    job_obj = Job(job_id="test-id", name="Test")

    with (
        patch("hud.telemetry.job.settings") as mock_settings,
        patch("hud.telemetry.job.make_request_sync") as mock_request,
    ):
        mock_settings.telemetry_enabled = True
        mock_settings.api_key = "test_key"
        mock_settings.hud_telemetry_url = "https://test.com"
        mock_request.side_effect = Exception("Network error")

        # Should not raise
        job_obj.log_sync({"loss": 0.5})


def test_job_repr():
    """Test Job __repr__."""
    job_obj = Job(job_id="test-id", name="Test Job")
    job_obj.status = "running"

    repr_str = repr(job_obj)
    assert "test-id" in repr_str
    assert "Test Job" in repr_str
    assert "running" in repr_str


def test_print_job_url_enabled():
    """Test _print_job_url when telemetry is enabled."""
    with (
        patch("hud.telemetry.job.settings") as mock_settings,
        patch("builtins.print") as mock_print,
    ):
        mock_settings.telemetry_enabled = True
        mock_settings.api_key = "test_key"

        _print_job_url("job-123", "My Job")

        # Should print multiple lines (box)
        assert mock_print.call_count > 0


def test_print_job_url_disabled():
    """Test _print_job_url when telemetry is disabled."""
    with (
        patch("hud.telemetry.job.settings") as mock_settings,
        patch("builtins.print") as mock_print,
    ):
        mock_settings.telemetry_enabled = False

        _print_job_url("job-123", "My Job")

        mock_print.assert_not_called()


def test_print_job_url_no_api_key():
    """Test _print_job_url when no API key is set."""
    with (
        patch("hud.telemetry.job.settings") as mock_settings,
        patch("builtins.print") as mock_print,
    ):
        mock_settings.telemetry_enabled = True
        mock_settings.api_key = None

        _print_job_url("job-123", "My Job")

        mock_print.assert_not_called()


def test_print_job_complete_url_success():
    """Test _print_job_complete_url for successful completion."""
    with (
        patch("hud.telemetry.job.settings") as mock_settings,
        patch("builtins.print") as mock_print,
    ):
        mock_settings.telemetry_enabled = True
        mock_settings.api_key = "test_key"

        _print_job_complete_url("job-123", "My Job", error_occurred=False)

        mock_print.assert_called_once()
        call_str = str(mock_print.call_args)
        assert "complete" in call_str.lower() or "✓" in call_str


def test_print_job_complete_url_failure():
    """Test _print_job_complete_url for failed completion."""
    with (
        patch("hud.telemetry.job.settings") as mock_settings,
        patch("builtins.print") as mock_print,
    ):
        mock_settings.telemetry_enabled = True
        mock_settings.api_key = "test_key"

        _print_job_complete_url("job-123", "My Job", error_occurred=True)

        mock_print.assert_called_once()
        call_str = str(mock_print.call_args)
        assert "fail" in call_str.lower() or "✗" in call_str


def test_print_job_complete_url_disabled():
    """Test _print_job_complete_url when telemetry is disabled."""
    with (
        patch("hud.telemetry.job.settings") as mock_settings,
        patch("builtins.print") as mock_print,
    ):
        mock_settings.telemetry_enabled = False

        _print_job_complete_url("job-123", "My Job")

        mock_print.assert_not_called()


def test_get_current_job_none():
    """Test get_current_job when no job is active."""
    result = get_current_job()
    assert result is None


def test_job_context_manager():
    """Test job context manager."""
    with (
        patch("hud.telemetry.job.settings") as mock_settings,
        patch("builtins.print"),
    ):
        mock_settings.telemetry_enabled = True
        mock_settings.api_key = "test_key"

        with job("Test Job", {"key": "value"}) as job_obj:
            assert job_obj.name == "Test Job"
            assert job_obj.metadata == {"key": "value"}
            assert get_current_job() == job_obj

        # After context, job should be cleared
        assert get_current_job() is None


def test_job_context_manager_with_job_id():
    """Test job context manager with explicit job_id."""
    with (
        patch("hud.telemetry.job.settings") as mock_settings,
        patch("builtins.print"),
    ):
        mock_settings.telemetry_enabled = True
        mock_settings.api_key = "test_key"

        with job("Test", job_id="my-custom-id") as job_obj:
            assert job_obj.id == "my-custom-id"


def test_job_context_manager_with_dataset_link():
    """Test job context manager with dataset link."""
    with (
        patch("hud.telemetry.job.settings") as mock_settings,
        patch("builtins.print"),
    ):
        mock_settings.telemetry_enabled = True
        mock_settings.api_key = "test_key"

        with job("Test", dataset_link="test/dataset") as job_obj:
            assert job_obj.dataset_link == "test/dataset"


def test_job_context_manager_exception():
    """Test job context manager handles exceptions."""
    with (
        patch("hud.telemetry.job.settings") as mock_settings,
        patch("builtins.print"),
    ):
        mock_settings.telemetry_enabled = True
        mock_settings.api_key = "test_key"

        with pytest.raises(ValueError), job("Test"):
            raise ValueError("Test error")

        # Job should be cleared even after exception
        assert get_current_job() is None


def test_create_job():
    """Test create_job function."""
    job_obj = create_job("Test Job", {"key": "value"}, dataset_link="test/dataset")

    assert job_obj.name == "Test Job"
    assert job_obj.metadata == {"key": "value"}
    assert job_obj.dataset_link == "test/dataset"
    assert job_obj.id  # Should have an auto-generated ID


def test_create_job_with_job_id():
    """Test create_job with explicit job_id."""
    job_obj = create_job("Test", job_id="custom-id")

    assert job_obj.id == "custom-id"


@pytest.mark.asyncio
async def test_job_decorator_async():
    """Test job_decorator on async function."""
    with (
        patch("hud.telemetry.job.settings") as mock_settings,
        patch("builtins.print"),
    ):
        mock_settings.telemetry_enabled = True
        mock_settings.api_key = "test_key"

        @job_decorator("test_job", model="gpt-4")
        async def test_func(x: int) -> int:
            return x * 2

        result = await test_func(5)
        assert result == 10


def test_job_decorator_sync():
    """Test job_decorator on sync function."""
    with (
        patch("hud.telemetry.job.settings") as mock_settings,
        patch("builtins.print"),
    ):
        mock_settings.telemetry_enabled = True
        mock_settings.api_key = "test_key"

        @job_decorator("test_job", model="gpt-4")
        def test_func(x: int) -> int:
            return x * 2

        result = test_func(5)
        assert result == 10


@pytest.mark.asyncio
async def test_job_decorator_async_default_name():
    """Test job_decorator uses function name as default."""
    with (
        patch("hud.telemetry.job.settings") as mock_settings,
        patch("builtins.print"),
    ):
        mock_settings.telemetry_enabled = True
        mock_settings.api_key = "test_key"

        @job_decorator()
        async def my_function():
            return "success"

        result = await my_function()
        assert result == "success"


def test_job_decorator_sync_default_name():
    """Test job_decorator sync uses function name as default."""
    with (
        patch("hud.telemetry.job.settings") as mock_settings,
        patch("builtins.print"),
    ):
        mock_settings.telemetry_enabled = True
        mock_settings.api_key = "test_key"

        @job_decorator()
        def my_function():
            return "success"

        result = my_function()
        assert result == "success"
