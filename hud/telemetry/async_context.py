"""Async context managers for HUD telemetry.

Provides async versions of trace and job context managers for high-concurrency
async code. These prevent event loop blocking by using async I/O operations.

Usage:
    >>> import hud
    >>> async with hud.async_job("My Job") as job:
    ...     async with hud.async_trace("Task", job_id=job.id) as trace:
    ...         await do_work()

When to use:
    - High-concurrency scenarios (200+ parallel tasks)
    - Custom async evaluation loops
    - Async frameworks with HUD telemetry integration

When NOT to use:
    - Typical scripts/notebooks → use `hud.trace()` and `hud.job()`
    - Low concurrency (< 30 tasks) → standard context managers are fine
    - Synchronous code → must use `hud.trace()` and `hud.job()`

Note:
    The `run_dataset()` function automatically uses these async context managers
    internally, so most users don't need to use them directly.
"""

from __future__ import annotations

import logging
import uuid
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from types import TracebackType

from hud.otel import configure_telemetry
from hud.otel.context import (
    _print_trace_complete_url,
    _print_trace_url,
    _update_task_status_async,
)
from hud.otel.context import (
    trace as OtelTrace,
)
from hud.settings import settings
from hud.shared import make_request
from hud.telemetry.job import Job, _print_job_complete_url, _print_job_url
from hud.telemetry.trace import Trace
from hud.utils.task_tracking import track_task

logger = logging.getLogger(__name__)

# Module exports
__all__ = ["AsyncJob", "AsyncTrace", "async_job", "async_trace"]

# Global state for current job
_current_job: Job | None = None


class AsyncTrace:
    """Async context manager for HUD trace tracking.

    This is the async equivalent of `hud.trace()`, designed for use in
    high-concurrency async contexts. It tracks task execution with automatic
    status updates that don't block the event loop.

    The context manager:
    - Creates a unique task_run_id for telemetry correlation
    - Sends async status updates ("running", "completed", "error")
    - Integrates with OpenTelemetry for span collection
    - Tracks all async operations for proper cleanup

    Use `async_trace()` helper function instead of instantiating directly.
    """

    def __init__(
        self,
        name: str = "Test task from hud",
        *,
        root: bool = True,
        attrs: dict[str, Any] | None = None,
        job_id: str | None = None,
        task_id: str | None = None,
        group_id: str | None = None,
    ) -> None:
        self.name = name
        self.root = root
        self.attrs = attrs or {}
        self.job_id = job_id
        self.task_id = task_id
        self.group_id = group_id
        self.task_run_id = str(uuid.uuid4())
        self.trace_obj = Trace(self.task_run_id, name, job_id, task_id, group_id)
        self._otel_trace = None

    async def __aenter__(self) -> Trace:
        """Enter the async trace context."""
        # Ensure telemetry is configured
        configure_telemetry()

        # Start the OpenTelemetry span
        self._otel_trace = OtelTrace(
            self.task_run_id,
            is_root=self.root,
            span_name=self.name,
            attributes=self.attrs,
            job_id=self.job_id,
            task_id=self.task_id,
            group_id=self.group_id,
        )
        self._otel_trace.__enter__()

        # Send async status update if this is a root trace
        if self.root and settings.telemetry_enabled and settings.api_key:
            track_task(
                _update_task_status_async(
                    self.task_run_id,
                    "running",
                    job_id=self.job_id,
                    trace_name=self.name,
                    task_id=self.task_id,
                    group_id=self.group_id,
                ),
                name=f"trace-status-{self.task_run_id[:8]}",
            )

            # Print trace URL if not part of a job
            if not self.job_id:
                _print_trace_url(self.task_run_id)

        logger.debug("Started trace: %s (%s)", self.name, self.task_run_id)
        return self.trace_obj

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Exit the async trace context."""
        # Send async status update if this is a root trace
        if self.root and settings.telemetry_enabled and settings.api_key:
            status = "error" if exc_type else "completed"

            track_task(
                _update_task_status_async(
                    self.task_run_id,
                    status,
                    job_id=self.job_id,
                    error_message=str(exc_val) if exc_val else None,
                    trace_name=self.name,
                    task_id=self.task_id,
                    group_id=self.group_id,
                ),
                name=f"trace-status-{self.task_run_id[:8]}-{status}",
            )

            # Print completion message if not part of a job
            if not self.job_id:
                _print_trace_complete_url(self.task_run_id, error_occurred=bool(exc_type))

        # Close the OpenTelemetry span
        if self._otel_trace:
            self._otel_trace.__exit__(exc_type, exc_val, exc_tb)

        logger.debug("Ended trace: %s (%s)", self.name, self.task_run_id)


class AsyncJob:
    """Async context manager for HUD job tracking.

    This is the async equivalent of `hud.job()`, designed for grouping
    related tasks in high-concurrency async contexts. It manages job
    status updates without blocking the event loop.

    The context manager:
    - Creates or uses a provided job_id
    - Sends async status updates ("running", "completed", "failed")
    - Associates all child traces with this job
    - Tracks async operations for proper cleanup

    Use `async_job()` helper function instead of instantiating directly.
    """

    def __init__(
        self,
        name: str,
        metadata: dict[str, Any] | None = None,
        job_id: str | None = None,
        dataset_link: str | None = None,
    ) -> None:
        self.job_id = job_id or str(uuid.uuid4())
        self.job = Job(self.job_id, name, metadata, dataset_link)

    async def __aenter__(self) -> Job:
        """Enter the async job context."""
        global _current_job

        # Save previous job and set this as current
        self._old_job = _current_job
        _current_job = self.job

        # Send async status update
        if settings.telemetry_enabled:
            payload = {
                "name": self.job.name,
                "status": "running",
                "metadata": self.job.metadata,
            }
            if self.job.dataset_link:
                payload["dataset_link"] = self.job.dataset_link

            track_task(
                make_request(
                    method="POST",
                    url=f"{settings.hud_telemetry_url}/jobs/{self.job.id}/status",
                    json=payload,
                    api_key=settings.api_key,
                ),
                name=f"job-status-{self.job.id[:8]}-running",
            )

        _print_job_url(self.job.id, self.job.name)
        logger.debug("Started job: %s (%s)", self.job.name, self.job.id)
        return self.job

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Exit the async job context."""
        global _current_job

        # Send async status update
        if settings.telemetry_enabled:
            status = "failed" if exc_type else "completed"
            payload = {
                "name": self.job.name,
                "status": status,
                "metadata": self.job.metadata,
            }
            if self.job.dataset_link:
                payload["dataset_link"] = self.job.dataset_link

            track_task(
                make_request(
                    method="POST",
                    url=f"{settings.hud_telemetry_url}/jobs/{self.job.id}/status",
                    json=payload,
                    api_key=settings.api_key,
                ),
                name=f"job-status-{self.job.id[:8]}-{status}",
            )

        _print_job_complete_url(self.job.id, self.job.name, error_occurred=bool(exc_type))

        # Restore previous job
        _current_job = self._old_job

        logger.debug("Ended job: %s (%s)", self.job.name, self.job.id)


def async_trace(
    name: str = "Test task from hud",
    *,
    root: bool = True,
    attrs: dict[str, Any] | None = None,
    job_id: str | None = None,
    task_id: str | None = None,
    group_id: str | None = None,
) -> AsyncTrace:
    """Create an async trace context for telemetry tracking.

    This is the async equivalent of `hud.trace()` for use in high-concurrency
    async contexts. Status updates are sent asynchronously and tracked to ensure
    completion before shutdown.

    Args:
        name: Descriptive name for this trace/task
        root: Whether this is a root trace (updates task status)
        attrs: Additional attributes to attach to the trace
        job_id: Optional job ID to associate with this trace
        task_id: Optional task ID for custom task identifiers
        group_id: Optional group ID to associate with this trace

    Returns:
        AsyncTrace context manager

    Example:
        >>> import hud
        >>> async with hud.async_trace("Process Data") as trace:
        ...     result = await process_async()
        ...     await trace.log({"items_processed": len(result)})

    Note:
        Most users should use `hud.trace()` which works fine for typical usage.
        Use this async version only in high-concurrency scenarios (200+ parallel
        tasks) or when writing custom async evaluation frameworks.
    """
    return AsyncTrace(
        name,
        root=root,
        attrs=attrs,
        job_id=job_id,
        task_id=task_id,
        group_id=group_id if group_id else str(uuid.uuid4()),
    )


def async_job(
    name: str,
    metadata: dict[str, Any] | None = None,
    job_id: str | None = None,
    dataset_link: str | None = None,
) -> AsyncJob:
    """Create an async job context for grouping related tasks.

    This is the async equivalent of `hud.job()` for use in high-concurrency
    async contexts. Job status updates are sent asynchronously and tracked
    to ensure completion before shutdown.

    Args:
        name: Human-readable job name
        metadata: Optional metadata dictionary
        job_id: Optional job ID (auto-generated if not provided)
        dataset_link: Optional HuggingFace dataset identifier

    Returns:
        AsyncJob context manager

    Example:
        >>> import hud
        >>> async with hud.async_job("Batch Processing") as job:
        ...     for item in items:
        ...         async with hud.async_trace(f"Process {item.id}", job_id=job.id):
        ...             await process(item)

    Note:
        Most users should use `hud.job()` which works fine for typical usage.
        Use this async version only in high-concurrency scenarios (200+ parallel
        tasks) or when writing custom async evaluation frameworks.
    """
    return AsyncJob(name, metadata=metadata, job_id=job_id, dataset_link=dataset_link)
