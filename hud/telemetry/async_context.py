"""Async context managers for HUD telemetry.

Provides async-native trace and job context managers for async code.

Usage:
    >>> import hud
    >>> async with hud.async_trace("Task"):
    ...     await agent.run(task)
    >>> async with hud.async_job("Evaluation") as job:
    ...     async with hud.async_trace("Task", job_id=job.id):
    ...         await agent.run(task)

Telemetry is fully automatic - status updates are awaited and spans are
flushed on context exit. No manual cleanup required.
"""

from __future__ import annotations

import logging
import traceback
import uuid
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from types import TracebackType

from hud.otel import configure_telemetry
from hud.otel.context import (
    _update_task_status_async,
)
from hud.otel.context import (
    trace as OtelTrace,
)
from hud.settings import settings
from hud.shared import make_request
from hud.telemetry.job import Job, _print_job_complete_url, _print_job_url
from hud.telemetry.trace import Trace

logger = logging.getLogger(__name__)

# Module exports
__all__ = ["AsyncJob", "AsyncTrace", "async_job", "async_trace"]

# Global state for current job
_current_job: Job | None = None


class AsyncTrace:
    """Async context manager for HUD trace tracking.

    This is the async equivalent of `hud.trace()`, designed for use in
    high-concurrency async contexts. It tracks task execution with automatic
    status updates.

    The context manager:
    - Creates a unique task_run_id for telemetry correlation
    - Sends and AWAITS status updates ("running" → "completed"/"error")
    - Integrates with OpenTelemetry for span collection
    - Ensures status is updated before exiting the context

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
        trace_id: str | None = None,
    ) -> None:
        self.name = name
        self.root = root
        self.attrs = attrs or {}
        self.job_id = job_id
        self.task_id = task_id
        self.group_id = group_id
        self.task_run_id = trace_id if trace_id else str(uuid.uuid4())
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

        # Update trace status to "running"
        if self.root and settings.telemetry_enabled and settings.api_key:
            await _update_task_status_async(
                self.task_run_id,
                "running",
                job_id=self.job_id,
                trace_name=self.name,
                task_id=self.task_id,
                group_id=self.group_id,
            )

        logger.debug("Started trace: %s (%s)", self.name, self.task_run_id)
        return self.trace_obj

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Exit the async trace context."""
        # Close the OpenTelemetry span
        if self._otel_trace:
            self._otel_trace.__exit__(exc_type, exc_val, exc_tb)

        # Update trace status to "completed" or "error"
        if self.root and settings.telemetry_enabled and settings.api_key:
            status = "error" if exc_type else "completed"
            error_msg = None
            if exc_type is not None:
                error_msg = "".join(traceback.format_exception(exc_type, exc_val, exc_tb))

            try:
                await _update_task_status_async(
                    self.task_run_id,
                    status,
                    job_id=self.job_id,
                    error_message=error_msg,
                    trace_name=self.name,
                    task_id=self.task_id,
                    group_id=self.group_id,
                )
            except Exception as e:
                logger.warning("Failed to update trace status: %s", e)

        # Flush spans for standalone traces (not part of a job)
        if not self.job_id and self.root:
            from hud.telemetry.utils import flush_telemetry

            await flush_telemetry()

        logger.debug("Ended trace: %s (%s)", self.name, self.task_run_id)


class AsyncJob:
    """Async context manager for HUD job tracking.

    This is the async equivalent of `hud.job()`, designed for grouping
    related tasks in high-concurrency async contexts.

    The context manager:
    - Creates or uses a provided job_id
    - Sends and AWAITS status updates ("running" → "completed"/"failed")
    - Associates all child traces with this job
    - Ensures status is updated before exiting the context

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

        # Update job status to "running"
        if settings.telemetry_enabled:
            payload = {
                "name": self.job.name,
                "status": "running",
                "metadata": self.job.metadata,
            }
            if self.job.dataset_link:
                payload["dataset_link"] = self.job.dataset_link

            try:
                await make_request(
                    method="POST",
                    url=f"{settings.hud_telemetry_url}/jobs/{self.job.id}/status",
                    json=payload,
                    api_key=settings.api_key,
                )
            except Exception as e:
                logger.warning("Failed to update job status: %s", e)

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

        # Flush all child trace spans before updating job status
        from hud.telemetry.utils import flush_telemetry

        await flush_telemetry()

        # Update job status to "completed" or "failed"
        if settings.telemetry_enabled:
            status = "failed" if exc_type else "completed"
            payload = {
                "name": self.job.name,
                "status": status,
                "metadata": self.job.metadata,
            }
            if self.job.dataset_link:
                payload["dataset_link"] = self.job.dataset_link

            try:
                await make_request(
                    method="POST",
                    url=f"{settings.hud_telemetry_url}/jobs/{self.job.id}/status",
                    json=payload,
                    api_key=settings.api_key,
                )
            except Exception as e:
                logger.warning("Failed to update job status: %s", e)

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
    trace_id: str | None = None,
) -> AsyncTrace:
    """Create an async trace context for telemetry tracking.

    This is the async equivalent of `hud.trace()` for use in async contexts.
    Status updates are automatically sent and awaited - the trace doesn't exit
    until its status is confirmed on the server.

    Args:
        name: Descriptive name for this trace/task
        root: Whether this is a root trace (updates task status)
        attrs: Additional attributes to attach to the trace
        job_id: Optional job ID to associate with this trace
        task_id: Optional task ID for custom task identifiers
        group_id: Optional group ID to associate with this trace
        trace_id: Optional trace ID (auto-generated if not provided)

    Returns:
        AsyncTrace context manager

    Example:
        >>> import hud
        >>> # Single task - everything is automatic!
        >>> async with hud.async_trace("My Task"):
        ...     result = await agent.run(task)
        >>> # Status is "completed" and spans are flushed - all done!
        >>>
        >>> # Multiple tasks - each trace handles itself
        >>> for task in tasks:
        ...     async with hud.async_trace(task.name):
        ...         await process(task)
        >>> # All traces completed and flushed - nothing more needed!

    Note:
        Use this async version in async code. For sync code, use `hud.trace()`.
        Telemetry is fully automatic - no manual flushing required.
    """
    return AsyncTrace(
        name,
        root=root,
        attrs=attrs,
        job_id=job_id,
        task_id=task_id,
        group_id=group_id,
        trace_id=trace_id,
    )


def async_job(
    name: str,
    metadata: dict[str, Any] | None = None,
    job_id: str | None = None,
    dataset_link: str | None = None,
) -> AsyncJob:
    """Create an async job context for grouping related tasks.

    This is the async equivalent of `hud.job()` for async contexts.
    Status updates are automatically sent and awaited - the job doesn't exit
    until its status is confirmed on the server.

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
        ...         async with hud.async_trace(f"Task {item.id}", job_id=job.id):
        ...             await process(item)
        >>> # Job exits - automatically flushes all child trace spans!

    Note:
        Use this async version in async code. For sync code, use `hud.job()`.
        Telemetry is fully automatic - no manual flushing required.
    """
    return AsyncJob(name, metadata=metadata, job_id=job_id, dataset_link=dataset_link)
