"""Async context managers for HUD telemetry.

This module provides async versions of trace and job context managers
that properly handle async operations without blocking the event loop.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Any, Optional

from hud.otel import configure_telemetry
from hud.otel.context import (
    trace as OtelTrace,
    _update_task_status_async,
    _print_trace_url,
    _print_trace_complete_url
)
from hud.settings import settings
from hud.shared import make_request
from hud.telemetry.job import Job, _print_job_url, _print_job_complete_url, get_current_job
from hud.telemetry.trace import Trace
from hud.utils.task_tracking import track_task

logger = logging.getLogger(__name__)

# Thread-local storage for current job
_current_job: Optional[Job] = None


class AsyncTrace:
    """Async context manager for HUD traces.
    
    This properly handles async operations without blocking the event loop.
    """
    
    def __init__(
        self,
        name: str = "Test task from hud",
        *,
        root: bool = True,
        attrs: dict[str, Any] | None = None,
        job_id: str | None = None,
        task_id: str | None = None,
    ):
        self.name = name
        self.root = root
        self.attrs = attrs or {}
        self.job_id = job_id
        self.task_id = task_id
        self.task_run_id = str(uuid.uuid4())
        self.trace_obj = Trace(self.task_run_id, name, job_id, task_id)
        self._otel_trace = None
    
    async def __aenter__(self) -> Trace:
        """Async enter - properly awaits status update."""
        # Configure telemetry if needed
        configure_telemetry()
        
        # Create the OpenTelemetry trace
        self._otel_trace = OtelTrace(
            self.task_run_id,
            is_root=self.root,
            span_name=self.name,
            attributes=self.attrs,
            job_id=self.job_id,
            task_id=self.task_id,
        )
        self._otel_trace.__enter__()
        
        # Update status asynchronously if root
        if self.root and settings.telemetry_enabled and settings.api_key:
            # Use tracked task instead of fire-and-forget
            track_task(
                _update_task_status_async(
                    self.task_run_id,
                    "running",
                    job_id=self.job_id,
                    trace_name=self.name,
                    task_id=self.task_id
                ),
                f"trace {self.task_run_id} status=running"
            )
            
            # Print URL if not part of a job
            if not self.job_id:
                _print_trace_url(self.task_run_id)
        
        logger.debug(f"Started async trace for task_run_id={self.task_run_id}")
        return self.trace_obj
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async exit - properly awaits status update."""
        # Update status asynchronously if root
        if self.root and settings.telemetry_enabled and settings.api_key:
            status = "error" if exc_type else "completed"
            
            # Use tracked task instead of fire-and-forget
            track_task(
                _update_task_status_async(
                    self.task_run_id,
                    status,
                    job_id=self.job_id,
                    error_message=str(exc_val) if exc_val else None,
                    trace_name=self.name,
                    task_id=self.task_id
                ),
                f"trace {self.task_run_id} status={status}"
            )
            
            # Print completion URL if not part of a job
            if not self.job_id:
                _print_trace_complete_url(self.task_run_id, error_occurred=bool(exc_type))
        
        # Exit the OpenTelemetry trace
        if self._otel_trace:
            self._otel_trace.__exit__(exc_type, exc_val, exc_tb)
        
        
        logger.debug(f"Ended async trace for task_run_id={self.task_run_id}")


class AsyncJob:
    """Async context manager for HUD jobs.
    
    This properly handles async operations without blocking the event loop.
    """
    
    def __init__(
        self,
        name: str,
        metadata: dict[str, Any] | None = None,
        job_id: str | None = None,
        dataset_link: str | None = None,
    ):
        self.job_id = job_id or str(uuid.uuid4())
        self.job = Job(self.job_id, name, metadata, dataset_link)
    
    async def __aenter__(self) -> Job:
        """Async enter - properly awaits status update."""
        global _current_job
        
        # Store old job
        self._old_job = _current_job
        _current_job = self.job
        
        # Update status asynchronously
        if settings.telemetry_enabled:
            payload = {
                "name": self.job.name,
                "status": "running",
                "metadata": self.job.metadata,
            }
            if self.job.dataset_link:
                payload["dataset_link"] = self.job.dataset_link
            
            # Use tracked task
            track_task(
                make_request(
                    method="POST",
                    url=f"{settings.hud_telemetry_url}/jobs/{self.job.id}/status",
                    json=payload,
                    api_key=settings.api_key,
                ),
                f"job {self.job.id} status=running"
            )
        
        # Print URL
        _print_job_url(self.job.id, self.job.name)
        
        logger.debug(f"Started async job {self.job.id}")
        return self.job
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async exit - properly awaits status update."""
        global _current_job
        
        # Update status asynchronously
        if settings.telemetry_enabled:
            status = "failed" if exc_type else "completed"
            payload = {
                "name": self.job.name,
                "status": status,
                "metadata": self.job.metadata,
            }
            if self.job.dataset_link:
                payload["dataset_link"] = self.job.dataset_link
            
            # Use tracked task
            track_task(
                make_request(
                    method="POST",
                    url=f"{settings.hud_telemetry_url}/jobs/{self.job.id}/status",
                    json=payload,
                    api_key=settings.api_key,
                ),
                f"job {self.job.id} status={status}"
            )
        
        # Print completion URL
        _print_job_complete_url(self.job.id, self.job.name, error_occurred=bool(exc_type))
        
        # Restore old job
        _current_job = self._old_job
        
        
        logger.debug(f"Ended async job {self.job.id}")


def async_trace(
    name: str = "Test task from hud",
    *,
    root: bool = True,
    attrs: dict[str, Any] | None = None,
    job_id: str | None = None,
    task_id: str | None = None,
) -> AsyncTrace:
    """Create an async trace context manager.
    
    This should be used with `async with`:
    
    ```python
    async with async_trace("My Task") as trace:
        await do_something()
        await trace.log({"progress": 0.5})
    ```
    """
    return AsyncTrace(name, root=root, attrs=attrs, job_id=job_id, task_id=task_id)


def async_job(
    name: str,
    metadata: dict[str, Any] | None = None,
    job_id: str | None = None,
    dataset_link: str | None = None,
) -> AsyncJob:
    """Create an async job context manager.
    
    This should be used with `async with`:
    
    ```python
    async with async_job("My Job") as job:
        async with async_trace("Task", job_id=job.id) as trace:
            await do_something()
    ```
    """
    return AsyncJob(name, metadata=metadata, job_id=job_id, dataset_link=dataset_link)
