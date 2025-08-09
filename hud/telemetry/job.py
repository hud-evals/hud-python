"""Job management for HUD SDK.

This module provides APIs for managing jobs - logical groupings of related tasks.
Jobs can be used to track experiments, batch processing, training runs, etc.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from contextlib import contextmanager
from datetime import UTC, datetime
from functools import wraps
from typing import TYPE_CHECKING, Any

from hud.server import make_request, make_request_sync
from hud.settings import settings
from hud.utils.async_utils import fire_and_forget

if TYPE_CHECKING:
    from collections.abc import Callable, Generator

logger = logging.getLogger(__name__)


class Job:
    """A job represents a collection of related tasks."""

    def __init__(self, job_id: str, name: str, metadata: dict[str, Any] | None = None) -> None:
        self.id = job_id
        self.name = name
        self.metadata = metadata or {}
        self.status = "created"
        self.created_at = datetime.now(UTC)
        self.tasks: list[str] = []

    def add_task(self, task_id: str) -> None:
        """Associate a task with this job."""
        self.tasks.append(task_id)

    async def update_status(self, status: str) -> None:
        """Update job status on the server."""
        self.status = status
        if settings.telemetry_enabled:
            try:
                await make_request(
                    method="POST",
                    url=f"{settings.base_url}/v2/jobs/{self.id}/status",
                    json={
                        "name": self.name,
                        "status": status,
                        "metadata": self.metadata,
                        "task_count": len(self.tasks),
                    },
                    api_key=settings.api_key,
                )
            except Exception as e:
                logger.warning("Failed to update job status: %s", e)

    def update_status_sync(self, status: str) -> None:
        """Synchronously update job status on the server."""
        self.status = status
        if settings.telemetry_enabled:
            try:
                make_request_sync(
                    method="POST",
                    url=f"{settings.base_url}/v2/jobs/{self.id}/status",
                    json={
                        "name": self.name,
                        "status": status,
                        "metadata": self.metadata,
                        "task_count": len(self.tasks),
                    },
                    api_key=settings.api_key,
                )
            except Exception as e:
                logger.warning("Failed to update job status: %s", e)

    def __repr__(self) -> str:
        return f"Job(id={self.id!r}, name={self.name!r}, status={self.status!r})"


# Global job registry for the decorator pattern
_current_job: Job | None = None


def get_current_job() -> Job | None:
    """Get the currently active job, if any."""
    return _current_job


@contextmanager
def job(
    name: str, metadata: dict[str, Any] | None = None, job_id: str | None = None
) -> Generator[Job, None, None]:
    """Context manager for job tracking.

    Groups related tasks together under a single job for tracking and organization.

    Args:
        name: Human-readable job name
        metadata: Optional metadata dictionary
        job_id: Optional job ID (auto-generated if not provided)

    Yields:
        Job: The job object

    Example:
        with hud.job("training_run", {"model": "gpt-4"}) as job:
            for epoch in range(10):
                with hud.trace(f"epoch_{epoch}", job_id=job.id):
                    train_epoch()
    """
    global _current_job

    if not job_id:
        job_id = f"job-{uuid.uuid4().hex[:12]}"

    job_obj = Job(job_id, name, metadata)

    # Set as current job
    old_job = _current_job
    _current_job = job_obj

    try:
        # Update status to running
        fire_and_forget(job_obj.update_status("running"), "update job status to running")
        yield job_obj
        # Update status to completed synchronously to ensure it completes before process exit
        job_obj.update_status_sync("completed")
    except Exception:
        # Update status to failed synchronously to ensure it completes before process exit
        job_obj.update_status_sync("failed")
        raise
    finally:
        _current_job = old_job


def create_job(name: str, metadata: dict[str, Any] | None = None) -> Job:
    """Create a job without using context manager.

    Useful when you need explicit control over job lifecycle.

    Args:
        name: Human-readable job name
        metadata: Optional metadata dictionary

    Returns:
        Job: The created job object

    Example:
        job = hud.create_job("data_processing")
        try:
            for item in items:
                with hud.trace(f"process_{item.id}", job_id=job.id):
                    process(item)
        finally:
            await job.update_status("completed")
    """
    job_id = f"job-{uuid.uuid4().hex[:12]}"
    return Job(job_id, name, metadata)


def job_decorator(name: str | None = None, **metadata: Any) -> Callable:
    """Decorator for functions that should be tracked as jobs.

    Args:
        name: Job name (defaults to function name)
        **metadata: Additional metadata for the job

    Example:
        @hud.job_decorator("model_training", model="gpt-4", dataset="v2")
        async def train_model(config):
            # This entire function execution is tracked as a job
            await model.train(config)
            return model.evaluate()
    """

    def decorator(func: Callable) -> Callable:
        job_name = name or func.__name__

        @wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            with job(job_name, metadata) as job_obj:
                # Store job ID in function for access
                func._current_job_id = job_obj.id
                try:
                    return await func(*args, **kwargs)
                finally:
                    delattr(func, "_current_job_id")

        @wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            with job(job_name, metadata) as job_obj:
                # Store job ID in function for access
                func._current_job_id = job_obj.id
                try:
                    return func(*args, **kwargs)
                finally:
                    delattr(func, "_current_job_id")

        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


# Convenience exports
__all__ = [
    "Job",
    "create_job",
    "get_current_job",
    "job",
    "job_decorator",
]
