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

from hud.settings import settings
from hud.shared import make_request, make_request_sync

if TYPE_CHECKING:
    from collections.abc import Callable, Generator

logger = logging.getLogger(__name__)


class Job:
    """A job represents a collection of related tasks."""

    def __init__(
        self,
        job_id: str,
        name: str,
        metadata: dict[str, Any] | None = None,
        dataset_link: str | None = None,
    ) -> None:
        self.id = job_id
        self.name = name
        self.metadata = metadata or {}
        self.dataset_link = dataset_link
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
                payload = {
                    "name": self.name,
                    "status": status,
                    "metadata": self.metadata,
                }
                if self.dataset_link:
                    payload["dataset_link"] = self.dataset_link

                await make_request(
                    method="POST",
                    url=f"{settings.hud_telemetry_url}/jobs/{self.id}/status",
                    json=payload,
                    api_key=settings.api_key,
                )
            except Exception as e:
                logger.warning("Failed to update job status: %s", e)

    def update_status_sync(self, status: str) -> None:
        """Synchronously update job status on the server."""
        self.status = status
        if settings.telemetry_enabled:
            try:
                payload = {
                    "name": self.name,
                    "status": status,
                    "metadata": self.metadata,
                }
                if self.dataset_link:
                    payload["dataset_link"] = self.dataset_link

                make_request_sync(
                    method="POST",
                    url=f"{settings.hud_telemetry_url}/jobs/{self.id}/status",
                    json=payload,
                    api_key=settings.api_key,
                )
            except Exception as e:
                logger.warning("Failed to update job status: %s", e)

    async def log(self, metrics: dict[str, Any]) -> None:
        """Log metrics to the job.

        Args:
            metrics: Dictionary of metric name to value pairs

        Example:
            await job.log({"loss": 0.5, "accuracy": 0.95, "epoch": 1})
        """
        if settings.telemetry_enabled:
            try:
                await make_request(
                    method="POST",
                    url=f"{settings.hud_telemetry_url}/jobs/{self.id}/log",
                    json={"metrics": metrics, "timestamp": datetime.now(UTC).isoformat()},
                    api_key=settings.api_key,
                )
            except Exception as e:
                logger.warning("Failed to log metrics to job: %s", e)

    def log_sync(self, metrics: dict[str, Any]) -> None:
        """Synchronously log metrics to the job.

        Args:
            metrics: Dictionary of metric name to value pairs

        Example:
            job.log_sync({"loss": 0.5, "accuracy": 0.95, "epoch": 1})
        """
        if settings.telemetry_enabled:
            try:
                make_request_sync(
                    method="POST",
                    url=f"{settings.hud_telemetry_url}/jobs/{self.id}/log",
                    json={"metrics": metrics, "timestamp": datetime.now(UTC).isoformat()},
                    api_key=settings.api_key,
                )
            except Exception as e:
                logger.warning("Failed to log metrics to job: %s", e)

    def __repr__(self) -> str:
        return f"Job(id={self.id!r}, name={self.name!r}, status={self.status!r})"


# Global job registry for the decorator pattern
_current_job: Job | None = None


def _print_job_url(job_id: str, job_name: str) -> None:
    """Print the job URL in a colorful box."""
    # Only print HUD URL if HUD telemetry is enabled and has API key
    if not (settings.telemetry_enabled and settings.api_key):
        return

    url = f"https://hud.ai/jobs/{job_id}"
    header = f"🚀 Job '{job_name}' started:"

    # ANSI color codes
    DIM = "\033[90m"  # Dim/Gray for border
    GOLD = "\033[33m"  # Gold/Yellow for URL
    RESET = "\033[0m"
    BOLD = "\033[1m"

    # Calculate box width based on the longest line
    box_width = max(len(url), len(header)) + 6

    # Box drawing characters
    top_border = "╔" + "═" * (box_width - 2) + "╗"
    bottom_border = "╚" + "═" * (box_width - 2) + "╝"
    divider = "╟" + "─" * (box_width - 2) + "╢"

    # Center the content
    header_padding = (box_width - len(header) - 2) // 2
    url_padding = (box_width - len(url) - 2) // 2

    # Print the box
    print(f"\n{DIM}{top_border}{RESET}")  # noqa: T201
    print(  # noqa: T201
        f"{DIM}║{RESET}{' ' * header_padding}{header}{' ' * (box_width - len(header) - header_padding - 3)}{DIM}║{RESET}"  # noqa: E501
    )
    print(f"{DIM}{divider}{RESET}")  # noqa: T201
    print(  # noqa: T201
        f"{DIM}║{RESET}{' ' * url_padding}{BOLD}{GOLD}{url}{RESET}{' ' * (box_width - len(url) - url_padding - 2)}{DIM}║{RESET}"  # noqa: E501
    )
    print(f"{DIM}{bottom_border}{RESET}\n")  # noqa: T201


def _print_job_complete_url(job_id: str, job_name: str, error_occurred: bool = False) -> None:
    """Print the job completion URL with appropriate messaging."""
    # Only print HUD URL if HUD telemetry is enabled and has API key
    if not (settings.telemetry_enabled and settings.api_key):
        return

    url = f"https://hud.ai/jobs/{job_id}"

    # ANSI color codes
    GREEN = "\033[92m"
    RED = "\033[91m"
    GOLD = "\033[33m"
    RESET = "\033[0m"
    DIM = "\033[2m"
    BOLD = "\033[1m"

    if error_occurred:
        print(  # noqa: T201
            f"\n{RED}✗ Job '{job_name}' failed!{RESET} {DIM}View details at:{RESET} {BOLD}{GOLD}{url}{RESET}\n"  # noqa: E501
        )
    else:
        print(  # noqa: T201
            f"\n{GREEN}✓ Job '{job_name}' complete!{RESET} {DIM}View all results at:{RESET} {BOLD}{GOLD}{url}{RESET}\n"  # noqa: E501
        )


def get_current_job() -> Job | None:
    """Get the currently active job, if any."""
    return _current_job


@contextmanager
def job(
    name: str,
    metadata: dict[str, Any] | None = None,
    job_id: str | None = None,
    dataset_link: str | None = None,
) -> Generator[Job, None, None]:
    """Context manager for job tracking and organization.

    Groups related tasks together under a single job for tracking and visualization.

    Args:
        name: Human-readable job name
        metadata: Optional metadata dictionary
        job_id: Optional job ID (auto-generated if not provided)
        dataset_link: Optional HuggingFace dataset identifier (e.g. "hud-evals/SheetBench-50")

    Yields:
        Job: The job object

    Example:
        >>> import hud
        >>> with hud.job("training_run", {"model": "gpt-4"}) as job:
        ...     for epoch in range(10):
        ...         with hud.trace(f"epoch_{epoch}", job_id=job.id):
        ...             train_epoch()
        >>> # For async code, use async_job
        >>> async with hud.async_job("batch_processing") as job:
        ...     async with hud.async_trace("task", job_id=job.id):
        ...         await process()

    Note:
        This is a synchronous context manager that uses blocking HTTP calls.
        For async code, use `hud.async_job()` instead.
    """
    global _current_job

    if not job_id:
        job_id = str(uuid.uuid4())

    job_obj = Job(job_id, name, metadata, dataset_link)

    # Set as current job
    old_job = _current_job
    _current_job = job_obj

    try:
        job_obj.update_status_sync("running")
        _print_job_url(job_obj.id, job_obj.name)
        yield job_obj
        job_obj.update_status_sync("completed")
        _print_job_complete_url(job_obj.id, job_obj.name, error_occurred=False)
    except Exception:
        job_obj.update_status_sync("failed")
        _print_job_complete_url(job_obj.id, job_obj.name, error_occurred=True)
        raise
    finally:
        _current_job = old_job


def create_job(
    name: str,
    metadata: dict[str, Any] | None = None,
    dataset_link: str | None = None,
    job_id: str | None = None,
) -> Job:
    """Create a job without using context manager.

    Useful when you need explicit control over job lifecycle.

    Args:
        name: Human-readable job name
        metadata: Optional metadata dictionary
        dataset_link: Optional HuggingFace dataset identifier (e.g. "hud-evals/SheetBench-50")
        job_id: Optional job ID (auto-generated if not provided)
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
    job_id = job_id or str(uuid.uuid4())
    return Job(job_id, name, metadata, dataset_link)


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
