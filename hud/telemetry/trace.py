"""User-facing trace context manager for HUD telemetry.

This module provides the simple trace() API that users interact with.
The actual OpenTelemetry implementation is in hud.otel.
"""

from __future__ import annotations

import logging
import uuid
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from hud.otel import configure_telemetry
from hud.otel import trace as OtelTrace
from hud.settings import settings
from hud.shared import make_request, make_request_sync

if TYPE_CHECKING:
    from collections.abc import Generator

logger = logging.getLogger(__name__)

__all__ = ["Trace", "trace"]


class Trace:
    """A trace represents a single task execution with telemetry."""

    def __init__(
        self,
        trace_id: str,
        name: str,
        job_id: str | None = None,
        task_id: str | None = None,
        group_id: str | None = None,
    ) -> None:
        self.id = trace_id
        self.name = name
        self.job_id = job_id
        self.task_id = task_id
        self.group_id = group_id
        self.created_at = datetime.now(UTC)

    async def log(self, metrics: dict[str, Any]) -> None:
        """Log metrics to this trace.

        Args:
            metrics: Dictionary of metric name to value pairs

        Example:
            await trace.log({"step": 1, "loss": 0.5, "accuracy": 0.92})
        """
        if settings.telemetry_enabled:
            try:
                await make_request(
                    method="POST",
                    url=f"{settings.hud_telemetry_url}/traces/{self.id}/log",
                    json={"metrics": metrics, "timestamp": datetime.now(UTC).isoformat()},
                    api_key=settings.api_key,
                )
            except Exception as e:
                logger.warning("Failed to log metrics to trace: %s", e)

    def log_sync(self, metrics: dict[str, Any]) -> None:
        """Synchronously log metrics to this trace.

        Args:
            metrics: Dictionary of metric name to value pairs

        Example:
            trace.log_sync({"step": 1, "loss": 0.5, "accuracy": 0.92})
        """
        if settings.telemetry_enabled:
            try:
                make_request_sync(
                    method="POST",
                    url=f"{settings.hud_telemetry_url}/traces/{self.id}/log",
                    json={"metrics": metrics, "timestamp": datetime.now(UTC).isoformat()},
                    api_key=settings.api_key,
                )
            except Exception as e:
                logger.warning("Failed to log metrics to trace: %s", e)

    def __repr__(self) -> str:
        return f"Trace(id={self.id!r}, name={self.name!r})"


@contextmanager
def trace(
    name: str = "Test task from hud",
    *,
    root: bool = True,
    attrs: dict[str, Any] | None = None,
    job_id: str | None = None,
    task_id: str | None = None,
    group_id: str | None = None,
    trace_id: str | None = None,
) -> Generator[Trace, None, None]:
    """Start a HUD trace context for telemetry tracking.

    A unique task_run_id is automatically generated for each trace unless provided.

    Args:
        name: Descriptive name for this trace/task
        root: Whether this is a root trace (updates task status)
        attrs: Additional attributes to attach to the trace
        job_id: Optional job ID to associate with this trace
        task_id: Optional task ID (for custom task identifiers)
        group_id: Optional group ID to associate with this trace
        trace_id: Optional trace ID (auto-generated if not provided)

    Yields:
        Trace: The trace object with logging capabilities

    Example:
        >>> import hud
        >>> with hud.trace("My Task") as trace:
        ...     do_work()
        ...     trace.log_sync({"step": 1, "progress": 0.5})
        >>> # For async code, use async_trace
        >>> async with hud.async_trace("Async Task") as trace:
        ...     await do_async_work()
        ...     await trace.log({"loss": 0.23})

    Note:
        This is a synchronous context manager that uses blocking HTTP calls.
        For async code, use `hud.async_trace()` instead.
    """
    # Ensure telemetry is configured
    configure_telemetry()

    # Use provided trace_id or generate one
    if trace_id:
        task_run_id = trace_id
    else:
        # Only generate task_run_id if using HUD backend
        # For custom OTLP backends, we don't need it
        from hud.settings import get_settings

        settings = get_settings()

        if settings.telemetry_enabled and settings.api_key:
            task_run_id = str(uuid.uuid4())
        else:
            # Use a placeholder for custom backends
            logger.warning(
                "HUD API key is not set, using a placeholder for the task run ID. If this looks wrong, check your API key."  # noqa: E501
            )
            task_run_id = str(uuid.uuid4())

    # Create trace object
    trace_obj = Trace(task_run_id, name, job_id, task_id, group_id)

    # Delegate to OpenTelemetry implementation
    with OtelTrace(
        task_run_id,
        is_root=root,
        span_name=name,
        attributes=attrs or {},
        job_id=job_id,
        task_id=task_id,
        group_id=group_id,
    ):
        yield trace_obj
