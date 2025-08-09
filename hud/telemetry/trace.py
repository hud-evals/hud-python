"""User-facing trace context manager for HUD telemetry.

This module provides the simple trace() API that users interact with.
The actual OpenTelemetry implementation is in hud.otel.
"""

from __future__ import annotations

import uuid
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

from hud.otel import configure_telemetry
from hud.otel import trace as OtelTrace

if TYPE_CHECKING:
    from collections.abc import Generator

__all__ = ["trace"]


@contextmanager
def trace(
    name: str = "Test task from hud",
    *,
    root: bool = True,
    attrs: dict[str, Any] | None = None,
    job_id: str | None = None,
) -> Generator[str, None, None]:
    """Start a HUD trace context.

    A unique task_run_id is automatically generated for each trace.

    Args:
        name: Descriptive name for this trace/task
        root: Whether this is a root trace (updates task status)
        attrs: Additional attributes to attach to the trace
        job_id: Optional job ID to associate with this trace

    Yields:
        str: The auto-generated task run ID

    Usage:
        import hud

        with hud.trace("My Task") as task_run_id:
            # Your code here
            print(f"Running task: {task_run_id}")

        # Or with default name:
        with hud.trace() as task_run_id:
            pass

        # Or with job_id:
        with hud.trace("My Task", job_id="job-123") as task_run_id:
            pass
    """
    # Ensure telemetry is configured
    configure_telemetry()

    # Always auto-generate a task_run_id
    task_run_id = str(uuid.uuid4())

    # Delegate to OpenTelemetry implementation
    with OtelTrace(
        task_run_id, is_root=root, span_name=name, attributes=attrs or {}, job_id=job_id
    ) as run_id:
        yield run_id
