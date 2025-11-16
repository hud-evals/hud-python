"""HUD Telemetry - Tracing and job management for agent execution.

Provides telemetry APIs for tracking agent execution and experiments.

Async Usage (Recommended):
    >>> import hud
    >>> async with hud.async_trace("Task"):
    ...     await agent.run(task)
    >>> async with hud.async_job("Evaluation") as job:
    ...     async with hud.async_trace("Task", job_id=job.id):
    ...         await agent.run(task)

Sync Usage:
    >>> import hud
    >>> with hud.trace("Task"):
    ...     do_work()
    >>> with hud.job("My Job") as job:
    ...     with hud.trace("Task", job_id=job.id):
    ...         do_work()

APIs:
    - async_trace(), async_job() - Async context managers (recommended)
    - trace(), job() - Sync context managers
    - flush_telemetry() - Manual span flushing (rarely needed)
    - instrument() - Function instrumentation decorator
"""

from __future__ import annotations

from .async_context import async_job, async_trace
from .instrument import instrument
from .job import Job, create_job, job
from .replay import clear_trace, get_trace
from .trace import Trace, trace

__all__ = [
    "Job",
    "Trace",
    "async_job",
    "async_trace",
    "clear_trace",
    "create_job",
    "get_trace",
    "instrument",
    "job",
    "trace",
]
