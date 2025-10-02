"""HUD Telemetry - Tracing and job management for agent execution.

Provides telemetry APIs for tracking agent execution and experiments.

Standard Usage:
    >>> import hud
    >>> with hud.trace("My Task"):
    ...     do_work()

    >>> with hud.job("My Job") as job:
    ...     with hud.trace("Task", job_id=job.id):
    ...         do_work()

High-Concurrency Usage (200+ parallel tasks):
    >>> import hud
    >>> async with hud.async_job("Evaluation") as job:
    ...     async with hud.async_trace("Task", job_id=job.id):
    ...         await do_async_work()

APIs:
    - trace(), job() - Standard context managers (for typical usage)
    - async_trace(), async_job() - Async context managers (for high concurrency)
    - instrument() - Decorator for instrumenting functions
    - get_trace() - Retrieve collected traces for replay

Note:
    Use async_trace/async_job only for high-concurrency scenarios (200+ tasks).
    The run_dataset() function uses them automatically.
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
