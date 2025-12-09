"""HUD Telemetry - Tracing and job management for agent execution.

.. deprecated::
    The `hud.telemetry` module is deprecated and will be removed in a future version.
    Use `env.trace()` from `hud.environment.Environment` instead.

    This module requires the [agents] extra:
        pip install hud-python[agents]

    Migration:
        # Old (deprecated):
        async with hud.async_trace("Task"):
            await agent.run(task)

        # New (recommended):
        async with env.trace("Task") as tc:
            await agent.run(task)
            tc.reward = result.reward

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

import warnings

warnings.warn(
    "The hud.telemetry module is deprecated. Use env.trace() instead. "
    "This module requires pip install hud-python[agents].",
    DeprecationWarning,
    stacklevel=2,
)

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
