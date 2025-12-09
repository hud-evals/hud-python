"""HUD Telemetry - Instrumentation for agent execution.

This module provides:
- instrument: Function instrumentation decorator

All other APIs are deprecated:
- Job, job, create_job, get_current_job - Use hud.eval() instead
- async_trace(), trace() - Use env.trace() instead
- async_job() - Use hud.eval() instead

Migration:
    # Old (deprecated):
    async with hud.async_trace("Task"):
        await agent.run(task)

    # New (recommended):
    async with env.trace("Task") as tc:
        await agent.run(task)
        tc.reward = result.reward
"""

from __future__ import annotations

from .instrument import instrument


def __getattr__(name: str):  # noqa: ANN202
    """Lazy load deprecated APIs and show warnings."""
    import warnings

    deprecated_apis = {
        # Job APIs (deprecated)
        "Job",
        "job",
        "create_job",
        "get_current_job",
        # OpenTelemetry-based APIs (deprecated, require [agents])
        "async_job",
        "async_trace",
        "clear_trace",
        "get_trace",
        "Trace",
        "trace",
    }

    if name in deprecated_apis:
        warnings.warn(
            f"hud.telemetry.{name} is deprecated. Use hud.eval() or env.trace() instead.",
            DeprecationWarning,
            stacklevel=2,
        )

        # Import from submodules
        if name in ("Job", "job", "create_job", "get_current_job"):
            from .job import Job, create_job, get_current_job, job

            return {"Job": Job, "job": job, "create_job": create_job, "get_current_job": get_current_job}[name]
        elif name in ("async_job", "async_trace"):
            from .async_context import async_job, async_trace

            return async_job if name == "async_job" else async_trace
        elif name in ("clear_trace", "get_trace"):
            from .replay import clear_trace, get_trace

            return clear_trace if name == "clear_trace" else get_trace
        elif name in ("Trace", "trace"):
            from .trace import Trace, trace

            return Trace if name == "Trace" else trace

    raise AttributeError(f"module 'hud.telemetry' has no attribute {name!r}")


__all__ = [
    # Core (always available)
    "instrument",
    # Deprecated
    "Job",
    "Trace",
    "async_job",
    "async_trace",
    "clear_trace",
    "create_job",
    "get_current_job",
    "get_trace",
    "job",
    "trace",
]
