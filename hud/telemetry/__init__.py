"""HUD Telemetry - Tracing and job management for agent execution.

This module provides telemetry APIs for tracking agent execution:

Standard Usage (for most users):
    - trace(): Context manager for tracing code execution
    - job(): Context manager for grouping related tasks
    - instrument(): Decorator for instrumenting functions
    - get_trace(): Retrieve collected traces for replay/analysis

High-Concurrency Usage (200+ parallel tasks):
    - async_trace(): Async context manager for traces (prevents event loop blocking)
    - async_job(): Async context manager for jobs (prevents event loop blocking)
    
The async versions are automatically used by run_dataset() and other high-concurrency
functions. Most users don't need to use them directly.
"""

from __future__ import annotations

from .async_context import async_job, async_trace
from .instrument import instrument
from .job import Job, create_job, job
from .replay import clear_trace, get_trace
from .trace import Trace, trace

__all__ = [
    # Standard synchronous APIs (for typical usage)
    "Job",
    "Trace",
    "clear_trace",
    "create_job",
    "get_trace",
    "instrument",
    "job",
    "trace",
    # Async APIs (for high-concurrency scenarios)
    "async_job",
    "async_trace",
]
