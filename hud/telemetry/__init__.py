"""HUD Telemetry - User-facing APIs for tracing and job management.

This module provides the main telemetry APIs that users interact with:
- trace: Context manager for tracing code execution
- job: Context manager and utilities for job management
- instrument: Decorator for instrumenting functions
- get_trace: Retrieve collected traces for replay/analysis
"""

from __future__ import annotations

from .instrument import instrument
from .job import Job, create_job, job
from .replay import clear_trace, get_trace
from .trace import trace

__all__ = [
    "Job",
    "clear_trace",
    "create_job",
    "get_trace",
    "instrument",
    "job",
    "trace",
]
