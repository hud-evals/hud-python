"""hud-python.

tools for building, evaluating, and training AI agents.
"""

from __future__ import annotations

from .telemetry import clear_trace, create_job, get_trace, instrument, job, trace

__all__ = [
    "clear_trace",
    "create_job",
    "get_trace",
    "instrument",
    "job",
    "trace",
]

try:
    from .version import __version__
except ImportError:
    __version__ = "unknown"
