"""hud-python.

tools for building, evaluating, and training AI agents.
"""

from __future__ import annotations

from .telemetry import (
    Trace,
    async_job,
    async_trace,
    clear_trace,
    create_job,
    get_trace,
    instrument,
    job,
    trace,
)

__all__ = [
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

try:
    from .version import __version__
except ImportError:
    __version__ = "unknown"

try:
    from .utils.pretty_errors import install_pretty_errors

    install_pretty_errors()
except Exception:  # noqa: S110
    pass
