"""hud-python.

tools for building, evaluating, and training AI agents.
"""

from __future__ import annotations

from .environment import Environment
from .telemetry.instrument import instrument
from .telemetry.job import Job, create_job, get_current_job, job

__all__ = [
    "Environment",
    "Job",
    "create_job",
    "get_current_job",
    "instrument",
    "job",
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
