"""hud-python.

tools for building, evaluating, and training AI agents.
"""

from __future__ import annotations

# Import submodules to make them available as attributes
from . import tools, utils
from .telemetry import clear_trace, create_job, get_trace, instrument, job, trace

__all__ = [
    "clear_trace",
    "create_job",
    "get_trace",
    "instrument",
    "job",
    "tools",
    "trace",
    "utils",
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
