"""hud-python.

tools for building, evaluating, and training AI agents.
"""

from __future__ import annotations

# Apply patches to third-party libraries early, before other imports
from . import patches as _patches  # noqa: F401
from ._legacy import install as _install_v5_compat
from .client import Grade, Run
from .environment import Environment
from .eval import Chat, Job, SyncPlan, Task, Taskset, launch, task
from .telemetry.instrument import instrument
from .types import Trace

_install_v5_compat()

__all__ = [
    "Chat",
    "Environment",
    "Grade",
    "Job",
    "Run",
    "SyncPlan",
    "Task",
    "Taskset",
    "Trace",
    "instrument",
    "launch",
    "task",
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
