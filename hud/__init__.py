"""hud-python.

tools for building, evaluating, and training AI agents.
"""

from __future__ import annotations

# Apply patches to third-party libraries early, before other imports
from . import patches as _patches  # noqa: F401
from .client import Grade, Run
from .environment import Environment
from .eval import Job, SyncPlan, Task, Taskset, launch, task
from .services import Chat
from .telemetry.instrument import instrument

__all__ = [
    "Chat",
    "Environment",
    "Grade",
    "Job",
    "Run",
    "SyncPlan",
    "Task",
    "Taskset",
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
