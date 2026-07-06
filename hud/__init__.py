"""hud.

tools for building, evaluating, and training AI agents.
"""

from __future__ import annotations

# Apply patches to third-party libraries early, before other imports
from . import patches as _patches  # noqa: F401
from ._legacy import install as _install_v5_compat
from .clients import connect
from .environment import Environment
from .eval import (
    Chat,
    DockerRuntime,
    Grade,
    HostedRuntime,
    HUDRuntime,
    Job,
    LocalRuntime,
    Run,
    Runtime,
    RuntimeConfig,
    RuntimeGPU,
    RuntimeLimits,
    RuntimeResources,
    SubprocessRuntime,
    SyncPlan,
    Task,
    Taskset,
)
from .telemetry.instrument import instrument
from .train import TrainingClient
from .types import Trace

_install_v5_compat()

__all__ = [
    "Chat",
    "DockerRuntime",
    "Environment",
    "Grade",
    "HUDRuntime",
    "HostedRuntime",
    "Job",
    "LocalRuntime",
    "Run",
    "Runtime",
    "RuntimeConfig",
    "RuntimeGPU",
    "RuntimeLimits",
    "RuntimeResources",
    "SubprocessRuntime",
    "SyncPlan",
    "Task",
    "Taskset",
    "Trace",
    "TrainingClient",
    "connect",
    "instrument",
]


def __getattr__(name: str) -> object:
    # Lazy: hud.wrap pulls in the robot extra (numpy, websockets) only when used.
    if name == "wrap":
        from .environment.robot import wrap

        return wrap
    raise AttributeError(f"module 'hud' has no attribute {name!r}")


try:
    from .version import __version__
except ImportError:
    __version__ = "unknown"
