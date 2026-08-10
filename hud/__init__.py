"""hud.

tools for building, evaluating, and training AI agents.
"""

from __future__ import annotations

# Apply patches to third-party libraries early, before other imports
from . import patches as _patches  # noqa: F401
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
    RootfsProfile,
    Run,
    Runtime,
    RuntimeConfig,
    RuntimeGPU,
    RuntimeLimits,
    RuntimeResources,
    StorageConfig,
    SubprocessRuntime,
    SyncPlan,
    Task,
    Taskset,
    WorkspaceProfile,
)
from .telemetry.instrument import instrument
from .train import TrainingClient
from .types import Trace
from .version import __version__

__all__ = [
    "Chat",
    "DockerRuntime",
    "Environment",
    "Grade",
    "HUDRuntime",
    "HostedRuntime",
    "Job",
    "LocalRuntime",
    "RootfsProfile",
    "Run",
    "Runtime",
    "RuntimeConfig",
    "RuntimeGPU",
    "RuntimeLimits",
    "RuntimeResources",
    "StorageConfig",
    "SubprocessRuntime",
    "SyncPlan",
    "Task",
    "Taskset",
    "Trace",
    "TrainingClient",
    "WorkspaceProfile",
    "__version__",
    "connect",
    "instrument",
]
