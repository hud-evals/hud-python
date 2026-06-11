"""HUD environment runtime: Workspace + Environment + Task.

The env-side robot runtime (bridges, action providers, sim runners, contract
tooling, recording glue) lives in :mod:`hud.environment.robots`; it is exposed
lazily because it pulls optional dependencies (numpy/msgpack — the ``robot``
extra).
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

from hud.capabilities import Capability

from .env import Environment
from .task import Task, TaskFn, TaskRunner
from .workspace import DEFAULT_SYSTEM_MOUNTS, Mount, MountKind, Workspace

if TYPE_CHECKING:  # static analysers still see the real symbols
    from . import robots


def __getattr__(name: str) -> object:
    if name == "robots":
        value = import_module(".robots", __name__)
        globals()[name] = value  # cache so subsequent lookups skip __getattr__
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)


__all__ = [
    "DEFAULT_SYSTEM_MOUNTS",
    "Capability",
    "Environment",
    "Mount",
    "MountKind",
    "Task",
    "TaskFn",
    "TaskRunner",
    "Workspace",
    "robots",
]
