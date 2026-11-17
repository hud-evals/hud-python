"""HUD: tools for building and evaluating AI agents."""

from __future__ import annotations

from typing import Any

from .environment.env import Environment
from .eval.chat import Chat
from .eval.job import Job
from .eval.run import Run
from .eval.task import Task
from .eval.taskset import Taskset
from .types import Grade, Trace
from .utils.deprecation import RUNTIME_NAMES, moved_to, resolve_retired
from .version import __version__

__all__ = [
    "Chat",
    "Environment",
    "Grade",
    "Job",
    "Run",
    "Task",
    "Taskset",
    "Trace",
    "__version__",
]

_RETIRED = {
    **moved_to(
        "hud.runtime",
        *(
            name
            for name in RUNTIME_NAMES
            if name
            not in {"DaytonaRuntime", "DockerBindMount", "ModalRuntime", "Provider", "Shared"}
        ),
    ),
    **moved_to("hud.capabilities", "Connection"),
    **moved_to("hud.clients", "connect"),
    **moved_to("hud.cli.tasksets", "SyncPlan"),
    **moved_to("hud.telemetry", "instrument"),
    **moved_to("hud.train", "TrainingClient"),
}


def __getattr__(name: str) -> Any:
    return resolve_retired(__name__, name, _RETIRED)
