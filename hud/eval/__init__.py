"""The execution engine: tasks, tasksets, rollouts and jobs."""

from __future__ import annotations

from importlib import import_module
from typing import Any

from hud.utils.deprecation import RUNTIME_NAMES, Retired, moved_to, resolve_retired

_RETIRED = {
    **moved_to("hud", "Grade", "Job", "Run", "Task", "Taskset", "Trace"),
    "Chat": Retired(
        lambda: import_module("hud.eval.chat").Chat,
        "copy it from the a2a-chat cookbook (`cookbooks/a2a-chat/chat.py`)",
    ),
    **moved_to("hud.eval.run", "rollout"),
    **moved_to("hud.cli.tasksets", "SyncPlan"),
    **moved_to("hud.runtime", *(name for name in RUNTIME_NAMES if name != "DockerBindMount")),
}


def __getattr__(name: str) -> Any:
    return resolve_retired(__name__, name, _RETIRED)
