"""The execution engine: tasks, tasksets, rollouts and jobs."""

from __future__ import annotations

from typing import Any

from hud.utils.deprecation import RUNTIME_NAMES, moved_to, resolve_retired

_RETIRED = {
    **moved_to("hud", "Grade", "Job", "Run", "Task", "Taskset", "Trace"),
    **moved_to("hud.eval.run", "rollout"),
    **moved_to("hud.cli.tasksets", "SyncPlan"),
    **moved_to("hud.runtime", *(name for name in RUNTIME_NAMES if name != "DockerBindMount")),
}


def __getattr__(name: str) -> Any:
    return resolve_retired(__name__, name, _RETIRED)
