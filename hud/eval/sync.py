"""Retired import paths for platform tasksets; hud v0.7.0 removes this module."""

from __future__ import annotations

from typing import Any

from hud.utils.deprecation import moved_to, resolve_retired

_RETIRED = {
    **moved_to(
        "hud.cli.tasksets",
        "SyncPlan",
        "TaskUploadResult",
        "diff",
        "task_upload_payload",
        "upload_taskset",
        "TasksetRecord",
    ),
}


def __getattr__(name: str) -> Any:
    return resolve_retired(__name__, name, _RETIRED)
