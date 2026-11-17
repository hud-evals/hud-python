"""Platform tasksets for the CLI: finding one, diffing a local taskset against it, uploading."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, with_config
from typing_extensions import TypedDict

from hud.eval.taskset import taskset_id_for
from hud.utils.exceptions import HudRequestError

if TYPE_CHECKING:
    from hud.eval.task import Task
    from hud.eval.taskset import Taskset
    from hud.utils.platform import PlatformClient


@with_config(ConfigDict(extra="allow"))
class TaskUploadResult(TypedDict):
    """What ``POST /tasks/upload`` did, with every other field of its reply."""

    taskset_id: str
    tasks_created: int
    tasks_updated: int


@dataclass(slots=True)
class SyncPlan:
    """Diff between a local taskset and a remote taskset."""

    to_create: list[Task] = field(default_factory=list)
    to_update: list[Task] = field(default_factory=list)
    unchanged: list[Task] = field(default_factory=list)
    remote_only: list[Task] = field(default_factory=list)
    taskset_name: str = ""

    @property
    def to_apply(self) -> list[Task]:
        return [*self.to_create, *self.to_update]

    def summary(self) -> str:
        lines = [f"Sync plan for '{self.taskset_name or 'taskset'}'"]
        lines.append(f"  Create: {len(self.to_create)}")
        lines.append(f"  Update: {len(self.to_update)}")
        lines.append(f"  Unchanged: {len(self.unchanged)}")
        lines.append(f"  Remote-only: {len(self.remote_only)}")
        return "\n".join(lines)


def diff(local: Taskset, remote: Taskset) -> SyncPlan:
    """Classify local tasks against a remote taskset by slug + content signature."""
    remote_by_slug = dict(remote.tasks)
    to_create: list[Task] = []
    to_update: list[Task] = []
    unchanged: list[Task] = []

    for slug, task in local.tasks.items():
        existing = remote_by_slug.pop(slug, None)
        if existing is None:
            to_create.append(task)
            continue
        if _task_signature(task) == _task_signature(existing):
            unchanged.append(task)
        else:
            to_update.append(task)

    return SyncPlan(
        to_create=to_create,
        to_update=to_update,
        unchanged=unchanged,
        remote_only=list(remote_by_slug.values()),
        taskset_name=remote.name or local.name,
    )


def upload_taskset(
    platform: PlatformClient,
    name: str,
    tasks: list[Task],
    *,
    project_id: str | None = None,
    taskset_id: str | None = None,
) -> TaskUploadResult:
    """Upload tasks to a platform taskset, creating it if needed."""
    payload: dict[str, Any] = {
        "taskset_name": name,
        "tasks": [task_upload_payload(task) for task in tasks],
    }
    if project_id:
        payload["project_id"] = project_id
    if taskset_id:
        payload["taskset_id"] = taskset_id
    return platform.post("/tasks/upload", TaskUploadResult, json=payload)


def task_upload_payload(task: Task) -> dict[str, Any]:
    row = task.model_dump(mode="json", exclude_none=True)
    return {
        "name": row.pop("slug"),
        "env": {"name": row.pop("env")},
        "task_id": row.pop("id"),
        **row,
    }


def _task_signature(task: Task) -> str:
    # The slug keys the row; everything else, its env included, is what it runs.
    sig_data = task.model_dump(mode="json", exclude_none=True, exclude={"slug"})
    return json.dumps(
        sig_data,
        sort_keys=True,
        default=str,
        separators=(",", ":"),
    )


class TasksetRecord(BaseModel):
    """A taskset as ``GET /tasksets/{id}`` and ``POST /tasksets`` return it."""

    id: uuid.UUID
    name: str
    project_id: uuid.UUID | None = None
    can_edit: bool = False


def find_taskset(platform: PlatformClient, name_or_id: str) -> TasksetRecord | None:
    """The platform taskset with this id or name, or None when there is none."""
    taskset_id = taskset_id_for(platform, name_or_id)
    if taskset_id is None:
        return None
    try:
        return platform.get(f"/tasksets/{taskset_id}", TasksetRecord)
    except HudRequestError as exc:
        if exc.status_code == 404:
            return None
        raise
