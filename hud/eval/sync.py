"""Platform persistence for tasksets: diff plans and the fetch/upload wire format.

Taskset endpoints and the upload payload shape.
Transport (auth, retries, errors) is :mod:`hud.utils.platform`; the shapes and
the local-vs-remote :func:`diff` live here, out of the collection itself.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any
from urllib.parse import quote

from pydantic import BaseModel, ConfigDict, TypeAdapter, field_validator, with_config
from typing_extensions import TypedDict

from hud.utils.exceptions import HudRequestError

from .task import Task

if TYPE_CHECKING:
    from hud.utils.platform import PlatformClient

    from .taskset import Taskset


class TasksetRecord(BaseModel):
    """A taskset as ``GET /tasksets/{id}`` and ``POST /tasksets`` return it."""

    id: uuid.UUID
    name: str
    project_id: uuid.UUID | None = None
    can_edit: bool = False


@with_config(ConfigDict(extra="allow"))
class TaskUploadResult(TypedDict):
    """What ``POST /tasks/upload`` did, with every other field of its reply."""

    taskset_id: str
    tasks_created: int
    tasks_updated: int


class _TasksetName(BaseModel):
    taskset_id: str
    name: str


class _TasksetExport(BaseModel):
    name: str
    tasks: list[Task]

    @field_validator("tasks", mode="before")
    @classmethod
    def _as_rows(cls, records: Any) -> list[dict[str, Any]]:
        """Map each platform export record onto the portable row shape."""
        return [
            {
                "env": record.get("env"),
                "id": record.get("scenario") or "",
                "args": record.get("args") or {},
                "slug": record.get("name"),
                "validation": record.get("validation"),
                "agent_config": record.get("agent_config"),
                "columns": record.get("columns"),
                "runtime_config": record.get("runtime_config"),
                "verifier": record.get("verifier"),
            }
            for record in TypeAdapter(list[dict[str, Any]]).validate_python(records)
        ]


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


# ─── fetch ──────────────────────────────────────────────────────────────


def resolve_taskset_id(platform: PlatformClient, name_or_id: str) -> tuple[str, str]:
    """Resolve a taskset name to ``(uuid, display_name)``; uuid is "" if not found.

    A UUID is returned as given, without checking that the taskset exists.
    """
    try:
        uuid.UUID(name_or_id)
        return name_or_id, name_or_id
    except ValueError:
        pass

    try:
        found = platform.get(f"/tasksets/by-name/{quote(name_or_id, safe='')}", _TasksetName)
    except HudRequestError as e:
        if e.status_code == 404:
            return "", name_or_id
        raise
    return found.taskset_id, found.name


def fetch_taskset_tasks(
    platform: PlatformClient,
    taskset_id: str,
) -> tuple[str | None, list[Task]]:
    """Fetch a platform taskset's export, mapped to ``(display_name, [Task])``."""
    try:
        export = platform.get(f"/tasksets/{taskset_id}/export", _TasksetExport)
    except HudRequestError as e:
        if e.status_code == 404:
            return None, []
        raise
    return export.name, export.tasks


# ─── upload ─────────────────────────────────────────────────────────────


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
    """One upload item: env name + bare task id, the v6 wire identity.

    The platform resolves `(env, task_id)` against the env's latest build
    manifest and validates `args` against the task's schema.
    """
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


__all__ = [
    "SyncPlan",
    "diff",
    "fetch_taskset_tasks",
    "resolve_taskset_id",
    "task_upload_payload",
    "upload_taskset",
]
