"""Platform persistence for tasksets: diff plans and the fetch/upload wire format.

Taskset endpoints ("evalsets" on the backend) and the upload payload shape.
Transport (auth, retries, errors) is :mod:`hud.utils.platform`; the shapes and
the local-vs-remote :func:`diff` live here, out of the collection itself.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any
from urllib.parse import quote

from hud.utils.exceptions import HudRequestError

from .task import Task

if TYPE_CHECKING:
    from hud.utils.platform import PlatformClient

    from .taskset import Taskset


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
    """Resolve a taskset name to ``(uuid, display_name)``; uuid is "" if not found."""
    try:
        uuid.UUID(name_or_id)
        return name_or_id, name_or_id
    except ValueError:
        pass

    try:
        data = platform.get(f"/tasks/evalset/{quote(name_or_id, safe='')}")
    except HudRequestError as e:
        if e.status_code == 404:
            return "", name_or_id
        raise
    return str(data.get("evalset_id", "")), str(data.get("evalset_name", name_or_id))


def fetch_taskset_tasks(
    platform: PlatformClient,
    taskset_id: str,
) -> tuple[str | None, list[Task]]:
    """Fetch a platform taskset's records, mapped to ``(display_name, [Task])``."""
    try:
        data = platform.get(f"/tasks/evalsets/{taskset_id}/tasks-by-id")
    except HudRequestError as e:
        if e.status_code == 404:
            return None, []
        raise
    tasks_payload = data.get("tasks") or {}
    display = data.get("evalset_name")
    taskset_name = display if isinstance(display, str) else None
    if not isinstance(tasks_payload, dict):
        return taskset_name, []
    records = [entry for entry in tasks_payload.values() if isinstance(entry, dict)]
    return taskset_name, [_record_to_task(record) for record in records]


def _record_to_task(record: dict[str, Any]) -> Task:
    """Map one platform task record onto the portable row shape.

    Platform records key the task id as ``scenario`` (env-prefixed, e.g.
    ``"e:solve"``) and may omit the env block — the prefix recovers the env
    name in that case.
    """
    task_id = record.get("scenario") or record.get("task") or record.get("id") or ""
    env_data = record.get("env")
    env_name = env_data.get("name") if isinstance(env_data, dict) else None
    if not env_name and isinstance(task_id, str) and ":" in task_id:
        env_name = task_id.split(":", 1)[0]
    return Task.from_dict(
        {
            "env": {"name": env_name},
            "task": task_id,
            "args": record.get("args") or {},
            "slug": record.get("slug") or record.get("external_id"),
            "validation": record.get("validation"),
            "agent_config": record.get("agent_config"),
            "columns": record.get("column_values"),
        }
    )


# ─── upload ─────────────────────────────────────────────────────────────


def upload_taskset(
    platform: PlatformClient,
    name: str,
    tasks: list[Task],
    *,
    columns: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Upload tasks to a platform taskset, creating it if needed."""
    payload: dict[str, Any] = {
        "name": name,
        "tasks": [task_upload_payload(task) for task in tasks],
    }
    if columns:
        payload["columns"] = columns
    data = platform.post("/tasks/upload", json=payload)
    return data if isinstance(data, dict) else {}


def task_upload_payload(task: Task) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "slug": task.slug or task.default_slug(),
        "env": {"name": task.env.name},
        "scenario": platform_task_id(task),
        "args": task.args,
    }
    if task.validation is not None:
        payload["validation"] = task.validation
    if task.agent_config:
        payload["agent_config"] = task.agent_config
    if task.columns:
        payload["column_values"] = task.columns
    return payload


def platform_task_id(task: Task) -> str:
    if ":" not in task.id:
        return f"{task.env.name}:{task.id}"
    return task.id


def taskset_column_definitions(tasks: list[Task]) -> dict[str, dict[str, Any]] | None:
    values_by_col: dict[str, list[Any]] = {}
    for task in tasks:
        if not task.columns:
            continue
        for col_name, col_val in task.columns.items():
            values_by_col.setdefault(col_name, []).append(col_val)

    if not values_by_col:
        return None

    definitions: dict[str, dict[str, Any]] = {}
    for col_name, vals in values_by_col.items():
        col_type = _infer_column_type(vals)
        col_def: dict[str, Any] = {"type": col_type}
        if col_type == "multi-select":
            all_opts: set[str] = set()
            for value in vals:
                if isinstance(value, list):
                    all_opts.update(str(item) for item in value)
                elif value is not None:
                    all_opts.add(str(value))
            col_def["options"] = sorted(all_opts)
        definitions[col_name] = col_def
    return definitions


def _infer_column_type(values: list[Any]) -> str:
    non_none = [value for value in values if value is not None]
    if not non_none:
        return "text"
    if any(isinstance(value, list) for value in non_none):
        return "multi-select"
    if all(isinstance(value, (int, float)) for value in non_none):
        return "number"
    return "text"


def _task_signature(task: Task) -> str:
    sig_data: dict[str, Any] = {"args": task.args or {}}
    if task.validation is not None:
        sig_data["validation"] = task.validation
    if task.agent_config:
        sig_data["agent_config"] = task.agent_config
    if task.columns:
        sig_data["columns"] = task.columns
    return f"{_short_task_id(task.id)}|" + json.dumps(
        sig_data,
        sort_keys=True,
        default=str,
        separators=(",", ":"),
    )


def _short_task_id(task_id: str) -> str:
    return task_id.rsplit(":", 1)[-1] if ":" in task_id else task_id


__all__ = [
    "SyncPlan",
    "diff",
    "fetch_taskset_tasks",
    "platform_task_id",
    "resolve_taskset_id",
    "task_upload_payload",
    "taskset_column_definitions",
    "upload_taskset",
]
