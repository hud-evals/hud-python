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
        data = platform.get(f"/tasksets/by-name/{quote(name_or_id, safe='')}")
    except HudRequestError as e:
        if e.status_code == 404:
            return "", name_or_id
        raise
    return str(data.get("taskset_id", "")), str(data.get("name", name_or_id))


def fetch_taskset_tasks(
    platform: PlatformClient,
    taskset_id: str,
) -> tuple[str | None, list[Task]]:
    """Fetch a platform taskset's export, mapped to ``(display_name, [Task])``."""
    try:
        data = platform.get(f"/tasksets/{taskset_id}/export")
    except HudRequestError as e:
        if e.status_code == 404:
            return None, []
        raise
    display = data.get("name")
    taskset_name = display if isinstance(display, str) else None
    records = data.get("tasks")
    if not isinstance(records, list):
        return taskset_name, []
    return taskset_name, [_record_to_task(r) for r in records if isinstance(r, dict)]


def _record_to_task(record: dict[str, Any]) -> Task:
    """Map one platform export record onto the portable row shape.

    The platform may store the scenario name env-prefixed (e.g. ``"e:solve"``).
    Local task ids are always env-local (envs register scenarios unprefixed,
    and ``:`` is rejected in scenario names), so the prefix is stripped here —
    it only recovers the env name when the record omits ``env``.
    ``task_upload_payload`` re-composes it on upload.
    """
    task_id = record.get("scenario") or ""
    env_name = record.get("env")
    if isinstance(task_id, str) and ":" in task_id:
        prefix, task_id = task_id.split(":", 1)
        env_name = env_name or prefix
    return Task.model_validate(
        {
            "env": env_name,
            "id": task_id,
            "args": record.get("args") or {},
            "slug": record.get("name"),
            "validation": record.get("validation"),
            "agent_config": record.get("agent_config"),
        }
    )


# ─── upload ─────────────────────────────────────────────────────────────


def upload_taskset(
    platform: PlatformClient,
    name: str,
    tasks: list[Task],
) -> dict[str, Any]:
    """Upload tasks to a platform taskset, creating it if needed."""
    payload: dict[str, Any] = {
        "taskset_name": name,
        "tasks": [task_upload_payload(task) for task in tasks],
    }
    data = platform.post("/tasks/upload", json=payload)
    return data if isinstance(data, dict) else {}


def task_upload_payload(task: Task) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "name": task.slug or task.default_slug(),
        "env": {"name": task.env},
        "scenario": platform_task_id(task),
        "args": task.args,
    }
    if task.validation is not None:
        payload["validation"] = task.validation
    if task.agent_config:
        payload["agent_config"] = task.agent_config
    return payload


def platform_task_id(task: Task) -> str:
    """The platform's composite wire key; local ``Task.id`` is always env-local."""
    return f"{task.env}:{task.id}"


def _task_signature(task: Task) -> str:
    sig_data: dict[str, Any] = {"args": task.args or {}}
    if task.validation is not None:
        sig_data["validation"] = task.validation
    if task.agent_config:
        sig_data["agent_config"] = task.agent_config
    return f"{task.id}|" + json.dumps(
        sig_data,
        sort_keys=True,
        default=str,
        separators=(",", ":"),
    )


__all__ = [
    "SyncPlan",
    "diff",
    "fetch_taskset_tasks",
    "platform_task_id",
    "resolve_taskset_id",
    "task_upload_payload",
    "upload_taskset",
]
