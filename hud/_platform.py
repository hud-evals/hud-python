"""Private HUD platform transport helpers.

This module is intentionally not part of the public SDK surface. Public flows
stay on domain objects such as ``Environment`` and ``Taskset``; this file owns
endpoint details and wire payloads for those objects.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from urllib import parse

import httpx

if TYPE_CHECKING:
    from pathlib import Path

    from hud.client import Run
    from hud.eval.task import Task

logger = logging.getLogger("hud._platform")


@dataclass(frozen=True)
class RegistryEnvironment:
    id: str
    name: str
    version: str = ""

    @classmethod
    def from_record(cls, data: dict[str, Any]) -> RegistryEnvironment:
        env_id = data.get("id")
        if not isinstance(env_id, str) or not env_id:
            raise ValueError("registry environment record needs an id")
        display = data.get("name_display") or data.get("name") or "unnamed"
        version = data.get("latest_version") or ""
        return cls(id=env_id, name=str(display), version=str(version) if version else "")

    @property
    def short_id(self) -> str:
        return self.id[:8]

    @property
    def version_label(self) -> str:
        return f" v{self.version}" if self.version else ""


@dataclass(frozen=True)
class BuildUpload:
    upload_url: str
    build_id: str


@dataclass(frozen=True)
class PlatformClient:
    api_url: str
    headers: dict[str, str]

    @classmethod
    def from_settings(cls) -> PlatformClient:
        from hud.settings import settings

        if not settings.api_key:
            raise ValueError("HUD_API_KEY is required for HUD platform API calls")
        headers = {
            "Authorization": f"Bearer {settings.api_key}",
            "X-API-Key": settings.api_key,
        }
        return cls(settings.hud_api_url, headers)

    def get_registry_environment(self, registry_id: str) -> RegistryEnvironment | None:
        response = httpx.get(
            f"{self.api_url}/registry/envs/{registry_id}",
            headers=self.headers,
            timeout=10.0,
        )
        if response.status_code == 404:
            return None
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, dict):
            return None
        return RegistryEnvironment.from_record(data)

    def list_registry_environments(
        self,
        *,
        limit: int = 20,
        sort_by: str | None = "updated_at",
    ) -> list[RegistryEnvironment]:
        params: dict[str, Any] = {"limit": limit}
        if sort_by:
            params["sort_by"] = sort_by
        response = httpx.get(
            f"{self.api_url}/registry/envs",
            headers=self.headers,
            params=params,
            timeout=30.0,
        )
        response.raise_for_status()
        data = response.json()
        return [RegistryEnvironment.from_record(item) for item in data if isinstance(item, dict)]

    def search_registry_environments(
        self,
        name: str,
        *,
        limit: int = 5,
    ) -> list[RegistryEnvironment]:
        response = httpx.get(
            f"{self.api_url}/registry/envs",
            headers=self.headers,
            params={"search": name, "limit": limit},
            timeout=30.0,
        )
        response.raise_for_status()
        data = response.json()
        envs = [RegistryEnvironment.from_record(item) for item in data if isinstance(item, dict)]
        exact = [env for env in envs if env.name == name]
        if exact:
            return exact
        lowered = name.lower()
        return [env for env in envs if lowered in env.name.lower()]

    def resolve_registry_environments(self, ref: str) -> list[RegistryEnvironment]:
        try:
            uuid.UUID(ref)
            return [RegistryEnvironment(id=ref, name=f"{ref[:8]}...")]
        except ValueError:
            return self.search_registry_environments(ref)

    def fetch_taskset_records(self, name: str) -> tuple[str, str, list[dict[str, Any]]]:
        taskset_id, display = self.resolve_taskset_id(name)
        if not taskset_id:
            raise ValueError(f"taskset not found: {name}")
        fetched_display, records = self.fetch_task_records(taskset_id)
        return taskset_id, fetched_display or display, records

    def resolve_taskset_id(self, name_or_id: str) -> tuple[str, str]:
        try:
            uuid.UUID(name_or_id)
            return name_or_id, name_or_id
        except ValueError:
            pass

        response = httpx.get(
            f"{self.api_url}/tasks/evalset/{parse.quote(name_or_id, safe='')}",
            headers=self.headers,
            timeout=30.0,
        )
        if response.status_code == 404:
            return "", name_or_id
        response.raise_for_status()
        data = response.json()
        return str(data.get("evalset_id", "")), str(data.get("evalset_name", name_or_id))

    def fetch_task_records(self, taskset_id: str) -> tuple[str | None, list[dict[str, Any]]]:
        response = httpx.get(
            f"{self.api_url}/tasks/evalsets/{taskset_id}/tasks-by-id",
            headers=self.headers,
            timeout=30.0,
        )
        if response.status_code == 404:
            return None, []
        response.raise_for_status()
        data = response.json()
        tasks_payload = data.get("tasks") or {}
        display = data.get("evalset_name")
        taskset_name = display if isinstance(display, str) else None
        if not isinstance(tasks_payload, dict):
            return taskset_name, []
        return taskset_name, [entry for entry in tasks_payload.values() if isinstance(entry, dict)]

    def upload_taskset(
        self,
        name: str,
        tasks: list[Task],
        *,
        columns: dict[str, dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "name": name,
            "tasks": [task_upload_payload(task) for task in tasks],
        }
        if columns:
            payload["columns"] = columns
        response = httpx.post(
            f"{self.api_url}/tasks/upload",
            json=payload,
            headers=self.headers,
            timeout=60.0,
        )
        response.raise_for_status()
        data = response.json()
        return data if isinstance(data, dict) else {}

    async def create_build_upload(self) -> BuildUpload:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{self.api_url.rstrip('/')}/builds/upload-url",
                headers=self.headers,
            )
        response.raise_for_status()
        data = response.json()
        return BuildUpload(upload_url=data["upload_url"], build_id=data["build_id"])

    async def trigger_direct_build(
        self,
        *,
        build_id: str,
        name: str,
        no_cache: bool,
        registry_id: str | None = None,
        env_vars: dict[str, str] | None = None,
        build_args: dict[str, str] | None = None,
        build_secrets: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "source": "direct",
            "build_id": build_id,
            "name": name,
            "no_cache": no_cache,
        }
        if registry_id:
            payload["registry_id"] = registry_id
        if env_vars:
            payload["environment_variables"] = env_vars
        if build_args:
            payload["build_args"] = build_args
        if build_secrets:
            payload["build_secrets"] = build_secrets

        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{self.api_url.rstrip('/')}/builds/trigger",
                json=payload,
                headers=self.headers,
            )
        response.raise_for_status()
        data = response.json()
        return data if isinstance(data, dict) else {}

    async def fetch_build_status(self, build_id: str) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.get(
                f"{self.api_url.rstrip('/')}/builds/{build_id}/status",
                headers=self.headers,
            )
        response.raise_for_status()
        data = response.json()
        return data if isinstance(data, dict) else {}


async def upload_build_context(upload_url: str, tarball_path: Path) -> None:
    with tarball_path.open("rb") as file:
        tarball_data = file.read()

    async with httpx.AsyncClient(timeout=300.0) as s3_client:
        response = await s3_client.put(
            upload_url,
            content=tarball_data,
            headers={"Content-Type": "application/gzip"},
        )
        response.raise_for_status()


# ─── job / trace reporting ─────────────────────────────────────────────
#
# Backend contract:
# - ``POST /trace/job/{job_id}/enter`` — register the batch job.
# - ``POST /trace/{trace_id}/enter``   — a rollout started.
# - ``POST /trace/{trace_id}/exit``    — a rollout finished (reward / success).
#
# All three are best-effort no-ops without telemetry + an API key, so local
# runs never depend on the platform.


def _reporting_enabled() -> bool:
    from hud.settings import settings

    return bool(settings.telemetry_enabled and settings.api_key)


async def job_enter(job_id: str, *, name: str, group: int) -> None:
    """Register a batch job with the platform."""
    if not _reporting_enabled():
        return
    await _report(f"/trace/job/{job_id}/enter", {"name": name, "group": group})
    logger.info("job: https://hud.ai/jobs/%s", job_id)


async def trace_enter(trace_id: str, *, job_id: str | None, group_id: str | None) -> None:
    """Report that one rollout started."""
    if not _reporting_enabled():
        return
    await _report(f"/trace/{trace_id}/enter", {"job_id": job_id, "group_id": group_id})


async def trace_exit(run: Run) -> None:
    """Report one finished rollout (reward / success / error) from its ``Run``."""
    if not _reporting_enabled() or run.trace.trace_id is None:
        return
    await _report(
        f"/trace/{run.trace.trace_id}/exit",
        {
            "prompt": run.prompt,
            "job_id": run.job_id,
            "group_id": run.group_id,
            "reward": run.reward,
            "success": not run.trace.isError,
            "error_message": run.trace.content if run.trace.isError else None,
            "evaluation_result": run.evaluation or None,
        },
    )


async def _report(path: str, payload: dict[str, Any]) -> None:
    from hud.settings import settings
    from hud.shared import make_request

    try:
        await make_request(
            method="POST",
            url=f"{settings.hud_api_url}{path}",
            json={k: v for k, v in payload.items() if v is not None},
            api_key=settings.api_key,
        )
    except Exception as exc:
        logger.warning("platform report %s failed: %s", path, exc)


def task_upload_payload(task: Task) -> dict[str, Any]:
    env_ref = task.to_dict()["env"]
    payload: dict[str, Any] = {
        "slug": task.slug or task.default_slug(),
        "env": {"name": env_ref["name"]} if env_ref.get("name") else {},
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
    env_ref = task.to_dict()["env"]
    env_name = env_ref.get("name")
    if env_name and ":" not in task.id:
        return f"{env_name}:{task.id}"
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
