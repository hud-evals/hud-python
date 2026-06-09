"""Taskset: a named, ordered collection of concrete tasks.

Launches each task, lets ``agent(run)`` fill ``run.trace``, grades it, and
gathers the :class:`Run`s — with optional GRPO grouping + a concurrency cap. HUD
job/trace reporting lives in :mod:`hud.eval.job`::

    job = await Taskset.from_tasks("bugs", [fix_bug(difficulty=d) for d in range(5)]).run(agent)
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from hud.client import Run

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from hud.agents.base import Agent

    from .task import Task

logger = logging.getLogger("hud.eval.taskset")


async def _rollout(
    task: Task,
    agent: Agent,
    *,
    job_id: str | None = None,
    group_id: str | None = None,
) -> Run:
    """Drive one task to a graded :class:`Run` (the rollout atom).

    Launch the env, let ``agent(run)`` fill ``run.trace``, and grade it on exit
    (``run.reward``). The rollout is wrapped in :func:`hud.eval.job.trace`,
    which binds the per-rollout ``trace_id`` into the trace context (so ``@instrument``
    spans upload to it) and reports the trace to HUD. A launch/connect failure is
    isolated into a failed ``Run`` so one bad rollout never collapses a batch.
    """
    from hud.eval.job import trace as report_trace

    trace_id = uuid.uuid4().hex
    async with report_trace(trace_id, job_id=job_id, group_id=group_id) as recorded:
        try:
            async with task as run:
                await agent(run)
            run.trace.trace_id = trace_id
        except (TimeoutError, asyncio.CancelledError, KeyboardInterrupt):
            raise
        except Exception as exc:
            logger.warning("rollout failed: %s", exc)
            run = Run.failed(str(exc), trace_id=trace_id)
        run.job_id = job_id
        run.group_id = group_id
        recorded.append(run)
    return run


def _job_name(tasks: list[Task], group: int) -> str:
    suffix = f" ({group} times)" if group > 1 else ""
    if len(tasks) == 1:
        return f"Task Run: {tasks[0].id}{suffix}"
    return f"Batch Run: {len(tasks)} tasks{suffix}"


@dataclass(slots=True)
class Job:
    """One execution of a taskset."""

    id: str
    name: str
    runs: list[Run]
    group: int = 1

    def __len__(self) -> int:
        return len(self.runs)

    def __iter__(self) -> Iterator[Run]:
        return iter(self.runs)

    def __getitem__(self, index: int) -> Run:
        return self.runs[index]


@dataclass(slots=True)
class SyncPlan:
    """Diff between a local taskset and a remote taskset."""

    to_create: list[Task] = field(default_factory=list)
    to_update: list[Task] = field(default_factory=list)
    unchanged: list[Task] = field(default_factory=list)
    remote_only: list[Task] = field(default_factory=list)
    taskset_name: str = ""
    api_url: str | None = None
    headers: dict[str, str] = field(default_factory=dict)
    column_definitions: dict[str, dict[str, Any]] | None = None

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

    def apply(
        self,
        *,
        taskset_name: str | None = None,
        api_url: str | None = None,
        headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        import httpx

        name = taskset_name or self.taskset_name
        target_url = api_url or self.api_url
        target_headers = headers or self.headers
        if not name:
            raise ValueError("taskset name is required to apply a sync plan")
        if not target_url:
            raise ValueError("api_url is required to apply a sync plan")
        payload: dict[str, Any] = {
            "name": name,
            "tasks": [_task_upload_payload(task) for task in self.to_apply],
        }
        if self.column_definitions:
            payload["columns"] = self.column_definitions
        response = httpx.post(
            f"{target_url}/tasks/upload",
            json=payload,
            headers=target_headers,
            timeout=60.0,
        )
        response.raise_for_status()
        return response.json()


class Taskset:
    """A named, ordered collection of :class:`~hud.eval.Task`s."""

    def __init__(
        self,
        tasks: Iterable[Task] = (),
        *,
        name: str | None = None,
        origin: str | None = None,
    ) -> None:
        self.name = name or "taskset"
        self.origin = origin
        self.tasks: list[Task] = list(tasks)
        self._by_slug = self._index_by_slug(self.tasks)

    @classmethod
    def from_tasks(cls, name: str, tasks: Iterable[Task]) -> Taskset:
        return cls(tasks, name=name)

    @classmethod
    def from_file(cls, path: str | Path) -> Taskset:
        source = Path(path)
        if source.suffix in {".json", ".jsonl"}:
            return cls(cls._load_tasks_json(source), name=source.stem, origin=f"file:{source}")
        if source.suffix == ".py" or source.is_dir():
            return cls.from_module(source)
        raise ValueError(f"unsupported taskset source: {source}")

    @classmethod
    def from_module(cls, source: str | Path) -> Taskset:
        from .sandbox import load_module

        path = Path(source).resolve()
        if path.is_file() and path.suffix == ".py":
            return cls(
                cls._scan_tasks(load_module(path)),
                name=path.stem,
                origin=f"module:{path}",
            )
        if path.is_dir():
            found: list[Task] = []
            for py_file in sorted(path.glob("*.py")):
                if py_file.stem in {"conftest", "setup", "__init__", "__main__"}:
                    continue
                try:
                    found.extend(cls._scan_tasks(load_module(py_file)))
                except ImportError:
                    logger.debug("skipping %s during taskset collection", py_file.name)
            return cls(found, name=path.name, origin=f"module:{path}")
        raise FileNotFoundError(f"Source not found: {source}")

    @classmethod
    def from_package(cls, package: str) -> Taskset:
        import importlib
        import pkgutil

        module = importlib.import_module(package)
        paths = getattr(module, "__path__", None)
        if paths is None:
            return cls.from_module(Path(module.__file__ or ""))

        found: list[Task] = []
        for info in pkgutil.iter_modules(paths, package + "."):
            if not info.ispkg:
                continue
            mod = importlib.import_module(info.name)
            found.extend(cls._scan_tasks(mod))
        return cls(found, name=package, origin=f"package:{package}")

    @classmethod
    def from_api(cls, name: str) -> Taskset:
        from hud.settings import settings

        if not settings.api_key:
            raise ValueError("HUD_API_KEY is required to load tasksets from the API")
        headers = {"Authorization": f"Bearer {settings.api_key}"}
        taskset_id, display, _created = _resolve_taskset_id(
            name,
            settings.hud_api_url,
            headers,
            create=False,
        )
        if not taskset_id:
            raise ValueError(f"taskset not found: {name}")
        remote = _fetch_remote_tasks(taskset_id, settings.hud_api_url, headers)
        return cls(
            (_remote_task_to_task(t) for t in remote),
            name=display,
            origin=f"api:{taskset_id}",
        )

    @classmethod
    def from_remote_tasks(cls, name: str, tasks: Iterable[dict[str, Any]]) -> Taskset:
        """Build a taskset from platform task records."""
        return cls(
            (_remote_task_to_task(task) for task in tasks),
            name=name,
            origin=f"api:{name}",
        )

    @classmethod
    def from_source(cls, source: str | Path) -> Taskset:
        path = Path(source)
        if path.exists():
            return cls.from_file(path)
        return cls.from_api(str(source))

    @staticmethod
    def _scan_tasks(module: Any) -> list[Task]:
        from .task import Task

        tasks: list[Task] = []
        for name in dir(module):
            if name.startswith("_"):
                continue
            value = getattr(module, name, None)
            if isinstance(value, Task):
                tasks.append(value)
            elif isinstance(value, Taskset):
                tasks.extend(value.tasks)
            elif isinstance(value, (list, tuple)):
                tasks.extend(item for item in value if isinstance(item, Task))
        return tasks

    @staticmethod
    def _load_tasks_json(path: Path) -> list[Task]:
        from .task import Task

        text = path.read_text(encoding="utf-8")
        if path.suffix == ".jsonl":
            entries = [json.loads(line) for line in text.splitlines() if line.strip()]
        else:
            data = json.loads(text)
            if isinstance(data, dict):
                entries = [data]
            elif isinstance(data, list):
                entries = data
            else:
                raise ValueError(f"{path}: expected a JSON object, list, or JSONL file")

        base = path.resolve().parent
        tasks: list[Task] = []
        for entry in entries:
            if not isinstance(entry, dict):
                raise ValueError(f"{path}: each task entry must be an object")
            env_ref = entry.get("env")
            if isinstance(env_ref, dict) and env_ref.get("type") == "module":
                module = env_ref.get("module")
                if isinstance(module, str) and not Path(module).is_absolute():
                    entry = {**entry, "env": {**env_ref, "module": str((base / module).resolve())}}
            tasks.append(Task.from_dict(entry))
        return tasks

    @staticmethod
    def _index_by_slug(tasks: list[Task]) -> dict[str, Task]:
        by_slug: dict[str, Task] = {}
        duplicates: set[str] = set()
        for task in tasks:
            slug = _task_slug(task)
            if slug in by_slug:
                duplicates.add(slug)
            by_slug[slug] = task
        if duplicates:
            raise ValueError(f"duplicate task slugs: {', '.join(sorted(duplicates))}")
        return by_slug

    def __len__(self) -> int:
        return len(self.tasks)

    def __iter__(self) -> Iterator[Task]:
        return iter(self.tasks)

    def __getitem__(self, slug: str) -> Task:
        return self._by_slug[slug]

    def filter(self, slugs: Iterable[str]) -> Taskset:
        selected = set(slugs)
        return Taskset(
            (task for task in self.tasks if _task_slug(task) in selected),
            name=self.name,
            origin=self.origin,
        )

    def exclude(self, slugs: Iterable[str]) -> Taskset:
        excluded = set(slugs)
        return Taskset(
            (task for task in self.tasks if _task_slug(task) not in excluded),
            name=self.name,
            origin=self.origin,
        )

    def diff(
        self,
        remote: Taskset,
        *,
        api_url: str | None = None,
        headers: dict[str, str] | None = None,
    ) -> SyncPlan:
        remote_by_slug = {_task_slug(task): task for task in remote.tasks}
        to_create: list[Task] = []
        to_update: list[Task] = []
        unchanged: list[Task] = []

        for task in self.tasks:
            slug = _task_slug(task)
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
            taskset_name=remote.name or self.name,
            api_url=api_url,
            headers=headers or {},
            column_definitions=_build_column_definitions(self.tasks),
        )

    def sync_to(
        self,
        remote: Taskset,
        *,
        dry_run: bool = False,
        api_url: str | None = None,
        headers: dict[str, str] | None = None,
    ) -> SyncPlan:
        plan = self.diff(remote, api_url=api_url, headers=headers)
        if not dry_run:
            plan.apply()
        return plan

    async def run(
        self,
        agent: Any,
        *,
        group: int = 1,
        max_concurrent: int | None = None,
    ) -> Job:
        """Run every task x ``group`` with an optional concurrency cap.

        One shared (stateless) ``agent`` drives every rollout; each rollout gets a
        fresh env (via the task) and its own :class:`Run`. Registers one HUD job
        for the batch and reports each rollout's trace under it. Returns a Job whose
        runs preserve expansion order (task-major, then group).
        """
        if group < 1:
            raise ValueError("group must be >= 1")
        from hud.eval.job import job_enter

        # Fresh Task per rollout (the Task CM holds per-enter state); the ``group``
        # repeats of one task share a group_id (the GRPO group).
        expanded: list[tuple[Task, str]] = []
        for task in self.tasks:
            group_id = uuid.uuid4().hex
            expanded.extend((replace(task), group_id) for _ in range(group))

        job_id = uuid.uuid4().hex
        name = _job_name(self.tasks, group)
        await job_enter(job_id, name=name, group=group)

        sem = asyncio.Semaphore(max_concurrent) if max_concurrent else None

        async def _one(task: Task, group_id: str) -> Run:
            if sem is None:
                return await _rollout(task, agent, job_id=job_id, group_id=group_id)
            async with sem:
                return await _rollout(task, agent, job_id=job_id, group_id=group_id)

        logger.info(
            "running %d rollouts (%d tasks x %d group)%s",
            len(expanded),
            len(self.tasks),
            group,
            f", max_concurrent={max_concurrent}" if max_concurrent else "",
        )
        runs = list(await asyncio.gather(*(_one(t, gid) for t, gid in expanded)))
        return Job(id=job_id, name=name, runs=runs, group=group)


def _resolve_taskset_id(
    name_or_id: str,
    api_url: str,
    headers: dict[str, str],
    *,
    create: bool,
) -> tuple[str, str, bool]:
    import uuid as _uuid
    from urllib import parse

    import httpx

    try:
        _uuid.UUID(name_or_id)
        return name_or_id, name_or_id, False
    except ValueError:
        pass

    if create:
        response = httpx.post(
            f"{api_url}/tasks/resolve-evalset",
            json={"name": name_or_id},
            headers=headers,
            timeout=30.0,
        )
        response.raise_for_status()
        data = response.json()
        return (
            str(data.get("evalset_id", "")),
            str(data.get("name", name_or_id)),
            bool(data.get("created", False)),
        )

    response = httpx.get(
        f"{api_url}/tasks/evalset/{parse.quote(name_or_id, safe='')}",
        headers=headers,
        timeout=30.0,
    )
    if response.status_code == 404:
        return "", name_or_id, False
    response.raise_for_status()
    data = response.json()
    return str(data.get("evalset_id", "")), str(data.get("evalset_name", name_or_id)), False


def _fetch_remote_tasks(
    taskset_id: str,
    api_url: str,
    headers: dict[str, str],
) -> list[dict[str, Any]]:
    import httpx

    response = httpx.get(
        f"{api_url}/tasks/evalsets/{taskset_id}/tasks-by-id",
        headers=headers,
        timeout=30.0,
    )
    if response.status_code == 404:
        return []
    response.raise_for_status()
    data = response.json()
    tasks_payload = data.get("tasks") or {}
    if not isinstance(tasks_payload, dict):
        return []
    return [entry for entry in tasks_payload.values() if isinstance(entry, dict)]


def _remote_task_to_task(remote: dict[str, Any]) -> Task:
    from .task import Task

    env_data = remote.get("env")
    env_ref = env_data if isinstance(env_data, dict) else {"type": "hud", "name": ""}
    if "type" not in env_ref:
        env_ref = {"type": "hud", "name": env_ref.get("name") or ""}
    return Task.from_dict(
        {
            "env": env_ref,
            "task": remote.get("scenario") or remote.get("task") or remote.get("id"),
            "args": remote.get("args") or {},
            "slug": remote.get("slug") or remote.get("external_id"),
            "validation": remote.get("validation"),
            "agent_config": remote.get("agent_config"),
            "columns": remote.get("column_values"),
        }
    )


def _short_task_id(task_id: str) -> str:
    return task_id.rsplit(":", 1)[-1] if ":" in task_id else task_id


def _task_slug(task: Task) -> str:
    return task.slug or task.default_slug()


def _task_env_ref(task: Task) -> dict[str, Any]:
    return task.to_dict()["env"]


def _platform_task_id(task: Task) -> str:
    env_ref = _task_env_ref(task)
    env_name = env_ref.get("name")
    if env_name and ":" not in task.id:
        return f"{env_name}:{task.id}"
    return task.id


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


def _task_upload_payload(task: Task) -> dict[str, Any]:
    env_ref = _task_env_ref(task)
    payload: dict[str, Any] = {
        "slug": _task_slug(task),
        "env": {"name": env_ref["name"]} if env_ref.get("name") else {},
        "scenario": _platform_task_id(task),
        "args": task.args,
    }
    if task.validation is not None:
        payload["validation"] = task.validation
    if task.agent_config:
        payload["agent_config"] = task.agent_config
    if task.columns:
        payload["column_values"] = task.columns
    return payload


def _infer_column_type(values: list[Any]) -> str:
    non_none = [v for v in values if v is not None]
    if not non_none:
        return "text"
    if any(isinstance(v, list) for v in non_none):
        return "multi-select"
    if all(isinstance(v, (int, float)) for v in non_none):
        return "number"
    return "text"


def _build_column_definitions(tasks: list[Task]) -> dict[str, dict[str, Any]] | None:
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
            for v in vals:
                if isinstance(v, list):
                    all_opts.update(str(item) for item in v)
                elif v is not None:
                    all_opts.add(str(v))
            col_def["options"] = sorted(all_opts)
        definitions[col_name] = col_def
    return definitions


__all__ = ["Job", "SyncPlan", "Taskset"]
