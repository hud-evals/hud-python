"""Taskset: a named, ordered collection of concrete tasks.

Launches each task, lets ``agent(run)`` fill ``run.trace``, grades it, and
returns a :class:`Job` receipt containing the resulting :class:`Run`s. HUD
job/trace reporting lives in :mod:`hud._platform`::

    job = await Taskset.from_tasks("bugs", [fix_bug(difficulty=d) for d in range(5)]).run(agent)
"""

from __future__ import annotations

import asyncio
import csv
import json
import logging
import uuid
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from hud.client import Run

from .job import Job

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
    (``run.reward``). The per-rollout ``trace_id`` is bound into the trace
    context (so ``@instrument`` spans attribute to it — always, even with
    telemetry off, for local training) and the trace is reported to HUD. A
    launch/connect failure is isolated into a failed ``Run`` so one bad rollout
    never collapses a batch.
    """
    from hud._platform import trace_enter, trace_exit
    from hud.telemetry.context import set_trace_context

    trace_id = uuid.uuid4().hex
    with set_trace_context(trace_id):
        await trace_enter(trace_id, job_id=job_id, group_id=group_id)
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
        await trace_exit(run)
    return run


def _job_name(tasks: list[Task], group: int) -> str:
    suffix = f" ({group} times)" if group > 1 else ""
    if len(tasks) == 1:
        return f"Task Run: {tasks[0].id}{suffix}"
    return f"Batch Run: {len(tasks)} tasks{suffix}"


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
        self.tasks: dict[str, Task] = self._index_by_slug(list(tasks))

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
        return cls._from_module(source, preloaded={})

    @classmethod
    def _from_module(cls, source: str | Path, *, preloaded: dict[Path, Any]) -> Taskset:
        from .sandbox import load_module

        path = Path(source).resolve()
        if path.is_file() and path.suffix == ".py":
            module = preloaded.get(path) or load_module(path)
            return cls(
                cls._scan_tasks(module),
                name=path.stem,
                origin=f"module:{path}",
            )
        if path.is_dir():
            found: list[Task] = []
            for py_file in sorted(path.glob("*.py")):
                if py_file.stem in {"conftest", "setup", "__init__", "__main__"}:
                    continue
                try:
                    module = preloaded.get(py_file.resolve()) or load_module(py_file)
                    found.extend(cls._scan_tasks(module))
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
        """Load a platform taskset by name or id (uses ``HUD_API_KEY`` settings)."""
        from hud._platform import PlatformClient

        taskset_id, display, remote = PlatformClient.from_settings().fetch_taskset_records(name)
        return cls(
            (_remote_task_to_task(t) for t in remote),
            name=display,
            origin=f"api:{taskset_id}",
        )

    def to_file(self, path: str | Path) -> Path:
        """Write this taskset to JSON, JSONL, or CSV."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        suffix = target.suffix.lower()
        data = [task.to_dict() for task in self]

        if suffix == ".json":
            target.write_text(json.dumps(data, indent=2, default=str) + "\n", encoding="utf-8")
            return target
        if suffix == ".jsonl":
            lines = (json.dumps(entry, default=str) for entry in data)
            target.write_text("\n".join(lines) + ("\n" if data else ""), encoding="utf-8")
            return target
        if suffix == ".csv":
            self._write_csv(target, data)
            return target
        raise ValueError(f"unsupported taskset export format: {suffix}; use .json, .jsonl, or .csv")

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
                tasks.extend(value)
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
    def _write_csv(path: Path, entries: list[dict[str, Any]]) -> None:
        arg_keys = sorted(
            {
                key
                for entry in entries
                for key in (entry.get("args") or {})
                if isinstance(entry.get("args"), dict)
            }
        )
        col_keys = sorted(
            {
                key
                for entry in entries
                for key in (entry.get("columns") or {})
                if isinstance(entry.get("columns"), dict)
            }
        )
        fieldnames = [
            "slug",
            "task",
            "env",
            *[f"arg:{key}" for key in arg_keys],
            *[f"col:{key}" for key in col_keys],
        ]
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for entry in entries:
                env_value = entry.get("env")
                args_value = entry.get("args")
                cols_value = entry.get("columns")
                env_ref: dict[str, Any] = env_value if isinstance(env_value, dict) else {}
                args: dict[str, Any] = args_value if isinstance(args_value, dict) else {}
                cols: dict[str, Any] = cols_value if isinstance(cols_value, dict) else {}
                row: dict[str, Any] = {
                    "slug": entry.get("slug") or "",
                    "task": entry.get("task") or "",
                    "env": env_ref.get("name") or env_ref.get("url") or "",
                }
                for key in arg_keys:
                    value = args.get(key)
                    row[f"arg:{key}"] = (
                        json.dumps(value, default=str) if isinstance(value, (dict, list)) else value
                    )
                for key in col_keys:
                    value = cols.get(key)
                    row[f"col:{key}"] = (
                        json.dumps(value, default=str) if isinstance(value, (dict, list)) else value
                    )
                writer.writerow(row)

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
        return iter(self.tasks.values())

    def __getitem__(self, slug: str) -> Task:
        return self.tasks[slug]

    def items(self) -> Iterator[tuple[str, Task]]:
        return iter(self.tasks.items())

    def filter(self, slugs: Iterable[str]) -> Taskset:
        selected = set(slugs)
        return Taskset(
            (task for slug, task in self.tasks.items() if slug in selected),
            name=self.name,
            origin=self.origin,
        )

    def exclude(self, slugs: Iterable[str]) -> Taskset:
        excluded = set(slugs)
        return Taskset(
            (task for slug, task in self.tasks.items() if slug not in excluded),
            name=self.name,
            origin=self.origin,
        )

    def environment_names(self) -> set[str]:
        """Return HUD environment names referenced by tasks in this taskset."""
        names: set[str] = set()
        for task in self:
            env_name = task.to_dict()["env"].get("name")
            if isinstance(env_name, str) and env_name:
                names.add(env_name)
        return names

    def diff(self, remote: Taskset) -> SyncPlan:
        remote_by_slug = dict(remote.tasks)
        to_create: list[Task] = []
        to_update: list[Task] = []
        unchanged: list[Task] = []

        for slug, task in self.tasks.items():
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
        )

    async def run(
        self,
        agent: Any,
        *,
        group: int = 1,
        max_concurrent: int | None = None,
    ) -> Job:
        """Run every task x ``group`` with an optional concurrency cap.

        One shared (stateless) ``agent`` drives every run; each run gets a fresh
        env via the task. Registers one HUD job as the batch/platform receipt and
        reports each run's trace under it. Returned ``job.runs`` preserves
        expansion order (task-major, then group).
        """
        if group < 1:
            raise ValueError("group must be >= 1")
        from hud._platform import job_enter

        # Fresh Task per rollout (the Task CM holds per-enter state); the ``group``
        # repeats of one task share a group_id (the GRPO group).
        expanded: list[tuple[Task, str]] = []
        task_list = list(self)
        for task in task_list:
            group_id = uuid.uuid4().hex
            expanded.extend((replace(task), group_id) for _ in range(group))

        job_id = uuid.uuid4().hex
        name = _job_name(task_list, group)
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
            len(task_list),
            group,
            f", max_concurrent={max_concurrent}" if max_concurrent else "",
        )
        runs = list(await asyncio.gather(*(_one(t, gid) for t, gid in expanded)))
        return Job(id=job_id, name=name, runs=runs, group=group)


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


def _task_slug(task: Task) -> str:
    return task.slug or task.default_slug()


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


__all__ = ["Job", "SyncPlan", "Taskset"]
