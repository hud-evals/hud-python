"""Taskset: a named, ordered collection of concrete tasks.

Loads rows from authored Python sources, JSON/JSONL data, or the platform, and
schedules the rollout engine over them. HUD job/trace reporting lives in
:mod:`hud.eval.job`; platform persistence in :mod:`hud.eval.sync`::

    job = await Taskset("bugs", [fix_bug(difficulty=d) for d in range(5)]).run(
        agent, runtime=LocalRuntime("env.py")
    )
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

from hud.utils.platform import PlatformClient

from .job import Job, job_enter
from .rollout import rollout
from .sync import fetch_taskset_tasks, resolve_taskset_id

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from hud.agents.base import Agent

    from .rollout import Run
    from .runtime import Provider
    from .task import Task

logger = logging.getLogger("hud.eval.taskset")


def _job_name(tasks: list[Task], group: int) -> str:
    suffix = f" ({group} times)" if group > 1 else ""
    if len(tasks) == 1:
        return f"Task Run: {tasks[0].id}{suffix}"
    return f"Batch Run: {len(tasks)} tasks{suffix}"


class Taskset:
    """A named, ordered collection of :class:`~hud.eval.Task`s."""

    def __init__(
        self,
        name: str | None = None,
        tasks: Iterable[Task] = (),
        *,
        origin: str | None = None,
    ) -> None:
        self.name = name or "taskset"
        self.origin = origin
        self.tasks: dict[str, Task] = self._index_by_slug(list(tasks))

    @classmethod
    def from_file(cls, path: str | Path) -> Taskset:
        """Load a taskset from ``.py`` source, a directory, or JSON/JSONL data.

        Data rows reference envs by bare name and are runnable as-is —
        placement is an execution-time concern (``run(agent, runtime=...)``).
        """
        source = Path(path)
        if source.suffix in {".json", ".jsonl"}:
            return cls(source.stem, cls._load_tasks_json(source), origin=f"file:{source}")
        if source.suffix == ".py" or source.is_dir():
            return cls.from_module(source)
        raise ValueError(f"unsupported taskset source: {source}")

    @classmethod
    def from_module(cls, source: str | Path) -> Taskset:
        from hud.utils.modules import iter_modules

        path = Path(source).resolve()
        found = [task for module in iter_modules(path) for task in cls._scan_tasks(module)]
        return cls(
            path.stem if path.is_file() else path.name,
            found,
            origin=f"module:{path}",
        )

    @classmethod
    def from_api(cls, name: str) -> Taskset:
        """Load a platform taskset by name or id (uses ``HUD_API_KEY`` settings)."""
        platform = PlatformClient.from_settings()
        taskset_id, display = resolve_taskset_id(platform, name)
        if not taskset_id:
            raise ValueError(f"taskset not found: {name}")
        fetched_display, tasks = fetch_taskset_tasks(platform, taskset_id)
        return cls(fetched_display or display, tasks, origin=f"api:{taskset_id}")

    def to_file(self, path: str | Path) -> Path:
        """Write this taskset's portable rows to JSON or JSONL."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        suffix = target.suffix.lower()
        # Compact rows: unset metadata is omitted (defaults restore it on load).
        data = [task.model_dump(exclude_none=True) for task in self]

        if suffix == ".json":
            target.write_text(json.dumps(data, indent=2, default=str) + "\n", encoding="utf-8")
            return target
        if suffix == ".jsonl":
            lines = (json.dumps(entry, default=str) for entry in data)
            target.write_text("\n".join(lines) + ("\n" if data else ""), encoding="utf-8")
            return target
        raise ValueError(f"unsupported taskset export format: {suffix}; use .json or .jsonl")

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

        tasks: list[Task] = []
        for entry in entries:
            if not isinstance(entry, dict):
                raise ValueError(f"{path}: each task entry must be an object")
            tasks.append(Task.model_validate(entry))
        return tasks

    @staticmethod
    def _index_by_slug(tasks: list[Task]) -> dict[str, Task]:
        by_slug: dict[str, Task] = {}
        duplicates: set[str] = set()
        for task in tasks:
            slug = task.slug or task.default_slug()
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
            self.name,
            (task for slug, task in self.tasks.items() if slug in selected),
            origin=self.origin,
        )

    def exclude(self, slugs: Iterable[str]) -> Taskset:
        excluded = set(slugs)
        return Taskset(
            self.name,
            (task for slug, task in self.tasks.items() if slug not in excluded),
            origin=self.origin,
        )

    def environment_names(self) -> set[str]:
        """Return env names referenced by tasks in this taskset."""
        return {task.env for task in self}

    async def run(
        self,
        agent: Agent,
        *,
        runtime: Provider | None = None,
        group: int | None = None,
        max_concurrent: int | None = None,
        job: Job | None = None,
    ) -> Job:
        """Run every task x ``group`` with an optional concurrency cap.

        One shared (stateless) ``agent`` drives every run; ``runtime`` is the
        placement provider, called once per rollout with that rollout's task
        row — so one provider serves a mixed-env taskset and can size each
        substrate per row (left unset: HUD-hosted provisioning by env name).
        Registers one HUD job as the platform receipt and reports each run's
        trace under it — or, given an open ``job`` (:meth:`Job.start`),
        accumulates this batch into it instead, so a longer arc (a training
        session) spans many calls under one id. Returned ``job.runs``
        preserves expansion order (task-major, then group).
        """
        group = group or (job.group if job else 1)
        if group < 1:
            raise ValueError("group must be >= 1")
        if max_concurrent is not None and max_concurrent < 1:
            raise ValueError("max_concurrent must be >= 1")

        # Tasks are pure rows, shared across rollouts; the ``group`` repeats of
        # one task share a group_id (the GRPO group).
        expanded: list[tuple[Task, str]] = []
        task_list = list(self)
        for task in task_list:
            group_id = uuid.uuid4().hex
            expanded.extend((task, group_id) for _ in range(group))

        if job is None:
            job = Job(id=uuid.uuid4().hex, name=_job_name(task_list, group), group=group)
            await job_enter(job.id, name=job.name, group=group)
        job_id = job.id

        sem = asyncio.Semaphore(max_concurrent) if max_concurrent else None

        async def _one(task: Task, group_id: str) -> Run:
            if sem is None:
                return await rollout(task, agent, runtime=runtime, job_id=job_id, group_id=group_id)
            async with sem:
                return await rollout(task, agent, runtime=runtime, job_id=job_id, group_id=group_id)

        logger.info(
            "running %d rollouts (%d tasks x %d group)%s",
            len(expanded),
            len(task_list),
            group,
            f", max_concurrent={max_concurrent}" if max_concurrent else "",
        )
        job.runs.extend(await asyncio.gather(*(_one(t, gid) for t, gid in expanded)))
        return job


__all__ = ["Job", "Taskset"]
