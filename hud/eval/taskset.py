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

from hud.telemetry import flush
from hud.utils.platform import PlatformClient

from .job import Job, job_enter
from .run import rollout, vec_rollout
from .runtime import HostedRuntime, HUDRuntime, LocalRuntime, _declared_env, _declared_names
from .sync import fetch_taskset_tasks, resolve_taskset_id

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from hud.agents.base import Agent

    from .run import Run
    from .runtime import Provider
    from .task import Task

logger = logging.getLogger("hud.eval.taskset")


def _job_name(taskset_name: str, tasks: list[Task], group: int) -> str:
    suffix = f" ({group} times)" if group > 1 else ""
    if len(tasks) == 1:
        return f"{tasks[0].id}{suffix}"
    return f"{taskset_name} ({len(tasks)} tasks){suffix}"


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

    @property
    def api_id(self) -> str | None:
        """The platform taskset id when loaded via :meth:`from_api`, else None.

        Threaded into the job so a remote run of a synced taskset links to it;
        ad-hoc/file/module tasksets have none and create no taskset.
        """
        if self.origin and self.origin.startswith("api:"):
            return self.origin[len("api:") :]
        return None

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

    def _resolve_placement(self) -> Provider | HUDRuntime:
        if self.origin and self.origin.startswith("module:"):
            # The origin claims the rows only if it actually declares their
            # envs (a tasks-only module importing its envs from elsewhere
            # does not) — and it serves as the exact path, so a same-named
            # variant in a sibling file is never dragged in.
            source = Path(self.origin[len("module:") :])
            if self.environment_names() <= _declared_names(source):
                return LocalRuntime(source)
        if self.origin and self.origin.startswith("api:"):
            return HUDRuntime()
        declared = {name: _declared_env(name) for name in self.environment_names()}
        if declared and all(declared.values()):
            providers = {
                name: LocalRuntime(env) for name, env in declared.items() if env is not None
            }
            logger.info(
                "no runtime given: serving %s fresh from their declaring modules",
                ", ".join(sorted(providers)),
            )
            return lambda task: providers[task.env](task)
        missing = sorted(name for name, env in declared.items() if env is None)
        raise ValueError(
            f"no placement for env(s) {', '.join(missing) or '<none>'}: pass runtime= — "
            'LocalRuntime("env.py") (a source file), LocalRuntime(env) (a live env), '
            "LocalRuntime(build) (a (task) -> Environment constructor), Runtime(url) "
            "(a served substrate), or HUDRuntime() (your deployed env). A row taken "
            "from a loaded taskset keeps its placement when run through it: "
            'taskset.filter(["slug"]).run(...)'
        )

    async def run(
        self,
        agent: Agent,
        *,
        runtime: Provider | HostedRuntime | None = None,
        group: int | None = None,
        num_envs: int | None = None,
        max_concurrent: int | None = None,
        job: Job | None = None,
        rollout_timeout: float | None = None,
    ) -> Job:
        """Run every task x ``group`` with an optional concurrency cap.

        One shared (stateless) ``agent`` drives every run. ``runtime`` is the
        placement: a :class:`~hud.eval.runtime.Provider` (the env served
        somewhere, the agent loop driven here by :func:`~hud.eval.run.rollout`),
        or :class:`~hud.eval.runtime.HostedRuntime` to run each rollout remotely
        on the platform. Left unset, what is already known decides: a
        taskset loaded from local ``.py`` source serves that source's
        directory; a platform taskset runs on the platform; rows naming envs
        declared in imported modules serve each fresh from its file; anything
        else raises, naming the forms to pass. One provider serves a
        mixed-env
        taskset and can size each substrate per row. Registers one HUD job as
        the platform receipt and reports each run's trace under it — or, given
        an open ``job`` (:meth:`Job.start`), accumulates this batch into it
        instead, so a longer arc (a training session) spans many calls under
        one id. Returned ``job.runs`` preserves expansion order (task-major,
        then group).

        ``num_envs`` selects vectorized execution (:func:`~hud.eval.run.vec_rollout`):
        each task instance runs one *vectorized* env whose N slots (each its own
        seeded perturbation) become N graded traces sharing a group_id. Distinct
        from ``group`` — statistical repeats as separate instances — and they
        compose: ``group=3, num_envs=5`` is 3 instances x 5 traces per task.
        ``max_concurrent`` always counts instances. Requires a self-managed
        runtime and a ``drive``-capable agent (e.g. ``RobotAgent``).

        ``rollout_timeout`` is a hard per-rollout wall-clock cap (seconds) for the
        local (Provider) path: a rollout that exceeds it is cancelled and recorded
        as a failed/errored run so one wedged rollout (e.g. a stuck sampling
        stream) cannot stall the whole batch. ``HUDRuntime`` carries its own
        ``run_timeout`` instead.
        """
        group = group or (job.group if job else 1)
        if group < 1:
            raise ValueError("group must be >= 1")
        if max_concurrent is not None and max_concurrent < 1:
            raise ValueError("max_concurrent must be >= 1")
        if num_envs is not None and num_envs < 1:
            raise ValueError("num_envs must be >= 1")

        # Tasks are pure rows, shared across rollouts. The ``group`` repeats of one
        # task share a group_id (the GRPO group) — except under ``num_envs``, where
        # each instance's N slot-traces are their own group.
        expanded: list[tuple[Task, str]] = []
        task_list = list(self)
        for task in task_list:
            group_id = uuid.uuid4().hex
            expanded.extend(
                (task, uuid.uuid4().hex if num_envs else group_id) for _ in range(group)
            )

        if job is None:
            job = Job(
                id=uuid.uuid4().hex,
                name=_job_name(self.name, task_list, group),
                group=group,
                taskset_id=self.api_id,
            )
            await job_enter(job.id, name=job.name, group=group, taskset_id=self.api_id)
        job_id = job.id

        # Placement is chosen once for the batch: HostedRuntime delegates the
        # whole rollout to the platform, anything else is a Provider driven
        # locally by rollout(). No runtime: what the taskset or this process
        # already knows decides (rows never carry placement) — a loaded
        # taskset runs where it came from; rows naming envs declared in
        # imported modules serve each fresh from its file; anything else is
        # an error naming the forms to pass.
        # An empty taskset schedules nothing, so it needs no placement.
        placement = runtime if runtime is not None or not task_list else self._resolve_placement()
        if num_envs is not None and isinstance(placement, HostedRuntime):
            raise ValueError("num_envs (vectorized rollouts) requires a self-managed runtime")
        sem = asyncio.Semaphore(max_concurrent) if max_concurrent else None

        async def _run(task: Task, group_id: str) -> list[Run]:
            assert placement is not None  # only reached when tasks were expanded
            if isinstance(placement, HostedRuntime):
                return [await placement.run(task, agent, job_id=job_id, group_id=group_id)]
            if num_envs is not None:  # vectorized: one instance -> num_envs graded runs
                return await vec_rollout(
                    task,
                    agent,
                    runtime=placement,
                    num_envs=num_envs,
                    job_id=job_id,
                    group_id=group_id,
                    rollout_timeout=rollout_timeout,
                )
            return [
                await rollout(
                    task,
                    agent,
                    runtime=placement,
                    job_id=job_id,
                    group_id=group_id,
                    rollout_timeout=rollout_timeout,
                )
            ]

        async def _one(task: Task, group_id: str) -> list[Run]:
            if sem is None:
                return await _run(task, group_id)
            async with sem:
                return await _run(task, group_id)

        logger.info(
            "running %d rollouts (%d tasks x %d group)%s%s",
            len(expanded),
            len(task_list),
            group,
            f" x {num_envs} envs" if num_envs else "",
            f", max_concurrent={max_concurrent}" if max_concurrent else "",
        )
        waves = await asyncio.gather(*(_one(t, gid) for t, gid in expanded))
        job.runs.extend(run for wave in waves for run in wave)
        # Drain telemetry before returning. The exporter uploads in parallel and
        # flush is completion-based (waits for in-flight uploads, not a fixed
        # sleep), so the timeout is only a safety cap for a wedged network.
        if not await asyncio.to_thread(flush, timeout=120.0):
            logger.warning("telemetry flush did not fully drain within 120s; some spans may lag")
        return job


__all__ = ["Job", "Taskset"]
