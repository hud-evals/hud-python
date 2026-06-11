"""The rollout engine: ``rollout(task, agent)`` and its schedulers.

These drive the engine end-to-end through the real placement path: a pure-data
``Task`` row plus ``runtime=LocalRuntime(env_file)`` — a child process serves the env, the
engine connects over the wire, the agent answers, grading comes back. The
engine contract is a graded :class:`Run` with a trace id (always under a job —
there are no standalone traces), and failure isolation that never raises: a
pre-launch failure yields a synthesized ``Run.failed``; a mid-run failure
keeps the real run and its evidence. ``Task.run`` / ``Taskset.run`` schedule
the atom and return a :class:`Job`.
"""

from __future__ import annotations

import textwrap
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

import pytest

from hud.agents.base import Agent
from hud.eval import Job, LocalRuntime, Task, Taskset
from hud.eval.rollout import rollout

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path

    from hud.eval.runtime import Runtime
    from hud.eval.task import Task as TaskRow

_SUMS_ENV = """\
from hud import Environment

env = Environment("sums")


@env.task()
async def add(a: int, b: int):
    answer = yield f"add:{a}:{b}"
    yield 1.0 if answer == str(a + b) else 0.0
"""


@pytest.fixture(scope="module")
def env_file(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("sums") / "env.py"
    path.write_text(textwrap.dedent(_SUMS_ENV), encoding="utf-8")
    return path


class _FnAgent(Agent):
    """Stateless agent: answers each run by applying ``fn`` to ``run.prompt``."""

    def __init__(self, fn: Any) -> None:
        self._fn = fn

    async def __call__(self, run: Any) -> None:
        run.trace.content = self._fn(run.prompt)


def _add_task(a: int, b: int) -> Task:
    """A pure data row; the env it names is defined by the spawned file."""
    return Task(env="sums", id="add", args={"a": a, "b": b})


def _solve_add(prompt: str) -> str:
    _, a, b = prompt.split(":")
    return str(int(a) + int(b))


async def test_rollout_returns_graded_run_with_trace_id(env_file: Path) -> None:
    run = await rollout(_add_task(2, 3), _FnAgent(_solve_add), runtime=LocalRuntime(env_file))

    assert run.reward == 1.0
    assert run.trace.content == "5"
    assert run.trace_id is not None
    # No standalone traces: a bare rollout registers a single-run job itself.
    assert run.job_id is not None
    # The factual placement record: the runtime this run executed against.
    assert run.runtime is not None
    assert run.runtime.startswith("tcp://127.0.0.1:")


async def test_mid_run_failure_keeps_the_real_run_and_its_evidence(env_file: Path) -> None:
    def boom(prompt: str) -> str:
        raise RuntimeError("agent exploded")

    run = await rollout(_add_task(2, 3), _FnAgent(boom), runtime=LocalRuntime(env_file))

    assert run.trace.isError
    assert "agent exploded" in (run.trace.content or "")
    assert run.trace_id is not None  # failed runs still key a trajectory
    # The session was live, so the receipt keeps the evidence: the prompt the
    # agent saw and the runtime the rollout executed against.
    assert run.prompt == "add:2:3"
    assert run.runtime is not None
    assert run.reward == 0.0  # never graded


async def test_pre_launch_failure_yields_a_synthesized_failed_run() -> None:
    @asynccontextmanager
    async def broken_provider(task: TaskRow) -> AsyncIterator[Runtime]:
        raise RuntimeError("no substrate for you")
        yield  # pragma: no cover

    run = await rollout(_add_task(1, 1), _FnAgent(_solve_add), runtime=broken_provider)

    assert run.trace.isError
    assert "no substrate for you" in (run.trace.content or "")
    assert run.trace_id is not None
    assert run.prompt is None  # nothing ever started
    assert run.runtime is None


async def test_provider_is_called_with_the_task_row_being_placed(env_file: Path) -> None:
    placed: list[str] = []

    def placer(task: TaskRow) -> Any:
        # The scheduler half of placement: the row is the request, so a
        # provider can size/route each substrate per task.
        placed.append(f"{task.env}/{task.id}:{task.args['a']}")
        return LocalRuntime(env_file)(task)

    run = await rollout(_add_task(2, 3), _FnAgent(_solve_add), runtime=placer)

    assert run.reward == 1.0
    assert placed == ["sums/add:2"]


async def test_task_run_schedules_a_single_task_job(env_file: Path) -> None:
    job = await _add_task(2, 3).run(_FnAgent(_solve_add), runtime=LocalRuntime(env_file))

    (run,) = job.runs
    assert job.reward == 1.0
    assert run.trace.content == "5"
    assert run.job_id == job.id  # the run's trace reports under the job


async def test_task_run_has_taskset_scheduling_semantics(env_file: Path) -> None:
    job = await _add_task(1, 2).run(
        _FnAgent(_solve_add), runtime=LocalRuntime(env_file), group=2, max_concurrent=1
    )

    assert job.group == 2
    assert [run.reward for run in job.runs] == [1.0, 1.0]
    # The group repeats one task, so they share a GRPO group id.
    assert len({run.group_id for run in job.runs}) == 1


async def test_open_job_spans_multiple_scheduler_calls(env_file: Path) -> None:
    session = await Job.start("session", group=2)
    provider = LocalRuntime(env_file)

    job1 = await _add_task(1, 1).run(_FnAgent(_solve_add), runtime=provider, job=session)
    job2 = await _add_task(2, 2).run(_FnAgent(_solve_add), runtime=provider, job=session)

    # Both calls accumulate into the one open job (group defaults to the job's).
    assert job1 is session
    assert job2 is session
    assert len(session.runs) == 4
    assert {run.job_id for run in session.runs} == {session.id}
    assert session.reward == 1.0


_TWO_ENVS = """\
from hud import Environment

alpha = Environment("alpha")
beta = Environment("beta")


@alpha.task()
async def add_a(a: int, b: int):
    answer = yield f"alpha:{a}:{b}"
    yield 1.0 if answer == str(a + b) else 0.0


@beta.task()
async def add_b(a: int, b: int):
    answer = yield f"beta:{a}:{b}"
    yield 1.0 if answer == str(a + b) else 0.0
"""


async def test_one_spawn_serves_each_rows_env_in_a_mixed_taskset(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    path = tmp_path_factory.mktemp("zoo") / "envs.py"
    path.write_text(_TWO_ENVS, encoding="utf-8")
    rows = [
        Task(env="alpha", id="add_a", args={"a": 1, "b": 2}),
        Task(env="beta", id="add_b", args={"a": 3, "b": 4}),
    ]

    # One provider, two envs: each acquisition serves the row it was called
    # with (the task ids only exist on their own env, so a misplacement
    # would fail the rollout).
    job = await Taskset("zoo", rows).run(_FnAgent(_solve_add), runtime=LocalRuntime(path))

    assert [run.reward for run in job.runs] == [1.0, 1.0]
    assert [run.prompt for run in job.runs] == ["alpha:1:2", "beta:3:4"]


async def test_rollout_threads_job_and_group_ids(env_file: Path) -> None:
    run = await rollout(
        _add_task(1, 1),
        _FnAgent(_solve_add),
        runtime=LocalRuntime(env_file),
        job_id="j1",
        group_id="g1",
    )

    assert run.reward == 1.0
    assert run.job_id == "j1"
    assert run.group_id == "g1"
