"""The rollout engine: ``task.run(agent)`` / ``rollout(task, agent)``.

These drive the engine end-to-end through the real placement path: a pure-data
``Task`` row plus ``on=spawn(env_file)`` — a child process serves the env, the
engine connects over the wire, the agent answers, grading comes back. The
engine contract is a graded :class:`Run` with a trace id, and failure
isolation that never raises: a pre-launch failure yields a synthesized
``Run.failed``; a mid-run failure keeps the real run and its evidence.
"""

from __future__ import annotations

import textwrap
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

import pytest

from hud.agents.base import Agent
from hud.environment import Environment, spawn
from hud.eval import Task, Taskset
from hud.eval.rollout import rollout

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path

    from hud.environment.runtime import Runtime
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
    return Task(env=Environment("sums"), id="add", args={"a": a, "b": b})


def _solve_add(prompt: str) -> str:
    _, a, b = prompt.split(":")
    return str(int(a) + int(b))


async def test_task_run_returns_graded_run_with_trace_id(env_file: Path) -> None:
    run = await _add_task(2, 3).run(_FnAgent(_solve_add), on=spawn(env_file))

    assert run.reward == 1.0
    assert run.trace.content == "5"
    assert run.trace_id is not None
    # The factual placement record: the runtime this run executed against.
    assert run.runtime is not None
    assert run.runtime.startswith("tcp://127.0.0.1:")


async def test_mid_run_failure_keeps_the_real_run_and_its_evidence(env_file: Path) -> None:
    def boom(prompt: str) -> str:
        raise RuntimeError("agent exploded")

    run = await _add_task(2, 3).run(_FnAgent(boom), on=spawn(env_file))

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

    run = await _add_task(1, 1).run(_FnAgent(_solve_add), on=broken_provider)

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
        placed.append(f"{task.env.name}/{task.id}:{task.args['a']}")
        return spawn(env_file)(task)

    run = await _add_task(2, 3).run(_FnAgent(_solve_add), on=placer)

    assert run.reward == 1.0
    assert placed == ["sums/add:2"]


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
        Task(env=Environment("alpha"), id="add_a", args={"a": 1, "b": 2}),
        Task(env=Environment("beta"), id="add_b", args={"a": 3, "b": 4}),
    ]

    # One provider, two envs: each acquisition serves the row it was called
    # with (the task ids only exist on their own env, so a misplacement
    # would fail the rollout).
    job = await Taskset("zoo", rows).run(_FnAgent(_solve_add), on=spawn(path))

    assert [run.reward for run in job.runs] == [1.0, 1.0]
    assert [run.prompt for run in job.runs] == ["alpha:1:2", "beta:3:4"]


async def test_rollout_threads_job_and_group_ids(env_file: Path) -> None:
    run = await rollout(
        _add_task(1, 1),
        _FnAgent(_solve_add),
        on=spawn(env_file),
        job_id="j1",
        group_id="g1",
    )

    assert run.reward == 1.0
    assert run.job_id == "j1"
    assert run.group_id == "g1"
