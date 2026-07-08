"""LocalRuntime: the in-process placement over live ``Environment`` objects.

Rows join by env name like every placement; an ``Environment`` instance is a
shared substrate (daemons started once, refcounted across acquisitions, one
control channel per acquisition), a factory in a mapping is fresh per
acquisition. Everything still crosses the control channel — these tests drive
the real rollout engine against envs that only exist in this process.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator  # noqa: TC003 - env.template resolves at runtime
from typing import Any, cast

import pytest

from hud.agents.base import Agent
from hud.environment import Environment
from hud.eval import LocalRuntime, Task, Taskset
from hud.eval.run import rollout


def _sums_env(name: str = "sums") -> Environment:
    env = Environment(name)

    @env.template(id="add")
    async def add(a: int, b: int) -> AsyncGenerator[Any, Any]:
        answer = yield f"add:{a}:{b}"
        yield 1.0 if answer == str(a + b) else 0.0

    return env


class _FnAgent(Agent):
    """Stateless agent: answers each run by applying ``fn`` to ``run.prompt``."""

    def __init__(self, fn: Any) -> None:
        self._fn = fn

    async def __call__(self, run: Any) -> None:
        run.trace.content = self._fn(run.prompt)


def _solve_add(prompt: str) -> str:
    _, a, b = prompt.split(":")
    return str(int(a) + int(b))


async def test_live_env_rollout_end_to_end() -> None:
    run = await rollout(
        Task(env="sums", id="add", args={"a": 2, "b": 3}),
        _FnAgent(_solve_add),
        runtime=LocalRuntime(_sums_env()),
    )

    assert run.reward == 1.0
    assert run.trace_id


async def test_shared_instance_serves_once_across_concurrent_acquisitions() -> None:
    env = _sums_env()
    starts, stops = [], []

    @env.initialize
    async def _up() -> None:
        starts.append(1)

    @env.shutdown
    async def _down() -> None:
        stops.append(1)

    provider = LocalRuntime(env)
    task = Task(env="sums", id="add")
    release = asyncio.Event()
    all_acquired = asyncio.Event()
    urls: list[str] = []

    async def _hold() -> None:
        async with provider(task) as runtime:
            urls.append(runtime.url)
            if len(urls) == 3:
                all_acquired.set()
            await release.wait()

    holders = [asyncio.create_task(_hold()) for _ in range(3)]
    await all_acquired.wait()
    # One shared env (daemons started once), but one channel per acquisition
    # so concurrent task lifecycles never collide.
    assert len(set(urls)) == 3
    assert starts == [1]
    assert stops == []

    release.set()
    await asyncio.gather(*holders)
    assert stops == [1]


async def test_shared_env_runs_grouped_taskset() -> None:
    taskset = Taskset(
        "sums",
        [Task(env="sums", id="add", args={"a": a, "b": a + 1}, slug=f"add-{a}") for a in range(3)],
    )

    job = await taskset.run(
        _FnAgent(_solve_add),
        runtime=LocalRuntime(_sums_env()),
        group=2,
        max_concurrent=3,
    )

    assert len(job.runs) == 6
    assert all(run.reward == 1.0 for run in job.runs)


async def test_factory_builds_fresh_env_per_acquisition() -> None:
    built: list[Environment] = []

    def factory() -> Environment:
        env = _sums_env()
        built.append(env)
        return env

    task = Task(env="sums", id="add", args={"a": 1, "b": 2})
    job = await task.run(
        _FnAgent(_solve_add),
        runtime=LocalRuntime({"sums": factory}),
        group=3,
    )

    assert all(run.reward == 1.0 for run in job.runs)
    assert len(built) == 3


async def test_mapping_joins_rows_by_env_name() -> None:
    doubles = Environment("doubles")

    @doubles.template(id="double")
    async def double(n: int) -> AsyncGenerator[Any, Any]:
        answer = yield f"double:{n}"
        yield 1.0 if answer == str(2 * n) else 0.0

    def _solve(prompt: str) -> str:
        kind, *parts = prompt.split(":")
        if kind == "double":
            return str(2 * int(parts[0]))
        return str(int(parts[0]) + int(parts[1]))

    taskset = Taskset(
        "mixed",
        [
            Task(env="sums", id="add", args={"a": 4, "b": 5}),
            Task(env="doubles", id="double", args={"n": 7}),
        ],
    )

    job = await taskset.run(
        _FnAgent(_solve),
        runtime=LocalRuntime({"sums": _sums_env(), "doubles": doubles}),
    )

    assert [run.reward for run in job.runs] == [1.0, 1.0]


async def test_unknown_env_name_fails_loudly() -> None:
    provider = LocalRuntime(_sums_env())

    with pytest.raises(KeyError, match="no environment named 'other'"):
        async with provider(Task(env="other", id="add")):
            pass


def test_rejects_path_argument_pointing_at_subprocess_runtime() -> None:
    with pytest.raises(TypeError, match="SubprocessRuntime"):
        LocalRuntime(cast("Any", "env.py"))


def test_rejects_bare_factory_without_a_name() -> None:
    with pytest.raises(TypeError, match="mapping"):
        LocalRuntime(cast("Any", _sums_env))
