"""``Shared``: one substrate leased to many concurrent rollouts.

The contract: the substrate boots lazily on the first lease and lives for the
enclosing ``async with`` scope (one boot, deterministic teardown); ``width``
bounds concurrent occupancy by making the next lease wait, not error; and
``Taskset.run`` scopes a context-manager placement to the call, so a bare
``runtime=Shared(...)`` needs no ceremony.
"""

from __future__ import annotations

import asyncio
import contextlib
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

import pytest
from typing_extensions import override

from hud.agents.base import Agent
from hud.environment import Environment
from hud.eval import Shared, Task, Taskset
from hud.eval.runtime import Runtime, _local

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from hud.eval.task import Task as TaskRow


class _CountingProvider:
    """Wraps ``_local`` around one env, counting provisions and teardowns."""

    def __init__(self, env: Environment) -> None:
        self.env = env
        self.provisions = 0
        self.teardowns = 0

    @asynccontextmanager
    async def __call__(self, task: TaskRow) -> AsyncIterator[Runtime]:
        self.provisions += 1
        try:
            async with _local(self.env) as runtime:
                yield runtime
        finally:
            self.teardowns += 1


def _echo_env() -> Environment:
    env = Environment("pool")

    @env.template()
    async def echo(tag: str = "x"):
        answer = yield f"go {tag}"
        yield 1.0 if answer == "ok" else 0.0

    return env


class _OkAgent(Agent):
    @override
    async def __call__(self, run: Any) -> None:
        run.trace.content = "ok"


async def test_one_boot_serves_a_whole_grouped_run_and_closes_after() -> None:
    inner = _CountingProvider(_echo_env())
    shared = Shared(inner, width=2)

    job = await Taskset("pool", [Task(env="pool", id="echo")]).run(
        _OkAgent(), runtime=shared, group=3
    )

    assert [run.reward for run in job.runs] == [1.0, 1.0, 1.0]
    # All repeats shared one substrate, and one group_id (ordinary semantics).
    assert inner.provisions == 1
    assert inner.teardowns == 1  # scoped to the run() call
    assert len({run.group_id for run in job.runs}) == 1


async def test_an_open_scope_keeps_the_substrate_warm_across_calls() -> None:
    inner = _CountingProvider(_echo_env())
    taskset = Taskset("pool", [Task(env="pool", id="echo")])

    async with Shared(inner, width=2) as shared:
        await taskset.run(_OkAgent(), runtime=shared)
        await taskset.run(_OkAgent(), runtime=shared)
        assert inner.teardowns == 0  # still warm between calls
    assert inner.provisions == 1
    assert inner.teardowns == 1


async def test_width_bounds_occupancy_by_waiting_not_erroring() -> None:
    live = 0
    peak = 0

    class _SlowAgent(Agent):
        @override
        async def __call__(self, run: Any) -> None:
            nonlocal live, peak
            live += 1
            peak = max(peak, live)
            await asyncio.sleep(0.05)
            live -= 1
            run.trace.content = "ok"

    job = await Taskset("pool", [Task(env="pool", id="echo")]).run(
        _SlowAgent(), runtime=Shared(_CountingProvider(_echo_env()), width=2), group=5
    )

    assert [run.reward for run in job.runs] == [1.0] * 5
    assert peak <= 2  # lease 3 waited for a slot instead of raising


async def test_a_failed_boot_fails_one_lease_and_the_next_retries() -> None:
    attempts = 0
    env = _echo_env()

    @asynccontextmanager
    async def flaky(task: TaskRow) -> AsyncIterator[Runtime]:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("boot boom")
        async with _local(env) as runtime:
            yield runtime

    job = await Taskset("pool", [Task(env="pool", id="echo")]).run(
        _OkAgent(), runtime=Shared(flaky, width=1), group=2
    )

    # One rollout absorbed the boot failure; the retry served the other.
    assert attempts == 2
    assert sorted(run.reward for run in job.runs) == [0.0, 1.0]
    assert any(run.trace.is_error for run in job.runs)


async def test_leasing_outside_any_scope_errors_loudly() -> None:
    shared = Shared(_CountingProvider(_echo_env()), width=1)
    with pytest.raises(RuntimeError, match="scope"):
        async with shared(Task(env="pool", id="echo")):
            pass


async def test_scope_exit_without_any_lease_is_clean() -> None:
    inner = _CountingProvider(_echo_env())
    async with Shared(inner, width=1):
        pass
    assert (inner.provisions, inner.teardowns) == (0, 0)


async def test_teardown_error_surfaces_at_scope_exit() -> None:
    @asynccontextmanager
    async def leaky(task: TaskRow) -> AsyncIterator[Runtime]:
        try:
            yield Runtime("tcp://127.0.0.1:1")
        finally:
            raise RuntimeError("teardown boom")

    with pytest.raises(RuntimeError, match="teardown boom"):
        async with Shared(leaky, width=1) as shared:
            async with shared(Task(env="pool", id="echo")):
                pass


async def test_width_must_be_positive() -> None:
    with pytest.raises(ValueError, match="width"):
        Shared(_CountingProvider(_echo_env()), width=0)


async def test_non_pooling_placements_are_untouched_by_the_scope_hook() -> None:
    # A plain provider is not a context manager; run() must not try to enter it.
    inner = _CountingProvider(_echo_env())
    assert not isinstance(inner, contextlib.AbstractAsyncContextManager)

    job = await Taskset("pool", [Task(env="pool", id="echo")]).run(
        _OkAgent(), runtime=inner, group=2
    )

    assert [run.reward for run in job.runs] == [1.0, 1.0]
    assert inner.provisions == 2  # one substrate per rollout, as before
