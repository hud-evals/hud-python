"""``InstancePool``: a few warm substrates serve many rollouts, rotated by lease count or age."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

import pytest

from hud import Environment
from hud.agents.base import Agent
from hud.eval import InstancePool, LocalRuntime, Runtime, Task, Taskset, rollout
from hud.eval.runtime.core import resolve_runtime_config
from tests.conftest import eventually
from tests.fixtures.agents import ScriptedAgent

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from hud.eval import Run


def make_env(name: str = "sim") -> Environment:
    env = Environment(name)

    @env.template()
    async def step(n: int = 0):
        yield "go"
        yield 1.0

    return env


class Booting:
    """A provider that counts boots and shutdowns and serves a fresh env per boot."""

    def __init__(self) -> None:
        self.boots = 0
        self.closed = 0
        self.serving = 0

    @asynccontextmanager
    async def __call__(self, task: Task) -> AsyncIterator[Runtime]:
        self.boots += 1
        self.serving += 1
        try:
            async with LocalRuntime(make_env(task.env))(task) as runtime:
                yield runtime
        finally:
            self.serving -= 1
            self.closed += 1


def rows(n: int) -> list[Task]:
    return [Task(env="sim", id="step", slug=f"row-{i}", args={"n": i}) for i in range(n)]


async def test_rows_flow_through_a_few_instances_each_holding_at_most_width() -> None:
    provider = Booting()
    live = [0]
    peak = [0]

    class Overlaps(Agent):
        async def __call__(self, run: Run) -> None:
            live[0] += 1
            peak[0] = max(peak[0], live[0])
            await asyncio.sleep(0.02)
            live[0] -= 1

    pool = InstancePool(provider, instances=2, width=3)
    job = await Taskset("sim", rows(12)).run(Overlaps(), runtime=pool)

    assert [run.reward for run in job.runs] == [1.0] * 12
    assert provider.boots == 2
    assert peak[0] == 6  # instances x width rollouts in flight, never more
    assert provider.closed == 2  # the scope closed both instances


async def test_an_instance_retires_after_rows_per_instance_and_a_fresh_one_boots() -> None:
    provider = Booting()

    pool = InstancePool(provider, instances=1, width=1, rows_per_instance=4)
    job = await Taskset("sim", rows(10)).run(ScriptedAgent("x"), runtime=pool, max_concurrent=1)

    assert job.reward == 1.0
    assert provider.boots == 3  # 4 + 4 + 2 rows
    assert provider.closed == 3


async def test_a_retired_instance_finishes_its_leases_before_it_stops() -> None:
    provider = Booting()
    release = asyncio.Event()

    class Waits(Agent):
        async def __call__(self, run: Run) -> None:
            await release.wait()

    async with InstancePool(provider, instances=1, width=2, rows_per_instance=2) as pool:
        first = asyncio.ensure_future(rollout(rows(1)[0], Waits(), runtime=pool))
        second = asyncio.ensure_future(rollout(rows(2)[1], Waits(), runtime=pool))
        await eventually(lambda: provider.boots == 1 and pool.live == 1)
        # The instance has handed out its two leases; a third waits for a replacement.
        third = asyncio.ensure_future(rollout(rows(3)[2], ScriptedAgent("x"), runtime=pool))
        await eventually(lambda: provider.boots == 2)
        assert provider.closed == 0 and provider.serving == 2
        release.set()
        runs = await asyncio.gather(first, second, third)

    assert [run.reward for run in runs] == [1.0, 1.0, 1.0]
    assert provider.closed == 2


async def test_ttl_rotates_an_aged_instance() -> None:
    provider = Booting()

    async with InstancePool(provider, instances=1, width=1, ttl_s=0.05) as pool:
        await rollout(rows(1)[0], ScriptedAgent("x"), runtime=pool)
        await asyncio.sleep(0.06)
        await rollout(rows(2)[1], ScriptedAgent("x"), runtime=pool)

    assert provider.boots == 2


async def test_a_failed_boot_fails_only_its_lease_and_the_next_lease_boots_again() -> None:
    attempts: list[str] = []
    env = make_env()

    @asynccontextmanager
    async def flaky(task: Task) -> AsyncIterator[Runtime]:
        attempts.append(task.slug)
        if len(attempts) == 1:
            raise RuntimeError("boot failed")
        async with LocalRuntime(env)(task) as runtime:
            yield runtime

    job = await Taskset("sim", rows(2)).run(
        ScriptedAgent("x"), runtime=InstancePool(flaky, instances=2, width=1), max_concurrent=1
    )

    assert [(run.reward, run.trace.error) for run in job.runs] == [
        (0.0, "[provisioning] RuntimeError: boot failed"),
        (1.0, None),
    ]
    assert attempts == ["row-0", "row-1"]


async def test_each_environment_and_runtime_configuration_gets_its_own_instances() -> None:
    placed = []

    @asynccontextmanager
    async def sized(task: Task) -> AsyncIterator[Runtime]:
        assert task.runtime_config is not None and task.runtime_config.resources is not None
        placed.append((task.env, task.runtime_config.resources.cpu))
        unsized = task.model_copy(update={"runtime_config": None})
        async with LocalRuntime(make_env(task.env))(unsized) as runtime:
            yield runtime

    mixed = [
        Task(env="sim", id="step", slug="small", runtime_config={"resources": {"cpu": 1}}),
        Task(env="sim", id="step", slug="small-again", runtime_config={"resources": {"cpu": 1}}),
        Task(env="sim", id="step", slug="large", runtime_config={"resources": {"cpu": 4}}),
        Task(env="lab", id="step", slug="elsewhere", runtime_config={"resources": {"cpu": 1}}),
    ]
    pool = InstancePool(sized, instances=1, width=4)
    job = await Taskset("sim", mixed).run(ScriptedAgent("x"), runtime=pool)

    assert job.reward == 1.0
    assert sorted(placed) == [("lab", 1), ("sim", 1), ("sim", 4)]
    assert resolve_runtime_config(pool, mixed[2]) == mixed[2].runtime_config


async def test_leasing_outside_a_scope_is_refused_and_arguments_are_validated() -> None:
    pool = InstancePool(Booting())
    with pytest.raises(RuntimeError, match="outlive single rollouts"):
        async with pool(rows(1)[0]):
            pass
    with pytest.raises(ValueError, match="rows_per_instance"):
        InstancePool(Booting(), rows_per_instance=0)
    with pytest.raises(ValueError, match="ttl_s"):
        InstancePool(Booting(), ttl_s=0)
