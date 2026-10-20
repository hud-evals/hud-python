"""``Shared``: one substrate per environment and runtime configuration, leased to at most
``width`` rollouts at once for the life of its scope."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from hud import Environment
from hud.agents.base import Agent
from hud.eval import LocalRuntime, Runtime, Shared, Task, Taskset, rollout
from tests.conftest import eventually
from tests.fixtures.agents import ScriptedAgent

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from hud.eval import Run


async def test_a_pool_boots_once_and_holds_at_most_width_rollouts_at_a_time() -> None:
    events = []
    env = Environment("sim")

    @env.initialize
    async def up() -> None:
        events.append("up")

    @env.template()
    async def step(n: int):
        yield "go"
        yield 1.0

    live = [0]
    seen = []

    class Overlaps(Agent):
        async def __call__(self, run: Run) -> None:
            live[0] += 1
            await eventually(lambda: live[0] >= 2)
            await asyncio.sleep(0.05)
            seen.append(live[0])
            live[0] -= 1

    pool = Shared(LocalRuntime(env), width=2)
    job = await Taskset("sim", [step(n=1), step(n=2)]).run(Overlaps(), runtime=pool, group=2)

    assert [run.reward for run in job.runs] == [1.0] * 4
    assert (events, max(seen)) == (["up"], 2)


async def test_an_open_pool_stays_warm_across_calls_and_stops_when_its_scope_closes() -> None:
    events = []
    env = Environment("sim")

    @env.initialize
    async def up() -> None:
        events.append("up")

    @env.shutdown
    async def down() -> None:
        events.append("down")

    @env.template()
    async def step():
        yield "go"
        yield 1.0

    async with Shared(LocalRuntime(env), width=2) as pool:
        await step().run(ScriptedAgent("x"), runtime=pool)
        await step().run(ScriptedAgent("x"), runtime=pool)
        assert events == ["up"]

    assert events == ["up", "down"]


async def test_each_environment_and_runtime_configuration_gets_its_own_substrate() -> None:
    placed = []

    def build(task: Task) -> Environment:
        env = Environment(task.env)

        @env.template()
        async def step():
            yield "go"
            yield 1.0

        return env

    @asynccontextmanager
    async def sized(task: Task) -> AsyncIterator[Runtime]:
        assert task.runtime_config is not None and task.runtime_config.resources is not None
        placed.append((task.env, task.runtime_config.resources.cpu))
        unsized = task.model_copy(update={"runtime_config": None})
        async with LocalRuntime(build)(unsized) as runtime:
            yield runtime

    rows = [
        Task(env="sim", id="step", slug="small", runtime_config={"resources": {"cpu": 1}}),
        Task(env="sim", id="step", slug="small-again", runtime_config={"resources": {"cpu": 1}}),
        Task(env="sim", id="step", slug="large", runtime_config={"resources": {"cpu": 4}}),
        Task(env="lab", id="step", slug="elsewhere", runtime_config={"resources": {"cpu": 1}}),
    ]
    job = await Taskset("sim", rows).run(ScriptedAgent("x"), runtime=Shared(sized, width=4))

    assert job.reward == 1.0
    assert sorted(placed) == [("lab", 1), ("sim", 1), ("sim", 4)]


async def test_a_failed_boot_fails_its_own_lease_and_the_next_lease_boots_again() -> None:
    attempts = []
    env = Environment("sim")

    @env.template()
    async def step():
        yield "go"
        yield 1.0

    @asynccontextmanager
    async def flaky(task: Task) -> AsyncIterator[Runtime]:
        attempts.append(task.slug)
        if len(attempts) == 1:
            raise RuntimeError("boot failed")
        async with LocalRuntime(env)(task) as runtime:
            yield runtime

    job = await step().run(
        ScriptedAgent("x"), runtime=Shared(flaky, width=1), group=2, max_concurrent=1
    )

    assert [(run.reward, run.trace.error) for run in job.runs] == [
        (0.0, "[provisioning] RuntimeError: boot failed"),
        (1.0, None),
    ]
    assert len(attempts) == 2


async def test_a_substrate_past_its_run_limit_is_replaced_on_the_next_lease() -> None:
    events = []
    env = Environment("sim")

    @env.initialize
    async def up() -> None:
        events.append("up")

    @env.shutdown
    async def down() -> None:
        events.append("down")

    @env.template()
    async def step():
        yield "go"
        yield 1.0

    @asynccontextmanager
    async def limited(task: Task) -> AsyncIterator[Runtime]:
        async with LocalRuntime(env)(task.model_copy(update={"runtime_config": None})) as runtime:
            yield runtime

    row = Task(env="sim", id="step", runtime_config={"limits": {"run_timeout_s": 1}})
    async with Shared(limited, width=1) as pool:
        await row.run(ScriptedAgent("x"), runtime=pool)
        await asyncio.sleep(1.1)
        await row.run(ScriptedAgent("x"), runtime=pool)

    assert events == ["up", "down", "up", "down"]


async def test_a_lease_outside_the_pools_scope_fails_the_run() -> None:
    env = Environment("sim")

    @env.template()
    async def step():
        yield "go"
        yield 1.0

    run = await rollout(step(), ScriptedAgent("x"), runtime=Shared(LocalRuntime(env), width=1))

    assert (run.trace.error or "").startswith(
        "[provisioning] RuntimeError: Shared substrates outlive single rollouts; lease inside "
        "the scope"
    )
