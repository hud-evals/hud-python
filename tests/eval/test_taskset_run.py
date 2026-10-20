"""``Task.run`` and ``Taskset.run``: rows expand into rollouts that are placed, bounded
and collected into one ``Job``."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import pytest

from hud import Environment
from hud.agents.base import Agent
from hud.agents.types import AgentConfig
from hud.eval import (
    LocalRuntime,
    Task,
    Taskset,
)
from tests.conftest import eventually

if TYPE_CHECKING:
    from hud.eval import Run


async def test_each_row_runs_group_times_in_task_major_order_under_one_job() -> None:
    env = Environment("sums")

    @env.template()
    async def add(a: int, b: int):
        answer = yield f"{a}+{b}"
        yield 1.0 if answer == str(a + b) else 0.0

    class Adds(Agent):
        async def __call__(self, run: Run) -> None:
            a, b = run.prompt_text.split("+")
            run.trace.content = str(int(a) + int(b))

    first, second = add(a=1, b=2), add(a=3, b=4)
    job = await Taskset("sums", [first, second]).run(Adds(), group=3)

    assert [run.slug for run in job.runs] == [first.slug] * 3 + [second.slug] * 3
    assert [run.reward for run in job.runs] == [1.0] * 6
    assert {run.job_id for run in job.runs} == {job.id}
    assert [len(job.results[row.slug]) for row in (first, second)] == [3, 3]
    groups = [{run.group_id for run in job.results[row.slug]} for row in (first, second)]
    assert [len(group) for group in groups] == [1, 1] and groups[0] != groups[1]


@pytest.mark.parametrize(("max_concurrent", "peak"), [(None, 6), (2, 2)])
async def test_every_rollout_gets_its_own_env_and_max_concurrent_bounds_them(
    max_concurrent: int | None, peak: int
) -> None:
    started = []

    def build(task: Task) -> Environment:
        env = Environment("sums")

        @env.initialize
        async def up() -> None:
            started.append(task.slug)

        @env.template()
        async def add(a: int, b: int):
            yield f"{a}+{b}"
            yield 1.0

        return env

    live = [0]
    seen = []

    class Overlaps(Agent):
        async def __call__(self, run: Run) -> None:
            live[0] += 1
            await eventually(lambda: live[0] >= peak)
            await asyncio.sleep(0.05)
            seen.append(live[0])
            live[0] -= 1

    rows = [Task(env="sums", id="add", args={"a": a, "b": 1}) for a in (1, 2)]
    await Taskset("sums", rows).run(
        Overlaps(), runtime=LocalRuntime(build), group=3, max_concurrent=max_concurrent
    )

    assert (len(started), max(seen)) == (6, peak)


async def test_rollouts_on_one_live_environment_take_turns() -> None:
    env = Environment("shared")
    live = []

    @env.template()
    async def wait():
        live.append(1)
        assert sum(live) == 1
        yield "go"
        live.append(-1)
        yield 1.0

    class Pauses(Agent):
        async def __call__(self, run: Run) -> None:
            await asyncio.sleep(0.05)

    job = await wait().run(Pauses(), group=3)

    assert [run.reward for run in job.runs] == [1.0] * 3


async def test_a_jobs_reward_averages_its_graded_runs_and_its_errors_are_the_rest() -> None:
    env = Environment("mixed")

    @env.template()
    async def check():
        answer = yield "say yes"
        yield 1.0 if answer == "yes" else 0.0

    @env.template()
    async def grade_raises():
        yield "go"
        raise RuntimeError("grader exploded")
        yield 0.0

    class AnswersThenWaits(Agent):
        async def __call__(self, run: Run) -> None:
            run.trace.content = "yes"
            await asyncio.sleep(30)

    rows = [
        Task(env="mixed", id="check", slug="timed-out-after-answering"),
        Task(env="mixed", id="grade_raises", slug="grader-raised"),
    ]
    agent = AnswersThenWaits(AgentConfig(timeout_seconds=0.3))
    job = await Taskset("mixed", rows).run(agent, runtime=LocalRuntime(env))

    assert {run.slug: run.reward for run in job.runs} == {
        "timed-out-after-answering": 1.0,
        "grader-raised": 0.0,
    }
    assert ([run.slug for run in job.errors], job.reward) == (["grader-raised"], 1.0)


async def test_cancelling_a_run_cancels_every_rollout_and_stops_their_envs() -> None:
    events = []

    def build(task: Task) -> Environment:
        env = Environment("sums")

        @env.shutdown
        async def down() -> None:
            events.append("stopped")

        @env.template()
        async def add(a: int, b: int):
            answer = yield f"{a}+{b}"
            events.append(f"graded {answer}")
            yield 1.0

        return env

    prompted = []

    class Waits(Agent):
        async def __call__(self, run: Run) -> None:
            prompted.append(run.prompt_text)
            await asyncio.sleep(30)

    rows = [Task(env="sums", id="add", args={"a": a, "b": 1}) for a in (1, 2)]
    pending = asyncio.create_task(
        Taskset("sums", rows).run(Waits(), runtime=LocalRuntime(build), group=2)
    )
    await eventually(lambda: len(prompted) == 4)
    pending.cancel()

    with pytest.raises(asyncio.CancelledError):
        await pending
    await eventually(lambda: events == ["stopped"] * 4)
