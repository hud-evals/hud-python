"""``Task.run`` and ``Taskset.run``: rows expand into rollouts that are placed, bounded
and collected into one ``Job``."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import pytest

from hud import Environment
from hud.agents.base import Agent
from hud.agents.types import AgentConfig
from hud.eval import (
    DockerRuntime,
    Job,
    LocalRuntime,
    RuntimeConfig,
    Shared,
    Task,
    Taskset,
)
from tests.conftest import eventually
from tests.fixtures.agents import ScriptedAgent

if TYPE_CHECKING:
    from hud.eval import Run
    from tests.conftest import HudEnv
    from tests.fixtures.services import FakeServices


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


async def test_an_open_job_collects_the_runs_of_several_calls_with_its_group() -> None:
    env = Environment("sums")

    @env.template()
    async def add(a: int, b: int):
        yield f"{a}+{b}"
        yield 1.0

    session = await Job.start("training", group=2)

    first = await add(a=1, b=2).run(ScriptedAgent("3"), job=session)
    second = await add(a=3, b=4).run(ScriptedAgent("7"), job=session)

    assert first is second is session
    assert [run.job_id for run in session.runs] == [session.id] * 4


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


@pytest.mark.parametrize(
    (
        "agent_timeout",
        "row_config",
        "provider_limits",
        "verifier_limits",
        "rollout_timeout",
        "message",
    ),
    [
        pytest.param(
            None, None, None, None, 0, "rollout_timeout must be greater than 0", id="no-time-at-all"
        ),
        pytest.param(
            10,
            None,
            None,
            None,
            10,
            r"agent timeout \(10s\) must be less than rollout_timeout",
            id="an-agent-timeout-at-the-rollout-timeout",
        ),
        pytest.param(
            3600,
            {"limits": {"run_timeout_s": 3600}},
            None,
            None,
            None,
            r"agent timeout \(3600s\) must be less than runtime_config\.limits\.run_timeout_s",
            id="an-agent-timeout-at-the-run-limit",
        ),
        pytest.param(
            None,
            {"limits": {"run_timeout_s": 3600}},
            None,
            None,
            3600,
            r"actor runtime_config\.limits\.run_timeout_s \(3600s\) must be less than",
            id="a-run-limit-at-the-rollout-timeout",
        ),
        pytest.param(
            None,
            None,
            {"startup_timeout_s": 90},
            None,
            60,
            r"actor runtime_config\.limits\.startup_timeout_s \(90s\) must be less than",
            id="a-providers-limit-past-the-rollout-timeout",
        ),
        pytest.param(
            None,
            None,
            None,
            {"startup_timeout_s": 90},
            90,
            r"verifier runtime_config\.limits\.startup_timeout_s \(90s\) must be less than",
            id="a-verifiers-limit-at-the-rollout-timeout",
        ),
    ],
)
async def test_timeouts_that_contradict_each_other_raise_before_any_rollout(
    agent_timeout: float | None,
    row_config: dict[str, Any] | None,
    provider_limits: dict[str, int] | None,
    verifier_limits: dict[str, int] | None,
    rollout_timeout: float | None,
    message: str,
    services: FakeServices,
    hud_env: HudEnv,
) -> None:
    hud_env.set(PATH="/nonexistent")
    task = Task.model_validate(
        {
            "env": "lab",
            "id": "solve",
            "agent_config": {"timeout_seconds": agent_timeout},
            "runtime_config": row_config,
            "verifier": {
                "env": "judge",
                "id": "verify",
                "runtime_config": {"limits": verifier_limits},
            },
        }
    )
    provider = DockerRuntime(
        "lab", runtime_config=RuntimeConfig.model_validate({"limits": provider_limits})
    )

    with pytest.raises(ValueError, match=message):
        await task.run(ScriptedAgent("x"), runtime=provider, rollout_timeout=rollout_timeout)

    assert services.requests() == []


@pytest.mark.parametrize(
    "runtime_config",
    [None, {"image": "lab:1"}],
    ids=["a-portable-row", "a-container-row-beside-a-portable-one"],
)
async def test_rows_nothing_places_need_a_runtime(runtime_config: dict[str, Any] | None) -> None:
    rows = [
        Task(env="lab", id="solve", slug="portable"),
        Task(env="lab", id="solve", slug="other", runtime_config=runtime_config),
    ]

    with pytest.raises(ValueError, match=r"no placement: pass runtime= .*LocalRuntime"):
        await Taskset("t", rows).run(ScriptedAgent("x"))


async def test_a_row_naming_an_image_is_placed_in_docker(hud_env: HudEnv) -> None:
    hud_env.set(PATH="/nonexistent")

    job = await Task(env="lab", id="solve", runtime_config={"image": "lab:1"}).run(
        ScriptedAgent("x")
    )

    assert "'docker'" in (job.runs[0].trace.error or "")


async def test_an_empty_taskset_needs_no_runtime() -> None:
    job = await Taskset("empty", []).run(ScriptedAgent("x"))

    assert (job.runs, job.reward, job.errors) == ([], 0.0, [])


@pytest.mark.parametrize(
    ("slugs", "group", "name"),
    [
        (["only"], 1, "solve"),
        (["only"], 3, "solve (3 times)"),
        (["first", "second"], 1, "letters (2 tasks)"),
        (["first", "second"], 2, "letters (2 tasks) (2 times)"),
    ],
)
async def test_a_run_registers_one_job_named_for_what_it_runs(
    slugs: list[str], group: int, name: str, reporting: FakeServices
) -> None:
    env = Environment("letters")

    @env.template()
    async def solve():
        yield "go"
        yield 1.0

    rows = [solve().model_copy(update={"slug": slug}) for slug in slugs]
    job = await Taskset("letters", rows).run(ScriptedAgent("x"), group=group)

    (registered,) = reporting.requests("api", "POST", "/v2/trace/job/{id}/enter")
    assert registered.path == f"/v2/trace/job/{job.id}/enter"
    assert (registered.json["name"], registered.json["group"]) == (name, group)


async def test_a_pool_placement_boots_once_for_the_call_and_closes_after_it() -> None:
    events = []
    env = Environment("pooled")

    @env.initialize
    async def up() -> None:
        events.append("up")

    @env.shutdown
    async def down() -> None:
        events.append("down")

    @env.template()
    async def solve(n: int):
        yield "go"
        yield 1.0

    rows = [solve(n=1), solve(n=2)]
    job = await Taskset("pooled", rows).run(
        ScriptedAgent("x"), runtime=Shared(LocalRuntime(env), width=4), group=2
    )

    assert ([run.reward for run in job.runs], events) == ([1.0] * 4, ["up", "down"])
