"""The scheduler: ``Task.run`` and ``Taskset.run`` expand rows into rollouts, place them,
bound their concurrency and collect the runs into a ``Job``."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

import pytest

from hud.agents.base import Agent
from hud.eval import (
    Job,
    LocalRuntime,
    ModalRuntime,
    Runtime,
    RuntimeConfig,
    RuntimeLimits,
    Shared,
    Task,
    Taskset,
)
from tests.eval.envs import containers, lab, solve
from tests.harness import RecordingProvider, ScriptedAgent

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable
    from pathlib import Path

    from hud.eval import Provider, Run
    from tests.harness import FakeDocker, FakeServices


class Counting(Agent):
    """Solves ``add`` rows after a pause, recording how many run at once."""

    def __init__(self) -> None:
        super().__init__()
        self.live = 0
        self.peak = 0

    async def __call__(self, run: Run) -> None:
        self.live += 1
        self.peak = max(self.peak, self.live)
        await asyncio.sleep(0.05)
        run.trace.content = solve(run.prompt_text)
        self.live -= 1


ROWS = [
    Task(env="lab", id="add", args={"a": 1, "b": 2}),
    Task(env="lab", id="add", args={"a": 3, "b": 4}),
]


def placements(events: list[tuple[str, str]]) -> tuple[int, int]:
    """How many placements started, and the most open at once."""
    open_now = peak = 0
    for event, _slug in events:
        open_now += 1 if event == "start" else -1
        peak = max(peak, open_now)
    return sum(event == "start" for event, _ in events), peak


SCHEDULES: dict[str, tuple[Callable[[RecordingProvider], Provider], int | None, int, int, int]] = {
    "every rollout gets its own placement": (lambda inner: inner, None, 6, 6, 6),
    "max_concurrent bounds the rollouts in flight": (lambda inner: inner, 2, 6, 2, 2),
    "a pool boots once and bounds occupancy by its width": (
        lambda inner: Shared(inner, width=2),
        None,
        1,
        1,
        2,
    ),
}


@pytest.mark.parametrize(
    ("wrap", "max_concurrent", "boots", "open_at_once", "peak"),
    SCHEDULES.values(),
    ids=SCHEDULES.keys(),
)
async def test_a_taskset_runs_each_row_group_times_in_task_major_order(
    wrap: Callable[[RecordingProvider], Provider],
    max_concurrent: int | None,
    boots: int,
    open_at_once: int,
    peak: int,
    services: FakeServices,
) -> None:
    recording = RecordingProvider(LocalRuntime(lambda task: lab()))
    agent = Counting()

    job = await Taskset("pair", ROWS).run(
        agent, runtime=wrap(recording), group=3, max_concurrent=max_concurrent
    )

    slugs = [row.slug for row in ROWS]
    assert [run.slug for run in job.runs] == [slugs[0]] * 3 + [slugs[1]] * 3
    assert [run.reward for run in job.runs] == [1.0] * 6
    assert {run.job_id for run in job.runs} == {job.id}
    first, second = ({run.group_id for run in job.results[slug]} for slug in slugs)
    assert len(first) == len(second) == 1 and first != second
    assert (job.name, job.group, job.reward) == ("pair (2 tasks) (3 times)", 3, 1.0)
    assert placements(recording.events) == (boots, open_at_once)
    assert recording.events[-1][0] == "stop"
    assert agent.peak == peak
    assert services.requests() == []


async def test_an_open_job_collects_the_runs_of_several_calls() -> None:
    session = await Job.start("training", group=2)
    row = ROWS[0]
    runtime = LocalRuntime(lab())

    first = await row.run(ScriptedAgent(solve), runtime=runtime, job=session)
    second = await row.run(ScriptedAgent(solve), runtime=runtime, job=session)

    assert first is second is session
    assert [run.job_id for run in session.runs] == [session.id] * 4
    assert (session.name, session.group, session.reward) == ("training", 2, 1.0)


async def test_an_open_pool_stays_warm_across_calls_and_closes_with_its_scope() -> None:
    recording = RecordingProvider(LocalRuntime(lab()))

    async with Shared(recording, width=2) as pool:
        await Taskset("one", ROWS[:1]).run(ScriptedAgent(solve), runtime=pool)
        await Taskset("two", ROWS[1:]).run(ScriptedAgent(solve), runtime=pool)
        assert recording.events == [("start", ROWS[0].slug)]

    assert recording.events == [("start", ROWS[0].slug), ("stop", ROWS[0].slug)]


def flaky(inner: Provider) -> tuple[Provider, list[int]]:
    """A provider whose first boot fails."""
    attempts: list[int] = []

    @asynccontextmanager
    async def place(task: Task) -> AsyncIterator[Runtime]:
        attempts.append(len(attempts) + 1)
        if len(attempts) == 1:
            raise RuntimeError("boot failed")
        async with inner(task) as runtime:
            yield runtime

    return place, attempts


async def test_a_failed_boot_fails_one_lease_and_the_next_lease_retries() -> None:
    provider, attempts = flaky(LocalRuntime(lab()))

    job = await ROWS[0].run(
        ScriptedAgent(solve), runtime=Shared(provider, width=1), group=2, max_concurrent=1
    )

    assert [(run.reward, run.trace.error) for run in job.runs] == [
        (0.0, "[provisioning] RuntimeError: boot failed"),
        (1.0, None),
    ]
    assert attempts == [1, 2]
    assert (job.reward, job.errors) == (1.0, [job.runs[0]])


async def test_a_pool_reboots_a_substrate_past_its_run_timeout() -> None:
    recording = RecordingProvider(LocalRuntime(lab()))
    limited = RuntimeConfig(limits=RuntimeLimits(run_timeout_s=1))
    row = ROWS[0].model_copy(update={"runtime_config": limited})

    def strip(task: Task) -> Any:
        return recording(task.model_copy(update={"runtime_config": None}))

    async with Shared(strip, width=1) as pool:
        async with pool(row):
            pass
        await asyncio.sleep(1.05)
        async with pool(row):
            pass

    assert [event for event, _ in recording.events] == ["start", "stop", "start", "stop"]


async def test_a_pool_keys_substrates_by_env_and_config_and_tears_them_down_in_reverse() -> None:
    log: list[str] = []

    @asynccontextmanager
    async def provider(task: Task) -> AsyncIterator[Runtime]:
        name = f"{task.env}:{task.runtime_config is not None}"
        log.append(f"start {name}")
        yield Runtime(f"tcp://127.0.0.1:1/{name}")
        log.append(f"stop {name}")

    sized = RuntimeConfig(limits=RuntimeLimits(startup_timeout_s=5))
    async with Shared(provider, width=4) as pool:
        urls = []
        for row in (
            Task(env="actor", id="solve"),
            Task(env="judge", id="verify"),
            Task(env="actor", id="solve", runtime_config=sized),
            Task(env="actor", id="other"),
        ):
            async with pool(row) as runtime:
                urls.append(runtime.url)

    assert urls == [
        "tcp://127.0.0.1:1/actor:False",
        "tcp://127.0.0.1:1/judge:False",
        "tcp://127.0.0.1:1/actor:True",
        "tcp://127.0.0.1:1/actor:False",
    ]
    assert log == [
        "start actor:False",
        "start judge:False",
        "start actor:True",
        "stop actor:True",
        "stop judge:False",
        "stop actor:False",
    ]


async def test_a_pool_scope_raises_what_its_teardown_raised() -> None:
    @asynccontextmanager
    async def provider(task: Task) -> AsyncIterator[Runtime]:
        yield Runtime("tcp://127.0.0.1:1")
        raise RuntimeError("teardown boom")

    with pytest.raises(RuntimeError, match="teardown boom"):
        async with Shared(provider, width=1) as pool, pool(ROWS[0]):
            pass


async def test_a_pool_leased_outside_its_scope_says_so() -> None:
    pool = Shared(LocalRuntime(lab()), width=1)

    with pytest.raises(RuntimeError, match="lease inside the scope"):
        async with pool(ROWS[0]):
            pass


def limited(**limits: int) -> RuntimeConfig:
    return RuntimeConfig(limits=RuntimeLimits(**limits))


def with_verifier(config: RuntimeConfig) -> Task:
    return ROWS[0].model_copy(
        update={"verifier": Task(env="judge", id="verify", runtime_config=config)}
    )


def timed(seconds: float, **row: Any) -> Task:
    return ROWS[0].model_copy(update={"agent_config": {"timeout_seconds": seconds}, **row})


INVALID: dict[str, tuple[Callable[[Provider], Any], str]] = {
    "group below one": (
        lambda runtime: Taskset("t", ROWS).run(ScriptedAgent(), runtime=runtime, group=0),
        "group must be >= 1",
    ),
    "max_concurrent below one": (
        lambda runtime: Taskset("t", ROWS).run(ScriptedAgent(), runtime=runtime, max_concurrent=0),
        "max_concurrent must be >= 1",
    ),
    "a zero rollout_timeout": (
        lambda runtime: ROWS[0].run(ScriptedAgent(), runtime=runtime, rollout_timeout=0),
        "rollout_timeout must be greater than 0",
    ),
    "a negative rollout_timeout": (
        lambda runtime: ROWS[0].run(ScriptedAgent(), runtime=runtime, rollout_timeout=-1),
        "rollout_timeout must be greater than 0",
    ),
    "an agent timeout at the rollout_timeout": (
        lambda runtime: timed(10).run(ScriptedAgent(), runtime=runtime, rollout_timeout=10),
        "agent timeout (10s) must be less than rollout_timeout (10s)",
    ),
    "an agent timeout past the row's run limit": (
        lambda runtime: timed(5000, runtime_config=limited(run_timeout_s=3600)).run(
            ScriptedAgent(), runtime=runtime
        ),
        "agent timeout (5000s) must be less than runtime_config.limits.run_timeout_s (3600s)",
    ),
    "an agent timeout past the provider's run limit": (
        lambda runtime: timed(900).run(
            ScriptedAgent(),
            runtime=ModalRuntime("img", runtime_config=limited(run_timeout_s=600)),
        ),
        "agent timeout (900s) must be less than runtime_config.limits.run_timeout_s (600s)",
    ),
    "an actor run limit at the rollout_timeout": (
        lambda runtime: (
            ROWS[0]
            .model_copy(update={"runtime_config": limited(run_timeout_s=3600)})
            .run(ScriptedAgent(), runtime=runtime, rollout_timeout=3600)
        ),
        "actor runtime_config.limits.run_timeout_s (3600s) must be less than "
        "rollout_timeout (3600s)",
    ),
    "a provider startup limit at the rollout_timeout": (
        lambda runtime: ROWS[0].run(
            ScriptedAgent(),
            runtime=ModalRuntime("img", runtime_config=limited(startup_timeout_s=600)),
            rollout_timeout=600,
        ),
        "actor runtime_config.limits.startup_timeout_s (600s) must be less than "
        "rollout_timeout (600s)",
    ),
    "a verifier startup limit at the rollout_timeout": (
        lambda runtime: with_verifier(limited(startup_timeout_s=90)).run(
            ScriptedAgent(), runtime=runtime, rollout_timeout=90
        ),
        "verifier runtime_config.limits.startup_timeout_s (90s) must be less than "
        "rollout_timeout (90s)",
    ),
    "duplicate slugs": (
        lambda runtime: Taskset("t", [ROWS[0], ROWS[0]]),
        f"duplicate task slugs: {ROWS[0].slug}",
    ),
    "a pool narrower than one": (
        lambda runtime: Shared(runtime, width=0),
        "Shared width must be >= 1",
    ),
    "a portable row without a runtime": (
        lambda runtime: Taskset("t", ROWS).run(ScriptedAgent()),
        "no placement: pass runtime=",
    ),
    "a row minted by a template, serialized and reloaded": (
        lambda runtime: Task.model_validate(lab().tasks["add"](a=1, b=2).model_dump()).run(
            ScriptedAgent()
        ),
        "no placement: pass runtime=",
    ),
    "container rows mixed with portable rows": (
        lambda runtime: Taskset(
            "t", [ROWS[0], ROWS[1].model_copy(update={"runtime_config": RuntimeConfig(image="x")})]
        ).run(ScriptedAgent()),
        "no placement: pass runtime=",
    ),
}


@pytest.mark.parametrize(("start", "message"), INVALID.values(), ids=INVALID.keys())
async def test_an_invalid_schedule_raises_before_any_rollout_starts(
    start: Callable[[Provider], Any], message: str, services: FakeServices
) -> None:
    recording = RecordingProvider(LocalRuntime(lab()))

    with pytest.raises(ValueError) as raised:
        outcome = start(recording)
        if asyncio.iscoroutine(outcome):
            await outcome

    assert str(raised.value).startswith(message)
    assert recording.events == []
    assert services.requests() == []


async def test_an_empty_taskset_needs_no_placement() -> None:
    job = await Taskset("empty", []).run(ScriptedAgent())

    assert (job.runs, job.reward, job.name) == ([], 0.0, "empty (0 tasks)")


async def test_rows_minted_by_a_live_environment_run_against_it() -> None:
    events: list[str] = []
    env = lab(events)
    row = env.tasks["add"](a=2, b=3)

    job = await Taskset("live", [row]).run(ScriptedAgent(solve), group=2, max_concurrent=2)

    assert [run.reward for run in job.runs] == [1.0, 1.0]
    assert events.count("initialize") == events.count("shutdown") == 2
    assert events[-1] == "shutdown"


async def test_a_module_taskset_runs_against_the_environments_it_defines(tmp_path: Path) -> None:
    source = tmp_path / "tasks.py"
    source.write_text(
        "from hud import Environment\n\n"
        'env = Environment("lab")\n\n\n'
        "@env.template()\n"
        "async def add(a: int, b: int):\n"
        '    answer = yield f"add {a} {b}"\n'
        "    yield 1.0 if answer == str(a + b) else 0.0\n\n\n"
        "tasks = [add(a=2, b=3), add(a=4, b=5)]\n"
    )

    job = await Taskset.from_module(source).run(ScriptedAgent(solve))

    assert [run.reward for run in job.runs] == [1.0, 1.0]


CONTAINER_ROWS = {
    "an image row starts its image": (
        ROWS[0].model_copy(update={"runtime_config": RuntimeConfig(image="lab")}),
        1.0,
    ),
    "a same-env verifier rides the actor's container": (
        ROWS[0].model_copy(
            update={
                "runtime_config": RuntimeConfig(image="lab"),
                "verifier": Task(env="lab", id="frame", args={"result": {"score": 0.5}}),
            }
        ),
        0.5,
    ),
}


@pytest.mark.parametrize(("row", "reward"), CONTAINER_ROWS.values(), ids=CONTAINER_ROWS.keys())
async def test_container_rows_start_their_image_without_a_runtime(
    row: Task, reward: float, fake_docker: FakeDocker, tmp_path: Path
) -> None:
    async with containers(fake_docker, tmp_path / "rootfs", {"lab": lab()}):
        job = await Taskset("images", [row]).run(ScriptedAgent(solve))

    assert job.reward == reward
    assert [command.split()[-1] for command in fake_docker.commands("run ")] == ["lab"]
