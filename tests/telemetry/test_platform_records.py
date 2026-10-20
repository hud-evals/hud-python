"""What the platform records about an execution: one job for it, a trace entered for each
rollout under that job, and each rollout's outcome when it exits, sent only while
telemetry is on with an API key."""

from __future__ import annotations

import asyncio
import logging
import uuid
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

import pytest

from hud import Environment
from hud.agents.base import Agent
from hud.agents.types import AgentConfig
from hud.eval import Job, LocalRuntime, Runtime, Task, Taskset, rollout
from hud.types import Step
from tests.conftest import eventually
from tests.fixtures.agents import ScriptedAgent

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from hud.eval import Run
    from tests.conftest import HudEnv
    from tests.fixtures.services import FakeServices

JOB_ENTER = "/v2/trace/job/{id}/enter"
TRACE_ENTER = "/v2/trace/{id}/enter"
TRACE_EXIT = "/v2/trace/{id}/exit"


def checked() -> Environment:
    env = Environment("checked")

    @env.template()
    async def check(n: int = 0):
        answer = yield "say yes"
        yield 1.0 if answer == "yes" else 0.0

    return env


async def test_a_taskset_reports_one_job_and_enters_each_rollout_under_it(
    reporting: FakeServices, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger="hud.eval.job")
    rows = [Task(env="checked", id="check", args={"n": n}) for n in (1, 2)]

    class Sampling(Agent):
        async def __call__(self, run: Run) -> None:
            run.trace.content = "yes"

    agent = Sampling(AgentConfig(model="claude-test"))
    job = await Taskset("pair", rows, taskset_id="ts-1").run(
        agent, runtime=LocalRuntime(checked()), group=2
    )

    (registered,) = reporting.requests("api", "POST", JOB_ENTER)
    entered = {r.path.split("/")[3]: r.json for r in reporting.requests("api", "POST", TRACE_ENTER)}
    assert registered.path == f"/v2/trace/job/{job.id}/enter"
    assert (registered.json["group"], registered.json["taskset_id"]) == (2, "ts-1")
    assert f"job: {reporting.url('web')}/jobs/{uuid.UUID(job.id)}" in caplog.messages
    assert set(entered) == {run.trace_id for run in job.runs}
    assert {(e["job_id"], e["model"]) for e in entered.values()} == {(job.id, "claude-test")}
    assert sorted((e["task_slug"], e["group_id"]) for e in entered.values()) == sorted(
        (run.slug, run.group_id) for run in job.runs
    )


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

    (registered,) = reporting.requests("api", "POST", JOB_ENTER)
    assert registered.path == f"/v2/trace/job/{job.id}/enter"
    assert (registered.json["name"], registered.json["group"]) == (name, group)


async def test_a_lone_rollout_reports_a_job_of_one_named_for_its_task(
    reporting: FakeServices,
) -> None:
    run = await rollout(
        Task(env="checked", id="check"), ScriptedAgent("yes"), runtime=LocalRuntime(checked())
    )

    (registered,) = reporting.requests("api", "POST", JOB_ENTER)
    assert (registered.json["name"], registered.json["group"]) == ("check", 1)
    assert registered.path == f"/v2/trace/job/{run.job_id}/enter"


async def test_a_rollout_given_a_job_reports_none_of_its_own(reporting: FakeServices) -> None:
    job_id = "a" * 32

    await rollout(
        Task(env="checked", id="check"),
        ScriptedAgent("yes"),
        runtime=LocalRuntime(checked()),
        job_id=job_id,
    )

    (entered,) = reporting.bodies("api", "POST", TRACE_ENTER)
    assert (reporting.requests("api", "POST", JOB_ENTER), entered["job_id"]) == ([], job_id)


async def test_an_open_job_is_reported_once_for_every_call_it_spans(
    reporting: FakeServices,
) -> None:
    env = checked()
    job = await Job.start("training", group=2)

    await Task(env="checked", id="check").run(
        ScriptedAgent("yes"), runtime=LocalRuntime(env), job=job
    )
    await Task(env="checked", id="check").run(
        ScriptedAgent("yes"), runtime=LocalRuntime(env), job=job
    )

    assert [(b["name"], b["group"]) for b in reporting.bodies("api", "POST", JOB_ENTER)] == [
        ("training", 2)
    ]
    assert {e["job_id"] for e in reporting.bodies("api", "POST", TRACE_ENTER)} == {job.id}


async def test_a_graded_rollout_reports_its_reward_evaluation_and_metadata(
    reporting: FakeServices,
) -> None:
    env = Environment("graded")

    @env.template()
    async def check():
        yield "go"
        yield {
            "score": 0.5,
            "subscores": [{"name": "format", "weight": 1.0, "value": 0.5, "info": {"seen": True}}],
        }

    class Annotating(Agent):
        async def __call__(self, run: Run) -> None:
            run.trace.content = "x"
            run.trace.extra["turns"] = 3

    await check().run(Annotating())

    (exited,) = reporting.bodies("api", "POST", TRACE_EXIT)
    assert (exited["status"], exited["reward"], exited["metadata"]["turns"]) == (
        "completed",
        0.5,
        3,
    )
    evaluation = exited["evaluation_result"]
    assert evaluation["score"] == 0.5
    assert [(s["name"], s["value"]) for s in evaluation["subscores"]] == [("format", 0.5)]


async def test_an_error_the_run_recovered_from_stays_off_its_report(
    reporting: FakeServices,
) -> None:
    class Recovers(Agent):
        async def __call__(self, run: Run) -> None:
            run.record(Step(source="system", error="model busy; asked again"))
            run.trace.content = "yes"

    await Task(env="checked", id="check").run(Recovers(), runtime=LocalRuntime(checked()))

    (exited,) = reporting.bodies("api", "POST", TRACE_EXIT)
    assert (exited["status"], "error" in exited) == ("completed", False)


async def test_a_rollout_whose_agent_raised_reports_its_error_beside_its_grade(
    reporting: FakeServices,
) -> None:
    class RaisesAfter(Agent):
        async def __call__(self, run: Run) -> None:
            run.trace.content = "yes"
            raise RuntimeError("agent exploded")

    await Task(env="checked", id="check").run(RaisesAfter(), runtime=LocalRuntime(checked()))

    (exited,) = reporting.bodies("api", "POST", TRACE_EXIT)
    assert (exited["status"], exited["reward"], exited["error"]) == (
        "error",
        1.0,
        "[agent loop] RuntimeError: agent exploded",
    )


async def test_a_rollout_whose_grading_failed_reports_no_evaluation(
    reporting: FakeServices,
) -> None:
    env = Environment("broken")

    @env.template()
    async def grade_raises():
        yield "go"
        raise RuntimeError("grader exploded")
        yield 0.0

    await grade_raises().run(ScriptedAgent("x"))

    (exited,) = reporting.bodies("api", "POST", TRACE_EXIT)
    assert (exited["status"], exited["reward"], "evaluation_result" in exited) == (
        "error",
        0.0,
        False,
    )
    assert exited["error"].startswith("[grading] ")


async def test_a_rollout_that_never_launched_reports_where_it_failed(
    reporting: FakeServices,
) -> None:
    @asynccontextmanager
    async def unavailable(task: Task) -> AsyncIterator[Runtime]:
        raise RuntimeError("no substrate")
        yield Runtime("tcp://127.0.0.1:1")

    await rollout(Task(env="checked", id="check"), ScriptedAgent("yes"), runtime=unavailable)

    (exited,) = reporting.bodies("api", "POST", TRACE_EXIT)
    assert (exited["status"], exited["reward"]) == ("error", 0.0)
    assert exited["error"].startswith("[provisioning] ")
    assert "no substrate" in exited["error"]


async def test_a_timed_out_rollout_reports_why_it_stopped(reporting: FakeServices) -> None:
    class Waits(Agent):
        async def __call__(self, run: Run) -> None:
            run.trace.content = "yes"
            await asyncio.sleep(30)

    await Task(env="checked", id="check").run(
        Waits(AgentConfig(timeout_seconds=0.2)), runtime=LocalRuntime(checked())
    )

    (exited,) = reporting.bodies("api", "POST", TRACE_EXIT)
    assert (exited["status"], exited["stop_reason"], exited["reward"]) == ("error", "timeout", 1.0)


@pytest.mark.parametrize(
    "variables",
    [{"HUD_API_KEY": None}, {"HUD_TELEMETRY_ENABLED": "0"}],
    ids=["telemetry-on-without-a-key", "a-key-with-telemetry-off"],
)
async def test_nothing_is_reported_unless_telemetry_is_on_with_a_key(
    variables: dict[str, str | None], reporting: FakeServices, hud_env: HudEnv
) -> None:
    hud_env.set(**variables)

    job = await Task(env="checked", id="check").run(
        ScriptedAgent("yes"), runtime=LocalRuntime(checked())
    )

    assert (job.reward, reporting.requests()) == (1.0, [])


async def test_a_cancelled_rollout_reports_its_start_but_no_outcome(
    reporting: FakeServices,
) -> None:
    prompted = asyncio.Event()

    class Waits(Agent):
        async def __call__(self, run: Run) -> None:
            prompted.set()
            await asyncio.sleep(30)

    pending = asyncio.create_task(
        rollout(Task(env="checked", id="check"), Waits(), runtime=LocalRuntime(checked()))
    )
    await eventually(prompted.is_set)
    pending.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pending

    assert len(reporting.requests("api", "POST", TRACE_ENTER)) == 1
    assert reporting.requests("api", "POST", TRACE_EXIT) == []
