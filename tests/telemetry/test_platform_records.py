"""What the platform records about an execution: one job for it, a trace entered for each
rollout under that job, and each rollout's outcome when it exits, sent only while
telemetry is on with an API key."""

from __future__ import annotations

import logging
import uuid
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

import pytest

from hud import Environment
from hud.agents.base import Agent
from hud.agents.types import AgentConfig
from hud.eval import Job, LocalRuntime, Runtime, Task, Taskset, rollout
from tests.fixtures.agents import ScriptedAgent
from tests.fixtures.environments import checked

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from hud.eval import Run
    from tests.conftest import HudEnv
    from tests.fixtures.services import FakeServices

JOB_ENTER = "/v2/trace/job/{id}/enter"
TRACE_ENTER = "/v2/trace/{id}/enter"
TRACE_EXIT = "/v2/trace/{id}/exit"


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
