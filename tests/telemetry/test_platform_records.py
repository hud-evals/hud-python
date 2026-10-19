"""Platform reporting: every rollout reports under a job, through the HUD API's job
and trace enter/exit endpoints, and only while telemetry is on with an API key."""

from __future__ import annotations

import asyncio
import logging
import uuid
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

import pytest
from dirty_equals import IsStr

from hud.agents.base import Agent
from hud.agents.types import AgentConfig
from hud.eval import Job, LocalRuntime, Runtime, Task, Taskset, rollout
from tests.fixtures.agents import ScriptedAgent
from tests.fixtures.environments import lab, solve
from tests.support import eventually
from tests.support.platform import WEB_URL, Trace, key

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Awaitable, Callable

    from hud.eval import Run
    from tests.support import HudEnv, Platform

ADD = Task(env="lab", id="add", args={"a": 2, "b": 3})


def reported(platform: Platform, run: Run) -> Trace:
    assert run.trace_id is not None
    return platform.traces[key(run.trace_id)]


class Modelled(Agent):
    """Answers with ``answer(prompt)`` while sampling a named model."""

    def __init__(self, answer: Callable[[str], str] = solve, **extra: Any) -> None:
        super().__init__(AgentConfig(model="claude-test"))
        self.answer = answer
        self.extra = extra

    async def __call__(self, run: Run) -> None:
        run.trace.content = self.answer(run.prompt_text)
        run.trace.extra.update(self.extra)


async def test_a_taskset_reports_one_job_and_each_rollout_under_it(
    reporting: Platform, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger="hud.eval.job")
    rows = [ADD, Task(env="lab", id="add", args={"a": 1, "b": 1})]

    job = await Taskset("pair", rows, taskset_id="ts-1").run(
        Modelled(), runtime=LocalRuntime(lab()), group=2
    )

    assert list(reporting.jobs) == [key(job.id)]
    assert f"job: {WEB_URL}/jobs/{uuid.UUID(job.id)}" in caplog.messages
    assert len(reporting.traces) == len(job.runs) == 4
    assert {(trace.enter or {})["job_id"] for trace in reporting.traces.values()} == {job.id}


async def bare(runtime: LocalRuntime) -> Any:
    return await rollout(ADD, ScriptedAgent(solve), runtime=runtime)


async def threaded(runtime: LocalRuntime) -> Any:
    return await rollout(ADD, ScriptedAgent(solve), runtime=runtime, job_id="j" * 32)


async def session(runtime: LocalRuntime) -> Any:
    job = await Job.start("training", group=2)
    await ADD.run(ScriptedAgent(solve), runtime=runtime, job=job)
    return await ADD.run(ScriptedAgent(solve), runtime=runtime, job=job)


JOBS: dict[str, tuple[Callable[[LocalRuntime], Awaitable[Any]], list[dict[str, Any]]]] = {
    "a bare rollout registers a job of one named after its task": (
        bare,
        [{"name": "add", "group": 1}],
    ),
    "a rollout given a job registers none": (threaded, []),
    "an open job registers once for all its calls": (
        session,
        [{"name": "training", "group": 2}],
    ),
}


@pytest.mark.parametrize(("start", "bodies"), JOBS.values(), ids=JOBS.keys())
async def test_each_execution_registers_the_job_its_rollouts_report_under(
    start: Callable[[LocalRuntime], Awaitable[Any]],
    bodies: list[dict[str, Any]],
    reporting: Platform,
) -> None:
    await start(LocalRuntime(lab()))

    assert list(reporting.jobs.values()) == bodies
    reported_under = {key((trace.enter or {})["job_id"]) for trace in reporting.traces.values()}
    assert reported_under == (set(reporting.jobs) or {"j" * 32})


@asynccontextmanager
async def unavailable(task: Task) -> AsyncIterator[Runtime]:
    raise RuntimeError("no substrate")
    yield Runtime("tcp://127.0.0.1:1")


def subscored(score: float) -> Task:
    subscores = [
        {"name": "format", "weight": 0.5, "value": 1.0, "info": {"seen": True}},
        {"name": "content", "weight": 0.5, "value": 0.0},
    ]
    return Task(env="lab", id="frame", args={"result": {"score": score, "subscores": subscores}})


EXITS: dict[str, tuple[Task, Agent, bool, dict[str, Any]]] = {
    "a graded run": (
        ADD,
        Modelled(),
        True,
        {"status": "completed", "reward": 1.0, "evaluation_result": {"score": 1.0}},
    ),
    "a run whose agent raised": (
        ADD,
        ScriptedAgent(solve, fail_after=RuntimeError("agent exploded")),
        True,
        {
            "status": "error",
            "reward": 1.0,
            "error": "[agent loop] RuntimeError: agent exploded",
            "evaluation_result": {"score": 1.0},
        },
    ),
    "a run graded with subscores": (
        subscored(0.5),
        Modelled(lambda prompt: "x"),
        True,
        {
            "status": "completed",
            "reward": 0.5,
            "evaluation_result": {
                "score": 0.5,
                "subscores": [
                    {"name": "format", "weight": 0.5, "value": 1.0, "children": None},
                    {"name": "content", "weight": 0.5, "value": 0.0, "children": None},
                ],
            },
        },
    ),
    "a run whose grading failed": (
        Task(env="lab", id="grade_raises"),
        Modelled(lambda prompt: "x"),
        True,
        {
            "status": "error",
            "reward": 0.0,
            "error": IsStr(regex=r"\[grading\] .*grader exploded"),
        },
    ),
    "a run that never launched": (
        ADD,
        Modelled(),
        False,
        {"status": "error", "reward": 0.0, "error": "[provisioning] RuntimeError: no substrate"},
    ),
}


@pytest.mark.parametrize(("row", "agent", "placed", "body"), EXITS.values(), ids=EXITS.keys())
async def test_a_finished_rollout_reports_its_outcome(
    row: Task, agent: Agent, placed: bool, body: dict[str, Any], reporting: Platform
) -> None:
    run = await rollout(row, agent, runtime=LocalRuntime(lab()) if placed else unavailable)

    assert reported(reporting, run).exit == body


GATES = {
    "telemetry on without a key": {"HUD_API_KEY": None},
    "a key with telemetry off": {"HUD_TELEMETRY_ENABLED": "0"},
}


@pytest.mark.parametrize("variables", GATES.values(), ids=GATES.keys())
async def test_nothing_is_reported_unless_telemetry_is_on_with_a_key(
    variables: dict[str, str | None], reporting: Platform, hud_env: HudEnv
) -> None:
    hud_env.set(**variables)

    job = await ADD.run(ScriptedAgent(solve), runtime=LocalRuntime(lab()))

    assert job.reward == 1.0
    assert reporting.services.requests() == []


async def test_a_cancelled_rollout_reports_its_start_but_no_outcome(
    reporting: Platform,
) -> None:
    agent = ScriptedAgent(solve, delay=30)
    pending = asyncio.create_task(rollout(ADD, agent, runtime=LocalRuntime(lab())))
    await eventually(lambda: agent.prompts != [])

    pending.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pending

    (trace,) = reporting.traces.values()
    assert (trace.enter is not None, trace.exit) == (True, None)
