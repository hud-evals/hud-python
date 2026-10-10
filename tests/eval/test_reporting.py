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
from inline_snapshot import snapshot

from hud.agents.base import Agent
from hud.agents.types import AgentConfig
from hud.eval import Job, LocalRuntime, Runtime, Task, Taskset, rollout
from hud.telemetry import flush
from hud.telemetry.context import get_trace_headers, set_trace_context
from tests.eval.envs import eventually, lab, solve
from tests.harness import Reply, ScriptedAgent

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Awaitable, Callable, Iterator

    from hud.eval import Run
    from tests.harness import FakeServices, HudEnv

HEX_ID = IsStr(regex=r"[0-9a-f]{32}")
ADD = Task(env="lab", id="add", args={"a": 2, "b": 3})
JOB_ENTER = "/v2/trace/job/{id}/enter"
TRACE_ENTER = "/v2/trace/{id}/enter"
TRACE_EXIT = "/v2/trace/{id}/exit"


@pytest.fixture
def platform(services: FakeServices, hud_env: HudEnv) -> Iterator[FakeServices]:
    """The fake platform accepting every report, with telemetry on and a key set.

    Span uploads drain before the test ends, so none reaches a later test's platform.
    """
    hud_env.set(HUD_API_KEY="k", HUD_TELEMETRY_ENABLED="1", HUD_WEB_URL="https://hud.test")
    for path in (JOB_ENTER, TRACE_ENTER, TRACE_EXIT):
        services.route("api", "POST", path, json={})
    services.route("telemetry", "POST", "/trace/{id}/telemetry-upload", json={})
    yield services
    assert flush(timeout=10)


class Modelled(Agent):
    """Answers with ``answer(prompt)`` while sampling a named model."""

    def __init__(self, answer: Callable[[str], str] = solve, **extra: Any) -> None:
        super().__init__(AgentConfig(model="claude-test"))
        self.answer = answer
        self.extra = extra
        self.headers: list[dict[str, str]] = []

    async def __call__(self, run: Run) -> None:
        self.headers.append(get_trace_headers())
        run.trace.content = self.answer(run.prompt_text)
        run.trace.extra.update(self.extra)


async def test_a_taskset_reports_one_job_and_each_rollout_under_it(
    platform: FakeServices, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger="hud.eval.job")
    rows = [ADD, Task(env="lab", id="add", args={"a": 1, "b": 1})]

    job = await Taskset("pair", rows, taskset_id="ts-1").run(
        Modelled(), runtime=LocalRuntime(lab()), group=2
    )

    (job_enter,) = platform.requests("api", "POST", JOB_ENTER)
    assert (job_enter.params["id"], job_enter.bearer) == (job.id, "k")
    assert job_enter.json == snapshot(
        {"name": "pair (2 tasks) (2 times)", "group": 2, "taskset_id": "ts-1"}
    )
    assert f"job: https://hud.test/jobs/{uuid.UUID(job.id)}" in caplog.messages
    enters = platform.requests("api", "POST", TRACE_ENTER)
    assert len(enters) == 4
    assert {request.params["id"] for request in enters} == {run.trace_id for run in job.runs}
    bodies = [request.json for request in enters]
    assert sorted(body.pop("task_slug") for body in bodies) == sorted(
        row.slug for row in rows for _ in range(2)
    )
    assert bodies == [{"job_id": job.id, "group_id": HEX_ID, "model": "claude-test"}] * 4
    assert {body["job_id"] for body in platform.bodies("api", "POST", TRACE_ENTER)} == {job.id}
    exits = platform.requests("api", "POST", TRACE_EXIT)
    assert len(exits) == 4
    assert {request.params["id"] for request in exits} == {run.trace_id for run in job.runs}
    assert [request.json for request in exits] == [
        {"status": "completed", "reward": 1.0, "evaluation_result": {"score": 1.0}}
    ] * 4


async def bare(runtime: LocalRuntime) -> Any:
    return await rollout(ADD, ScriptedAgent(solve), runtime=runtime)


async def threaded(runtime: LocalRuntime) -> Any:
    return await rollout(ADD, ScriptedAgent(solve), runtime=runtime, job_id="j" * 32)


async def single(runtime: LocalRuntime) -> Any:
    return await ADD.run(ScriptedAgent(solve), runtime=runtime)


async def grouped(runtime: LocalRuntime) -> Any:
    return await ADD.run(ScriptedAgent(solve), runtime=runtime, group=2)


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
    "a single task's job is named after the task": (single, [{"name": "add", "group": 1}]),
    "a grouped task's job counts its repeats": (
        grouped,
        [{"name": "add (2 times)", "group": 2}],
    ),
    "an open job registers once for all its calls": (
        session,
        [{"name": "training", "group": 2}],
    ),
}


@pytest.mark.parametrize(("start", "bodies"), JOBS.values(), ids=JOBS.keys())
async def test_each_execution_registers_the_job_its_rollouts_report_under(
    start: Callable[[LocalRuntime], Awaitable[Any]],
    bodies: list[dict[str, Any]],
    platform: FakeServices,
) -> None:
    await start(LocalRuntime(lab()))

    assert platform.bodies("api", "POST", JOB_ENTER) == bodies
    job_ids = {request.params["id"] for request in platform.requests("api", "POST", JOB_ENTER)}
    assert {body["job_id"] for body in platform.bodies("api", "POST", TRACE_ENTER)} == (
        job_ids or {"j" * 32}
    )


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
    "a run cut off by its agent deadline": (
        ADD.model_copy(update={"agent_config": {"timeout_seconds": 0.05}}),
        ScriptedAgent(solve, linger=30),
        True,
        {
            "status": "error",
            "reward": 1.0,
            "error": "agent timed out after 0.05s",
            "evaluation_result": {"score": 1.0},
            "stop_reason": "timeout",
        },
    ),
    "a run carrying trajectory metadata": (
        ADD,
        Modelled(turns=3),
        True,
        {
            "status": "completed",
            "reward": 1.0,
            "evaluation_result": {"score": 1.0},
            "metadata": {"turns": 3},
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
    row: Task, agent: Agent, placed: bool, body: dict[str, Any], platform: FakeServices
) -> None:
    run = await rollout(row, agent, runtime=LocalRuntime(lab()) if placed else unavailable)

    (exit_request,) = platform.requests("api", "POST", TRACE_EXIT)
    assert exit_request.params["id"] == run.trace_id
    assert exit_request.json == body


PARENTS = {
    "a rollout inside another trace names it as its parent": (
        "a" * 32,
        "b" * 32,
        {"Trace-Id": "b" * 32, "X-HUD-Parent-Trace-Id": "a" * 32},
        "a" * 32,
    ),
    "an ambient trace that is the rollout's own is not its parent": (
        str(uuid.UUID("b" * 32)),
        "b" * 32,
        {"Trace-Id": "b" * 32},
        None,
    ),
}


@pytest.mark.parametrize(
    ("ambient", "trace_id", "headers", "parent"), PARENTS.values(), ids=PARENTS.keys()
)
async def test_a_nested_rollout_reports_and_propagates_its_parent_trace(
    ambient: str,
    trace_id: str,
    headers: dict[str, str],
    parent: str | None,
    platform: FakeServices,
) -> None:
    agent = Modelled()

    with set_trace_context(ambient):
        run = await rollout(ADD, agent, runtime=LocalRuntime(lab()), trace_id=trace_id)

    (enter,) = platform.bodies("api", "POST", TRACE_ENTER)
    assert (run.trace_id, agent.headers) == (trace_id, [headers])
    assert enter.get("parent_trace_id") == parent


RETRIES = {
    "a rejected report is retried once": ([Reply(status=500), Reply(json={})], 2, []),
    "a report rejected twice is given up with a warning": (
        [Reply(status=500)],
        2,
        [IsStr(regex=r"platform report /trace/job/[0-9a-f]{32}/enter failed: .*500.*")],
    ),
    "an accepted report is sent once": ([Reply(json={})], 1, []),
}


@pytest.mark.parametrize(("replies", "attempts", "warnings"), RETRIES.values(), ids=RETRIES.keys())
async def test_reporting_never_fails_the_run(
    replies: list[Reply],
    attempts: int,
    warnings: list[Any],
    platform: FakeServices,
    caplog: pytest.LogCaptureFixture,
) -> None:
    platform.route("api", "POST", JOB_ENTER, *replies)

    job = await Job.start("flaky")

    assert len(platform.requests("api", "POST", JOB_ENTER)) == attempts
    assert [
        record.getMessage()
        for record in caplog.records
        if record.name == "hud.eval.job" and record.levelno == logging.WARNING
    ] == warnings
    assert job.name == "flaky"


GATES = {
    "telemetry on without a key": {"HUD_TELEMETRY_ENABLED": "1", "HUD_API_KEY": None},
    "a key with telemetry off": {"HUD_TELEMETRY_ENABLED": "0", "HUD_API_KEY": "k"},
}


@pytest.mark.parametrize("variables", GATES.values(), ids=GATES.keys())
async def test_nothing_is_reported_unless_telemetry_is_on_with_a_key(
    variables: dict[str, str | None], services: FakeServices, hud_env: HudEnv
) -> None:
    hud_env.set(**variables)

    job = await ADD.run(ScriptedAgent(solve), runtime=LocalRuntime(lab()))

    assert job.reward == 1.0
    assert services.requests() == []


async def test_a_cancelled_rollout_reports_its_start_but_no_outcome(
    platform: FakeServices,
) -> None:
    agent = ScriptedAgent(solve, delay=30)
    pending = asyncio.create_task(rollout(ADD, agent, runtime=LocalRuntime(lab())))
    await eventually(lambda: agent.prompts != [])

    pending.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pending

    assert len(platform.requests("api", "POST", TRACE_ENTER)) == 1
    assert platform.requests("api", "POST", TRACE_EXIT) == []
