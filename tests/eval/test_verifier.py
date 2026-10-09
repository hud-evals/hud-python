"""Verifier tasks: the grade of record comes from a second, agent-less task.

The actor (``solve``) grades 0.25 and hands its answer on; the judge (``verify``)
pays 1.0 only when that answer was ``secret``. A reward of 1.0 therefore proves
the actor's result reached the verifier, through its grade frame or, across
containers, through the session files the runtime transferred.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import pytest
from dirty_equals import IsStr
from pydantic import BaseModel

from hud import Environment
from hud.eval import (
    DockerRuntime,
    LocalRuntime,
    Runtime,
    RuntimeConfig,
    RuntimeLimits,
    Shared,
    Task,
)
from tests.eval.envs import actor, containers, eventually, judge
from tests.harness import RecordingProvider, ScriptedAgent, steps

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path

    from hud.eval import Provider
    from tests.harness import FakeDocker

OWN_LIMITS = RuntimeConfig(limits=RuntimeLimits(run_timeout_s=20))


def by_env(providers: dict[str, Provider]) -> Provider:
    """Place each row with the provider for its environment."""

    def place(task: Task) -> Any:
        return providers[task.env](task)

    return place


def portable(provider: Provider) -> Provider:
    """Place rows without their runtime_config, as a provider that cannot size them would."""

    def place(task: Task) -> Any:
        return provider(task.model_copy(update={"runtime_config": None}))

    return place


@asynccontextmanager
async def unavailable(task: Task) -> AsyncIterator[Runtime]:
    raise RuntimeError("verifier unavailable")
    yield Runtime("tcp://127.0.0.1:1")


@dataclass
class Case:
    placement: str
    expected: dict[str, Any]
    agent: ScriptedAgent
    grade: str = "score"
    verdict: str = "check"


SEPARATE = ["start:solve", "stop:solve", "start:verify", "stop:verify"]
SHARED = ["start:solve", "stop:solve"]
EVALUATED = ["solve", "verify"]


def graded(**changes: Any) -> dict[str, Any]:
    return {
        "reward": 1.0,
        "raw": {"score": 1.0},
        "status": "completed",
        "error": None,
        "placements": SEPARATE,
        "evaluated": EVALUATED,
        "in_errors": False,
        "fallback_warning": False,
    } | changes


def ungraded(error: Any, **changes: Any) -> dict[str, Any]:
    return graded(reward=0.0, raw={}, status="error", error=error, in_errors=True) | changes


ANSWER = ScriptedAgent("secret")
CASES = {
    "a verifier on its own env is placed after the actor is torn down": Case(
        "separate", graded(), ANSWER
    ),
    "a verifier on the actor's env without a runtime shares its substrate": Case(
        "same env", graded(placements=SHARED), ANSWER
    ),
    "a verifier with its own runtime runs beside an actor that cannot transport its session": (
        Case(
            "same env, own runtime",
            graded(placements=SHARED, fallback_warning=True),
            ANSWER,
        )
    ),
    "session files reach a verifier in another container": Case("containers", graded(), ANSWER),
    "session files reach a verifier in a fresh container of the same env": Case(
        "same env, own container", graded(), ANSWER
    ),
    "a one-wide pool releases the actor's lease before the verifier takes one": Case(
        "pooled container", graded(placements=["start:solve", "stop:solve"]), ANSWER
    ),
    "the verifier stays authoritative when the agent raised after answering": Case(
        "separate",
        graded(status="error", error="[agent loop] RuntimeError: agent exploded"),
        ScriptedAgent("secret", fail_after=RuntimeError("agent exploded")),
    ),
    "the verifier stays authoritative when actor grading raised": Case(
        "separate",
        graded(
            status="error",
            error=IsStr(regex=r"\[grading\] .*actor grade exploded"),
            evaluated=["verify"],
        ),
        ANSWER,
        grade="raise",
    ),
    "the verifier stays authoritative over a scoreless actor frame": Case(
        "separate",
        graded(
            status="error",
            error=IsStr(regex=r"\[grading\] .*numeric 'score'.*"),
            evaluated=["verify"],
        ),
        ANSWER,
        grade="scoreless",
    ),
    "a verifier that cannot be placed leaves the run ungraded": Case(
        "verifier unavailable",
        ungraded(
            "[provisioning verifier] RuntimeError: verifier unavailable",
            placements=["start:solve", "stop:solve", "start:verify", "stop:verify"],
            evaluated=["solve"],
        ),
        ANSWER,
    ),
    "a verifier that raises leaves the run ungraded": Case(
        "separate",
        ungraded(IsStr(regex=r"\[verifying\] .*verifier exploded"), evaluated=["solve"]),
        ANSWER,
        verdict="raise",
    ),
    "a scoreless verifier frame leaves the run ungraded": Case(
        "separate",
        ungraded(IsStr(regex=r"\[verifying\] .*numeric 'score'.*"), evaluated=["solve"]),
        ANSWER,
        verdict="scoreless",
    ),
    "a zero verdict on an errored run still counts toward the job": Case(
        "separate",
        graded(
            reward=0.0,
            raw={"score": 0.0},
            status="error",
            error="[agent loop] RuntimeError: agent exploded",
        ),
        ScriptedAgent("secret", fail_after=RuntimeError("agent exploded")),
        verdict="zero",
    ),
}


def arrange(case: Case, rootfs: Path) -> tuple[Task, Provider, dict[str, Environment]]:
    """The row, the provider placing it, and the container images it needs."""
    if case.placement in {"separate", "verifier unavailable"}:
        row = Task(env="actor", id="solve", verifier=Task(env="judge", id="verify"))
        judge_provider = (
            unavailable
            if case.placement == "verifier unavailable"
            else LocalRuntime(judge(verdict=case.verdict))
        )
        provider = by_env({"actor": LocalRuntime(actor(grade=case.grade)), "judge": judge_provider})
        return row, provider, {}
    if case.placement in {"same env", "same env, own runtime"}:
        env = judge(actor(Environment("reviewed"), grade=case.grade), verdict=case.verdict)
        own = OWN_LIMITS if case.placement == "same env, own runtime" else None
        row = Task(
            env="reviewed",
            id="solve",
            verifier=Task(env="reviewed", id="verify", runtime_config=own),
        )
        return row, portable(LocalRuntime(env)), {}
    if case.placement == "containers":
        row = Task(
            env="actor",
            id="solve",
            runtime_config=RuntimeConfig(image="actor"),
            verifier=Task(env="judge", id="verify", runtime_config=RuntimeConfig(image="judge")),
        )
        images = {
            "actor": actor(sessions=rootfs / "actor"),
            "judge": judge(sessions=rootfs / "judge"),
        }
        return row, DockerRuntime(), images
    image = RuntimeConfig(image="reviewed")
    row = Task(
        env="reviewed",
        id="solve",
        runtime_config=image,
        verifier=Task(env="reviewed", id="verify", runtime_config=image),
    )
    sessions = rootfs / "reviewed"
    env = judge(actor(Environment("reviewed"), sessions=sessions), sessions=sessions)
    return row, DockerRuntime(), {"reviewed": env}


@pytest.mark.parametrize("case", CASES.values(), ids=CASES.keys())
async def test_the_verifier_grade_is_the_grade_of_record(
    case: Case,
    tmp_path: Path,
    fake_docker: FakeDocker,
    caplog: pytest.LogCaptureFixture,
) -> None:
    rootfs = tmp_path / "rootfs"
    row, inner, images = arrange(case, rootfs)
    recording = RecordingProvider(inner)
    provider: Provider = (
        Shared(recording, width=1) if case.placement == "pooled container" else recording
    )

    async with containers(fake_docker, rootfs, images):
        job = await row.run(case.agent, runtime=provider, rollout_timeout=30)
    (run,) = job.runs

    assert {
        "reward": run.reward,
        "raw": run.grade.raw,
        "status": run.trace.status,
        "error": run.trace.error,
        "placements": [f"{event}:{slug}" for event, slug in recording.events],
        "evaluated": [
            step["task_call"]["name"]
            for step in steps(run.trace_id)
            if step["source"] == "task" and step["task_call"]["phase"] == "evaluate"
        ],
        "in_errors": run in job.errors,
        "fallback_warning": "verifier runtime_config ignored" in caplog.text,
    } == case.expected


class ActorResult(BaseModel):
    score: float
    answer: str


async def test_the_verifier_receives_the_actor_grade_as_its_declared_type() -> None:
    received: list[Any] = []
    env = actor(Environment("reviewed"))

    @env.template(returns=ActorResult)
    async def verify():
        result = yield ""
        received.append(result.content)
        yield 1.0

    row = Task(env="reviewed", id="solve", verifier=Task(env="reviewed", id="verify"))
    job = await row.run(ScriptedAgent("secret"), runtime=LocalRuntime(env))

    assert received == [ActorResult(score=0.25, answer="secret")]
    assert job.reward == 1.0


def held_teardown(
    providers: dict[str, Provider], held: str, release: asyncio.Event, log: list[str]
) -> Provider:
    """Places rows by env; teardown of the ``held`` env waits for ``release``."""

    @asynccontextmanager
    async def place(task: Task) -> AsyncIterator[Runtime]:
        log.append(f"start:{task.env}")
        try:
            async with providers[task.env](task) as runtime:
                yield runtime
        finally:
            if task.env == held:
                await release.wait()
            log.append(f"stop:{task.env}")

    return place


DEADLINES: dict[str, tuple[str, float, str, list[str]]] = {
    "a deadline in the actor's teardown never starts the verifier": (
        "actor",
        0.0,
        "rollout timed out after 0.2s during actor cleanup",
        ["start:actor", "stop:actor"],
    ),
    "a deadline in the verifier's teardown keeps the verdict": (
        "judge",
        1.0,
        "rollout timed out after 0.2s during cleanup",
        ["start:actor", "stop:actor", "start:judge", "stop:judge"],
    ),
}


@pytest.mark.parametrize(
    ("held", "reward", "error", "placements"), DEADLINES.values(), ids=DEADLINES.keys()
)
async def test_a_deadline_during_teardown_returns_and_lets_teardown_finish(
    held: str, reward: float, error: str, placements: list[str]
) -> None:
    release = asyncio.Event()
    log: list[str] = []
    provider = held_teardown(
        {"actor": LocalRuntime(actor()), "judge": LocalRuntime(judge())}, held, release, log
    )
    row = Task(env="actor", id="solve", verifier=Task(env="judge", id="verify"))

    job = await row.run(ScriptedAgent("secret"), runtime=provider, rollout_timeout=0.2)
    assert log[-1] == f"start:{held}"
    release.set()
    await eventually(lambda: log[-1] == f"stop:{held}")
    await asyncio.sleep(0)

    (run,) = job.runs
    assert (run.reward, run.trace.stop_reason, run.trace.error) == (reward, "timeout", error)
    assert log == placements


SESSION_IDS = ["", ".", "..", "../actor", "a/b", "a\\b"]


@pytest.mark.parametrize("session_id", SESSION_IDS)
async def test_a_runtime_rejects_a_session_id_that_is_not_one_path_component(
    session_id: str, tmp_path: Path
) -> None:
    runtime = Runtime("tcp://127.0.0.1:1")

    with pytest.raises(ValueError, match="runtime session id must be a single path component"):
        async with runtime.snapshot_session(session_id):
            pass
    with pytest.raises(ValueError, match="runtime session id must be a single path component"):
        await runtime.restore_session(session_id, tmp_path)
