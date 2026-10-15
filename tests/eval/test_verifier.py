"""Verifier tasks: the grade of record comes from a second, agent-less task.

The actor (``solve``) grades 0.25 and hands its answer on; the judge (``verify``)
pays 1.0 only when that answer was ``secret``. A reward of 1.0 therefore proves
the actor's result reached the verifier through its grade frame. Session files
moving between containers are covered with real containers in ``test_docker``.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

import pytest
from pydantic import BaseModel

from hud import Environment
from hud.eval import (
    LocalRuntime,
    Runtime,
    RuntimeConfig,
    RuntimeLimits,
    Task,
)
from tests.eval.envs import actor, judge
from tests.harness import RecordingProvider, ScriptedAgent, eventually, steps

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path

    from hud.eval import Provider

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


def separate(
    *, grade: str = "score", verdict: str = "check", judged: Provider | None = None
) -> tuple[Task, Provider]:
    """An actor row whose verifier runs on its own ``judge`` env."""
    row = Task(env="actor", id="solve", verifier=Task(env="judge", id="verify"))
    judging = judged or LocalRuntime(judge(verdict=verdict))
    return row, by_env({"actor": LocalRuntime(actor(grade=grade)), "judge": judging})


def evaluated(run: Any) -> list[str]:
    """The tasks whose grade the trace records, in order."""
    return [
        step["task_call"]["name"]
        for step in steps(run.trace_id)
        if step["source"] == "task" and step["task_call"]["phase"] == "evaluate"
    ]


@pytest.mark.parametrize(
    ("own_runtime", "same_env", "placements", "warned"),
    [
        pytest.param(
            False,
            False,
            ["start:solve", "stop:solve", "start:verify", "stop:verify"],
            False,
            id="on-its-own-env-after-the-actor",
        ),
        pytest.param(
            False,
            True,
            ["start:solve", "stop:solve"],
            False,
            id="on-the-actors-env-sharing-its-substrate",
        ),
        pytest.param(
            True,
            True,
            ["start:solve", "stop:solve"],
            True,
            id="with-its-own-runtime-beside-an-actor-that-cannot-move-its-session",
        ),
    ],
)
async def test_the_verifier_grade_is_the_grade_of_record_wherever_it_runs(
    own_runtime: bool,
    same_env: bool,
    placements: list[str],
    warned: bool,
    caplog: pytest.LogCaptureFixture,
) -> None:
    if same_env:
        own = OWN_LIMITS if own_runtime else None
        row = Task(
            env="reviewed",
            id="solve",
            verifier=Task(env="reviewed", id="verify", runtime_config=own),
        )
        inner = portable(LocalRuntime(judge(actor(Environment("reviewed")))))
    else:
        row, inner = separate()
    recording = RecordingProvider(inner)

    job = await row.run(ScriptedAgent("secret"), runtime=recording, rollout_timeout=30)

    (run,) = job.runs
    assert (run.reward, run.trace.status, run in job.errors) == (1.0, "completed", False)
    assert [f"{event}:{slug}" for event, slug in recording.events] == placements
    assert evaluated(run) == ["solve", "verify"]
    assert ("verifier runtime_config ignored" in caplog.text) is warned


@pytest.mark.parametrize(
    ("agent", "grade", "phase", "graded"),
    [
        pytest.param(
            ScriptedAgent("secret", fail_after=RuntimeError("agent exploded")),
            "score",
            "[agent loop]",
            ["solve", "verify"],
            id="the-agent-raised-after-answering",
        ),
        pytest.param(
            ScriptedAgent("secret"), "raise", "[grading]", ["verify"], id="actor-grading-raised"
        ),
        pytest.param(
            ScriptedAgent("secret"),
            "scoreless",
            "[grading]",
            ["verify"],
            id="a-scoreless-actor-grade",
        ),
    ],
)
async def test_the_verifier_stays_authoritative_when_the_actor_side_fails(
    agent: ScriptedAgent, grade: str, phase: str, graded: list[str]
) -> None:
    row, provider = separate(grade=grade)

    job = await row.run(agent, runtime=provider, rollout_timeout=30)

    (run,) = job.runs
    assert (run.reward, run.trace.status, run in job.errors) == (1.0, "error", False)
    assert str(run.trace.error).startswith(phase)
    assert evaluated(run) == graded


@pytest.mark.parametrize(
    ("verdict", "judged", "phase"),
    [
        pytest.param("raise", None, "[verifying]", id="the-verifier-raised"),
        pytest.param("scoreless", None, "[verifying]", id="a-scoreless-verdict"),
        pytest.param(
            "check", unavailable, "[provisioning verifier]", id="the-verifier-could-not-be-placed"
        ),
    ],
)
async def test_a_failed_verifier_leaves_the_run_ungraded(
    verdict: str, judged: Provider | None, phase: str
) -> None:
    row, provider = separate(verdict=verdict, judged=judged)

    job = await row.run(ScriptedAgent("secret"), runtime=provider, rollout_timeout=30)

    (run,) = job.runs
    assert (run.reward, run.grade.raw, run.trace.status, run in job.errors) == (
        0.0,
        {},
        "error",
        True,
    )
    assert str(run.trace.error).startswith(phase)
    assert evaluated(run) == ["solve"]


async def test_a_zero_verdict_on_an_errored_run_still_counts_toward_the_job() -> None:
    row, provider = separate(verdict="zero")
    agent = ScriptedAgent("secret", fail_after=RuntimeError("agent exploded"))

    job = await row.run(agent, runtime=provider, rollout_timeout=30)

    (run,) = job.runs
    assert (run.reward, run.grade.raw, run.trace.status, run in job.errors) == (
        0.0,
        {"score": 0.0},
        "error",
        False,
    )


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
