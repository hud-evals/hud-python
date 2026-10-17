"""Verifier tasks: the grade of record comes from a second, agent-less task.

The actor (``solve``) grades 0.25 and hands its answer on; the judge (``verify``)
pays 1.0 only when that answer was ``secret``. A reward of 1.0 therefore proves
the actor's result reached the verifier through its grade frame. A verifier
with a runtime of its own receives the actor's session files on every runtime
that can move them.
"""

from __future__ import annotations

import contextlib
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

import pytest
from pydantic import BaseModel

from hud import Environment
from hud.eval import (
    HUDRuntime,
    LocalRuntime,
    Runtime,
    RuntimeConfig,
    RuntimeLimits,
    Task,
)
from tests.fixtures.envs import source
from tests.support import RecordingProvider, ScriptedAgent, actor, eventually, judge, steps
from tests.support.runtime import host_on_runtime

from .placements import PLACEMENTS, World, rows

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path

    from hud.eval import Provider
    from tests.support import FakeServices, HudEnv

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
    ("own_runtime", "same_env", "tunneled", "placements", "warned"),
    [
        pytest.param(
            False,
            False,
            False,
            ["start:solve", "stop:solve", "start:verify", "stop:verify"],
            False,
            id="on-its-own-env-after-the-actor",
        ),
        pytest.param(
            False,
            True,
            False,
            ["start:solve", "stop:solve"],
            False,
            id="on-the-actors-env-sharing-its-substrate",
        ),
        pytest.param(
            True,
            True,
            False,
            ["start:solve", "stop:solve"],
            True,
            id="with-its-own-runtime-beside-an-actor-that-cannot-move-its-session",
        ),
        pytest.param(
            True,
            True,
            True,
            ["start:solve", "stop:solve"],
            True,
            id="with-its-own-runtime-beside-an-actor-on-the-hud-tunnel",
        ),
    ],
)
async def test_the_verifier_grade_is_the_grade_of_record_wherever_it_runs(
    own_runtime: bool,
    same_env: bool,
    tunneled: bool,
    placements: list[str],
    warned: bool,
    caplog: pytest.LogCaptureFixture,
    services: FakeServices,
    hud_env: HudEnv,
) -> None:
    reviewed = LocalRuntime(judge(actor(Environment("reviewed"))))
    if same_env:
        own = OWN_LIMITS if own_runtime else None
        row = Task(
            env="reviewed",
            id="solve",
            verifier=Task(env="reviewed", id="verify", runtime_config=own),
        )
        inner: Provider = portable(reviewed)
    else:
        row, inner = separate()

    async with contextlib.AsyncExitStack() as stack:
        if tunneled:
            hud_env.set(HUD_API_KEY="k")
            served = await stack.enter_async_context(reviewed(Task(env="reviewed", id="serve")))
            host_on_runtime(services, int(served.url.rsplit(":", 1)[1]))
            inner = HUDRuntime()
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


SESSION_FILES = {
    "files": (False, 1.0),
    "a link": (True, 0.0),
}


@pytest.mark.parametrize("placement", rows("session-transport"))
@pytest.mark.parametrize(("link", "reward"), SESSION_FILES.values(), ids=SESSION_FILES.keys())
async def test_a_verifier_in_its_own_sandbox_receives_the_actors_session_files(
    link: bool,
    reward: float,
    placement: str,
    services: FakeServices,
    hud_env: HudEnv,
    request: pytest.FixtureRequest,
) -> None:
    """The verifier's own limits give it its own sandbox, so the actor's files must move;
    a session holding anything but files and directories does not."""
    own = RuntimeConfig(limits=RuntimeLimits(startup_timeout_s=600))
    row = Task(
        env="sessions",
        id="solve",
        args={"link": link},
        verifier=Task(env="sessions", id="verify", runtime_config=own),
    )
    world = World(source("sessions"), services, hud_env, request)

    async with PLACEMENTS[placement].open(world) as runtime:
        run = (await row.run(ScriptedAgent("secret"), runtime=runtime)).runs[0]
        await eventually(lambda: PLACEMENTS[placement].clean(world))

    assert run.reward == reward
    assert ("runtime session contains" in (run.trace.error or "")) is link


SESSION_IDS = ["", "..", "a/b", "a\\b"]


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
