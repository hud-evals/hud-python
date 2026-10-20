"""Verifier tasks: an agent-less second task whose grade is the run's grade of record.

The actor's ``solve`` grades 0.25 and hands its answer on; the judge's ``verify`` pays
1.0 only when that answer was ``secret``. A reward of 1.0 proves the actor's result
reached the verifier.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

import pytest
from pydantic import BaseModel

from hud import Environment
from hud.agents.base import Agent
from hud.eval import LocalRuntime, Runtime, RuntimeConfig, RuntimeLimits, Task
from tests.conftest import eventually
from tests.fixtures.agents import ScriptedAgent

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from hud.eval import Run


def verified(actor: Environment, judge: Environment) -> Task:
    """``actor``'s ``solve`` task, with ``judge``'s ``verify`` as its verifier."""

    @actor.template()
    async def solve():
        answer = yield "answer secret"
        yield {"score": 0.25, "answer": answer}

    @judge.template()
    async def verify():
        result = yield ""
        yield 1.0 if result["answer"] == "secret" else 0.0

    task = solve()
    task.verifier = verify()
    return task


async def test_a_verifier_on_its_own_env_runs_after_the_actors_env_is_torn_down() -> None:
    events = []
    actor = Environment("actor")
    judge = Environment("judge")

    @actor.initialize
    async def actor_up() -> None:
        events.append("actor up")

    @actor.shutdown
    async def actor_down() -> None:
        events.append("actor down")

    @judge.initialize
    async def judge_up() -> None:
        events.append("judge up")

    @judge.shutdown
    async def judge_down() -> None:
        events.append("judge down")

    job = await verified(actor, judge).run(ScriptedAgent("secret"))

    (run,) = job.runs
    assert (run.reward, run.trace.status) == (1.0, "completed")
    assert events == ["actor up", "actor down", "judge up", "judge down"]


async def test_a_verifier_on_the_actors_env_grades_on_the_same_substrate() -> None:
    events = []
    env = Environment("reviewed")

    @env.initialize
    async def up() -> None:
        events.append("up")

    @env.shutdown
    async def down() -> None:
        events.append("down")

    @env.template()
    async def solve():
        answer = yield "answer secret"
        events.append("actor graded")
        yield {"score": 0.25, "answer": answer}

    @env.template()
    async def verify():
        result = yield ""
        events.append("verifier graded")
        yield 1.0 if result["answer"] == "secret" else 0.0

    task = solve()
    task.verifier = verify()
    job = await task.run(ScriptedAgent("secret"))

    assert job.runs[0].reward == 1.0
    assert events == ["up", "actor graded", "verifier graded", "down"]


async def test_a_verifier_asking_for_its_own_runtime_runs_beside_an_actor_that_cannot_move(
    caplog: pytest.LogCaptureFixture,
) -> None:
    events = []
    env = Environment("reviewed")

    @env.initialize
    async def up() -> None:
        events.append("up")

    @env.template()
    async def solve():
        answer = yield "answer secret"
        yield {"score": 0.25, "answer": answer}

    @env.template()
    async def verify():
        result = yield ""
        yield 1.0 if result["answer"] == "secret" else 0.0

    task = solve()
    task.verifier = verify()
    task.verifier.runtime_config = RuntimeConfig(limits=RuntimeLimits(run_timeout_s=20))
    async with LocalRuntime(env)(task.model_copy(update={"verifier": None})) as served:
        job = await task.run(ScriptedAgent("secret"), runtime=Runtime(served.url))

    assert (job.runs[0].reward, events) == (1.0, ["up"])
    assert "verifier runtime_config ignored: Runtime cannot transport the actor's session" in (
        caplog.text
    )


@pytest.mark.parametrize(
    ("grading", "verdict", "reward", "error", "graded"),
    [
        pytest.param(
            "raise",
            "check",
            1.0,
            ("[grading] ", "actor grade exploded"),
            True,
            id="the-actors-grader-raised",
        ),
        pytest.param(
            "scoreless",
            "check",
            1.0,
            ("[grading] ", "graded with a dict missing a numeric 'score' (keys: ['answer'])"),
            True,
            id="a-scoreless-actor-grade",
        ),
        pytest.param(
            "score",
            "raise",
            0.0,
            ("[verifying] ", "verifier exploded"),
            False,
            id="the-verifier-raised",
        ),
        pytest.param(
            "score",
            "scoreless",
            0.0,
            ("[verifying] ", "graded with a dict missing a numeric 'score' (keys: ['verdict'])"),
            False,
            id="a-scoreless-verdict",
        ),
        pytest.param("score", "zero", 0.0, None, True, id="a-zero-verdict"),
    ],
)
async def test_the_verdict_is_the_grade_whichever_side_fails(
    grading: str, verdict: str, reward: float, error: tuple[str, str] | None, graded: bool
) -> None:
    actor = Environment("actor")
    judge = Environment("judge")

    @actor.template()
    async def solve(grading: str):
        answer = yield "answer secret"
        if grading == "raise":
            raise RuntimeError("actor grade exploded")
        yield {"score": 0.25, "answer": answer} if grading == "score" else {"answer": answer}

    @judge.template()
    async def verify(verdict: str):
        result = yield ""
        if verdict == "raise":
            raise RuntimeError("verifier exploded")
        if verdict == "scoreless":
            yield {"verdict": "pass"}
        else:
            yield 1.0 if verdict == "check" and result["answer"] == "secret" else 0.0

    task = solve(grading=grading)
    task.verifier = verify(verdict=verdict)
    job = await task.run(ScriptedAgent("secret"))

    (run,) = job.runs
    assert (run.reward, run in job.errors) == (reward, not graded)
    if error is None:
        assert run.trace.error is None
    else:
        phase, phrase = error
        assert (run.trace.error or "").startswith(phase)
        assert phrase in (run.trace.error or "")


async def test_an_agent_that_fails_after_answering_is_still_judged() -> None:
    class RaisesAfter(Agent):
        async def __call__(self, run: Run) -> None:
            run.trace.content = "secret"
            raise RuntimeError("agent exploded")

    job = await verified(Environment("actor"), Environment("judge")).run(RaisesAfter())

    (run,) = job.runs
    assert (run.reward, run.trace.status, run.trace.error, job.errors) == (
        1.0,
        "error",
        "[agent loop] RuntimeError: agent exploded",
        [],
    )


async def test_a_verifier_that_cannot_be_placed_leaves_the_run_ungraded() -> None:
    actor = Environment("actor")

    @actor.template()
    async def solve():
        answer = yield "answer secret"
        yield {"score": 0.25, "answer": answer}

    @asynccontextmanager
    async def actor_only(task: Task) -> AsyncIterator[Runtime]:
        if task.env != "actor":
            raise RuntimeError("no judge here")
        async with LocalRuntime(actor)(task) as runtime:
            yield runtime

    task = solve()
    task.verifier = Task(env="judge", id="verify")
    job = await task.run(ScriptedAgent("secret"), runtime=actor_only)

    (run,) = job.runs
    assert (run.reward, run.trace.error, job.errors) == (
        0.0,
        "[provisioning verifier] RuntimeError: no judge here",
        [run],
    )


async def test_the_verifier_receives_the_actors_result_as_its_declared_type() -> None:
    class ActorResult(BaseModel):
        score: float
        answer: str

    received = []
    env = Environment("reviewed")

    @env.template()
    async def solve():
        answer = yield "answer secret"
        yield {"score": 0.25, "answer": answer}

    @env.template(returns=ActorResult)
    async def verify():
        result = yield ""
        received.append(result.content)
        yield 1.0

    task = solve()
    task.verifier = verify()
    await task.run(ScriptedAgent("secret"))

    assert received == [ActorResult(score=0.25, answer="secret")]


@pytest.mark.parametrize(
    ("held", "reward", "error", "events"),
    [
        pytest.param(
            "actor",
            0.0,
            "rollout timed out after 0.3s during actor cleanup",
            ["actor down"],
            id="in-the-actors-teardown-the-verifier-never-starts",
        ),
        pytest.param(
            "judge",
            1.0,
            "rollout timed out after 0.3s during cleanup",
            ["actor down", "judge up", "judge down"],
            id="in-the-verifiers-teardown-the-verdict-stands",
        ),
    ],
)
async def test_a_rollout_timeout_in_teardown_returns_and_lets_teardown_finish(
    held: str, reward: float, error: str, events: list[str]
) -> None:
    seen = []
    release = asyncio.Event()
    actor = Environment("actor")
    judge = Environment("judge")

    @actor.shutdown
    async def actor_down() -> None:
        if held == "actor":
            await release.wait()
        seen.append("actor down")

    @judge.initialize
    async def judge_up() -> None:
        seen.append("judge up")

    @judge.shutdown
    async def judge_down() -> None:
        if held == "judge":
            await release.wait()
        seen.append("judge down")

    job = await verified(actor, judge).run(ScriptedAgent("secret"), rollout_timeout=0.3)
    release.set()

    (run,) = job.runs
    assert (run.reward, run.trace.stop_reason, run.trace.error) == (reward, "timeout", error)
    await eventually(lambda: seen == events)
