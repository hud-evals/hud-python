"""How a rollout ends a run: what the agent, the grader or a deadline did decides the
run's reward, status and error, and whether the job counts it as graded.

``rollout`` makes these decisions the same way on every runtime, so each case runs
in this process against a fixture environment; what a runtime adds (placing the
env, carrying its frames, tearing it down) is ``test_runtimes``.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any, NamedTuple

import pytest
from dirty_equals import IsStr

from hud.eval import LocalRuntime, Runtime, Task, rollout
from tests.fixtures.envs import source
from tests.support import ScriptedAgent, actor, eventually, judge, steps

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable

    from hud.agents.base import Agent
    from hud.eval import Provider

ECHO = Task(env="echo", id="echo", args={"word": "tangerine"})
EXPLODED = RuntimeError("agent exploded")


class Ending(NamedTuple):
    reward: float
    status: str | None
    error: Any = None
    stop_reason: str | None = None
    #: Whether the agent was given a prompt.
    prompted: bool = True
    #: Whether the job counts the run as graded rather than among its errors.
    graded: bool = True


ENDINGS: dict[str, tuple[Task, Callable[[], Agent], Ending]] = {
    "an agent that raises before answering is graded on no answer": (
        ECHO,
        lambda: ScriptedAgent(fail_before=EXPLODED),
        Ending(reward=0.0, status="error", error="[agent loop] RuntimeError: agent exploded"),
    ),
    "an agent that raises after answering keeps its grade": (
        ECHO,
        lambda: ScriptedAgent("tangerine", fail_after=EXPLODED),
        Ending(reward=1.0, status="error", error="[agent loop] RuntimeError: agent exploded"),
    ),
    "an agent's own TimeoutError is an ordinary failure": (
        ECHO,
        lambda: ScriptedAgent("tangerine", fail_after=TimeoutError("provider timed out")),
        Ending(reward=1.0, status="error", error="[agent loop] TimeoutError: provider timed out"),
    ),
    "the agent's deadline stops the agent and still grades": (
        ECHO.model_copy(update={"agent_config": {"timeout_seconds": 0.3}}),
        lambda: ScriptedAgent("tangerine", linger=30),
        Ending(
            reward=1.0, status="error", error="agent timed out after 0.3s", stop_reason="timeout"
        ),
    ),
    "a grader that raises leaves the run ungraded": (
        Task(env="faulty", id="grade_raises"),
        lambda: ScriptedAgent("x"),
        Ending(
            reward=0.0,
            status="error",
            error=IsStr(regex=r"\[grading\] .*grader exploded.*"),
            graded=False,
        ),
    ),
    "a boolean score leaves the run ungraded": (
        Task(env="faulty", id="bool_score"),
        lambda: ScriptedAgent("x"),
        Ending(
            reward=0.0,
            status="error",
            error=IsStr(regex=r"\[grading\] .*must include a numeric 'score'.*"),
            graded=False,
        ),
    ),
    "args over the frame limit fail the start before the template runs": (
        Task(env="echo", id="echo", args={"word": "x" * (16 * 1024 * 1024)}),
        lambda: ScriptedAgent("x"),
        Ending(
            reward=0.0,
            status="error",
            error=IsStr(
                regex=r"(?s)\[starting task\] .*'tasks\.start' request.*limit is 16777216 bytes.*"
            ),
            prompted=False,
            graded=False,
        ),
    ),
}


@pytest.mark.parametrize(("row", "agent", "ending"), ENDINGS.values(), ids=ENDINGS.keys())
async def test_what_happened_in_a_rollout_decides_how_its_run_ends(
    row: Task, agent: Callable[[], Agent], ending: Ending
) -> None:
    job = await row.run(agent(), runtime=LocalRuntime(source(row.env)))

    (run,) = job.runs
    assert (
        Ending(
            reward=run.reward,
            status=run.trace.status,
            error=run.trace.error,
            stop_reason=run.trace.stop_reason,
            prompted=run.prompt is not None,
            graded=run not in job.errors,
        )
        == ending
    )
    assert job.reward == (ending.reward if ending.graded else 0.0)


# ─── the rollout deadline ──────────────────────────────────────────────


@asynccontextmanager
async def never_places(task: Task) -> AsyncIterator[Runtime]:
    await asyncio.Event().wait()
    yield Runtime("tcp://127.0.0.1:1")


def fixture(task: Task) -> Any:
    return LocalRuntime(source(task.env))(task)


@pytest.mark.parametrize(
    ("phase", "row", "provider", "agent"),
    [
        ("provisioning", ECHO, never_places, lambda: ScriptedAgent("tangerine")),
        ("agent loop", ECHO, fixture, lambda: ScriptedAgent("tangerine", linger=30)),
        ("grading", Task(env="faulty", id="hang_grading"), fixture, lambda: ScriptedAgent("x")),
    ],
    ids=["provisioning", "agent-loop", "grading"],
)
async def test_the_rollout_deadline_ends_the_run_naming_the_phase_it_landed_in(
    phase: str, row: Task, provider: Provider, agent: Callable[[], Agent]
) -> None:
    job = await row.run(agent(), runtime=provider, rollout_timeout=0.3)

    (run,) = job.runs
    assert (run.reward, run.trace.status, run.trace.stop_reason, run.trace.error) == (
        0.0,
        "error",
        "timeout",
        f"rollout timed out after 0.3s during {phase}",
    )
    assert job.errors == [run]


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


SOLVE = Task(env="actor", id="solve")
REVIEWED = SOLVE.model_copy(update={"verifier": Task(env="judge", id="verify")})
TEARDOWNS: dict[str, tuple[Task, str, float, str, list[str]]] = {
    "a deadline in the actor's teardown keeps the actor's grade": (
        SOLVE,
        "actor",
        0.25,
        "rollout timed out after 0.2s during actor cleanup",
        ["start:actor", "stop:actor"],
    ),
    "a deadline in the actor's teardown never starts the verifier": (
        REVIEWED,
        "actor",
        0.0,
        "rollout timed out after 0.2s during actor cleanup",
        ["start:actor", "stop:actor"],
    ),
    "a deadline in the verifier's teardown keeps the verdict": (
        REVIEWED,
        "judge",
        1.0,
        "rollout timed out after 0.2s during cleanup",
        ["start:actor", "stop:actor", "start:judge", "stop:judge"],
    ),
}


@pytest.mark.parametrize(
    ("row", "held", "reward", "error", "placements"), TEARDOWNS.values(), ids=TEARDOWNS.keys()
)
async def test_a_deadline_during_teardown_returns_and_lets_teardown_finish(
    row: Task, held: str, reward: float, error: str, placements: list[str]
) -> None:
    release = asyncio.Event()
    log: list[str] = []
    provider = held_teardown(
        {"actor": LocalRuntime(actor()), "judge": LocalRuntime(judge())}, held, release, log
    )

    job = await row.run(ScriptedAgent("secret"), runtime=provider, rollout_timeout=0.2)
    assert log[-1] == f"start:{held}"
    release.set()
    await eventually(lambda: log[-1] == f"stop:{held}")

    (run,) = job.runs
    assert (run.reward, run.trace.stop_reason, run.trace.error) == (reward, "timeout", error)
    assert log == placements


# ─── a provider that fails ─────────────────────────────────────────────


@asynccontextmanager
async def fails_with_notes(task: Task) -> AsyncIterator[Runtime]:
    error = EOFError("sandbox closed")
    error.add_note("env output: ImportError: bugs")
    raise error
    yield Runtime("tcp://127.0.0.1:1")


async def test_a_provider_that_fails_still_returns_a_run_with_what_it_reported() -> None:
    run = await rollout(ECHO, ScriptedAgent("tangerine"), runtime=fails_with_notes)

    assert (run.reward, run.trace.status, run.grade.raw, run.prompt, run.runtime) == (
        0.0,
        "error",
        {},
        None,
        None,
    )
    assert (
        run.trace.error == "[provisioning] EOFError: sandbox closed\nenv output: ImportError: bugs"
    )
    assert [step["source"] for step in steps(run.trace_id)] == ["system"]
    assert (run.trace_id, run.job_id, run.slug) == (
        IsStr(regex=r"[0-9a-f]{32}"),
        IsStr(regex=r"[0-9a-f]{32}"),
        ECHO.slug,
    )
