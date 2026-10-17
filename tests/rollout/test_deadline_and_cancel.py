"""A rollout stopped early, wherever it runs: a deadline ends it as a timed-out run, and a
cancel propagates to the caller; either way its placement is left clean.

The deadline and the cancel run on every runtime (``placements``). A deadline
that lands while a placement is torn down returns at once and lets the teardown
finish behind it.
"""

from __future__ import annotations

import asyncio
import contextlib
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

import pytest
from dirty_equals import IsStr

from hud.agents.openai_compatible import OpenAIChatAgent
from hud.agents.types import OpenAIChatConfig
from hud.eval import HostedRuntime, LocalRuntime, Task
from tests.fixtures.envs import source
from tests.support import Reply, ScriptedAgent, actor, eventually, judge

from .placements import PLACEMENTS, ROLLOUTS, World, rows

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from hud.agents.base import Agent
    from hud.eval import Provider, Runtime
    from tests.support import FakeServices, HudEnv

ECHO = Task(env="echo", id="echo", args={"word": "tangerine"})
# The row, the phase the deadline lands in, and how long the agent keeps running after it answers.
MOMENTS: dict[str, tuple[Task, str | None, float]] = {
    "during the agent loop": (ECHO, "agent loop", 3600),
    "during grading": (Task(env="faulty", id="hang_grading"), "grading", 0),
    "before the platform accepts it": (ECHO, None, 0),
}


def agent_for(placement: str, linger: float = 3600) -> Agent:
    """An agent that answers, then keeps running for ``linger`` seconds; the platform runs
    only the agent types it knows."""
    if PLACEMENTS[placement].kind is HostedRuntime:
        return OpenAIChatAgent(
            OpenAIChatConfig(model="test-model", api_key="secret", base_url="http://model.test")
        )
    return ScriptedAgent("tangerine", linger=linger)


def underway(world: World, agent: Agent) -> bool:
    """The agent here has its prompt, or the platform has accepted the rollout."""
    prompted = isinstance(agent, ScriptedAgent) and agent.prompts != []
    return prompted or world.services.requests("api", "POST", ROLLOUTS) != []


DEADLINES = [
    *(
        pytest.param(
            placement.values[0],
            moment,
            "rollout_timeout",
            id=f"{moment}-{placement.id}",
            marks=placement.marks,
        )
        for placement in rows("deadline")
        for moment in ("during the agent loop", "during grading")
        if not (placement.id == "hosted" and moment == "during grading")
    ),
    pytest.param("hosted", "before the platform accepts it", "rollout_timeout", id="unsubmitted"),
    pytest.param(
        "hud-tunnel", "during the agent loop", "constructor", id="deprecated-run_timeout-hud"
    ),
    pytest.param(
        "hosted", "during the agent loop", "constructor", id="deprecated-run_timeout-hosted"
    ),
]


@pytest.mark.parametrize(("placement", "moment", "api"), DEADLINES)
async def test_a_deadline_ends_the_run_as_timed_out_and_leaves_its_placement_clean(
    placement: str,
    moment: str,
    api: str,
    services: FakeServices,
    hud_env: HudEnv,
    request: pytest.FixtureRequest,
) -> None:
    row, phase, linger = MOMENTS[moment]
    within = PLACEMENTS[placement].patience
    deprecated = api == "constructor"
    world = World(
        source(row.env), services, hud_env, request, run_timeout=within if deprecated else None
    )
    warned = (
        pytest.warns(DeprecationWarning, match="rollout_timeout=... to Task.run")
        if deprecated
        else contextlib.nullcontext()
    )

    with warned:
        async with PLACEMENTS[placement].open(world) as runtime:
            if phase is None:
                services.route("api", "POST", ROLLOUTS, Reply(json={}, delay=5))
            job = await row.run(
                agent_for(placement, linger),
                runtime=runtime,
                rollout_timeout=None if deprecated else within,
            )
            await eventually(lambda: PLACEMENTS[placement].clean(world), within=max(10, within))

    (run,) = job.runs
    hosted = PLACEMENTS[placement].kind is HostedRuntime
    error = (
        IsStr(regex=rf"hosted rollout [0-9a-f]{{32}} did not finish within {within:g}s")
        if hosted
        else f"rollout timed out after {within:g}s during {phase}"
    )
    assert (run.trace.status, run.trace.stop_reason, run.reward, run.trace.error) == (
        "error",
        "timeout",
        0.0,
        error,
    )
    assert job.errors == [run]


@pytest.mark.parametrize("placement", rows("cancel"))
async def test_cancelling_a_rollout_propagates_and_leaves_its_placement_clean(
    placement: str,
    services: FakeServices,
    hud_env: HudEnv,
    request: pytest.FixtureRequest,
) -> None:
    world = World(source("echo"), services, hud_env, request)
    agent = agent_for(placement)

    async with PLACEMENTS[placement].open(world) as runtime:
        pending = asyncio.create_task(ECHO.run(agent, runtime=runtime))
        await eventually(lambda: underway(world, agent), within=PLACEMENTS[placement].patience)
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending

        await eventually(
            lambda: PLACEMENTS[placement].clean(world),
            within=max(10, PLACEMENTS[placement].patience),
        )


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
