"""A rollout, wherever it runs: what the Run and the Job say for each way it can end.

Each outcome runs on every runtime (``placements``), against a fixture
environment, and must end the same way on all of them and leave its placement
clean. An environment that cannot serve fails the run before the agent sees a
prompt, on every runtime that provisions one. Rollout failures that depend on
the shape of the provider rather than on the runtime follow.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import pytest
from dirty_equals import IsStr

from hud.eval import LocalRuntime, Runtime, Task, rollout
from tests.fixtures.envs import source
from tests.support import ScriptedAgent, eventually, lab, solve, steps

from .placements import NOT_APPLICABLE, PLACEMENTS, World, exported_runtimes, rows

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable

    from hud.agents.base import Agent
    from hud.eval import Provider
    from tests.support import FakeServices, HudEnv


@pytest.mark.parametrize("contract", NOT_APPLICABLE)
def test_every_exported_runtime_has_a_row_in_every_contract(contract: str) -> None:
    placed = {placement.kind for placement in PLACEMENTS.values()}

    assert placed | set(NOT_APPLICABLE[contract]) == exported_runtimes()


@dataclass(frozen=True)
class Outcome:
    fixture: str
    task: str
    agent: Callable[[], Agent]
    reward: float
    status: str
    error: Any = None
    stop_reason: str | None = None
    graded: bool = True
    started: bool = True
    args: dict[str, Any] = field(default_factory=dict)
    agent_config: dict[str, Any] | None = None


def echo(agent: Callable[[], Agent], word: str = "tangerine", **outcome: Any) -> Outcome:
    return Outcome("echo", "echo", agent, args={"word": word}, **outcome)


def faulty(task: str, error: Any, **outcome: Any) -> Outcome:
    return Outcome(
        "faulty", task, lambda: ScriptedAgent("x"), 0.0, "error", error, graded=False, **outcome
    )


OUTCOMES = {
    "the right answer earns the reward": echo(
        lambda: ScriptedAgent("tangerine"), reward=1.0, status="completed"
    ),
    "a typed answer is parsed before grading": Outcome(
        "echo", "count", lambda: ScriptedAgent("9"), 1.0, "completed", args={"word": "tangerine"}
    ),
    "an agent that raises before answering is graded on no answer": echo(
        lambda: ScriptedAgent(fail_before=RuntimeError("agent exploded")),
        reward=0.0,
        status="error",
        error="[agent loop] RuntimeError: agent exploded",
    ),
    "an agent that raises after answering keeps its grade": echo(
        lambda: ScriptedAgent("tangerine", fail_after=RuntimeError("agent exploded")),
        reward=1.0,
        status="error",
        error="[agent loop] RuntimeError: agent exploded",
    ),
    "an agent's own TimeoutError is an ordinary failure": echo(
        lambda: ScriptedAgent("tangerine", fail_after=TimeoutError("provider timed out")),
        reward=1.0,
        status="error",
        error="[agent loop] TimeoutError: provider timed out",
    ),
    "the agent deadline stops the agent and still grades": echo(
        lambda: ScriptedAgent("tangerine", linger=30),
        agent_config={"timeout_seconds": 0.3},
        reward=1.0,
        status="error",
        stop_reason="timeout",
        error="agent timed out after 0.3s",
    ),
    "a grader that raises leaves the run ungraded": faulty(
        "grade_raises", IsStr(regex=r"\[grading\] .*grader exploded.*")
    ),
    "a grade without a score leaves the run ungraded": faulty(
        "scoreless", IsStr(regex=r"\[grading\] .*missing a numeric 'score'.*")
    ),
    "a boolean score leaves the run ungraded": faulty(
        "bool_score", IsStr(regex=r"\[grading\] .*must include a numeric 'score'.*")
    ),
    "a start that raises fails the run before the agent sees a prompt": faulty(
        "start_raises", IsStr(regex=r"\[starting task\] .*start exploded.*"), started=False
    ),
    "args over the frame limit fail the start before the template runs": echo(
        lambda: ScriptedAgent("x"),
        word="x" * (16 * 1024 * 1024),
        reward=0.0,
        status="error",
        error=IsStr(
            regex=r"(?s)\[starting task\] .*'tasks\.start' request.*limit is 16777216 bytes.*"
        ),
        graded=False,
        started=False,
    ),
}


@pytest.mark.parametrize("placement", rows("outcomes"))
@pytest.mark.parametrize("outcome", OUTCOMES.values(), ids=OUTCOMES.keys())
async def test_a_rollout_ends_the_same_way_wherever_it_runs(
    outcome: Outcome,
    placement: str,
    services: FakeServices,
    hud_env: HudEnv,
    request: pytest.FixtureRequest,
) -> None:
    row = Task(
        env=outcome.fixture, id=outcome.task, args=outcome.args, agent_config=outcome.agent_config
    )
    world = World(source(outcome.fixture), services, hud_env, request)

    async with PLACEMENTS[placement].open(world) as runtime:
        job = await row.run(outcome.agent(), runtime=runtime)
        await eventually(lambda: PLACEMENTS[placement].clean(world))

    (run,) = job.runs
    assert {
        "reward": run.reward,
        "status": run.trace.status,
        "stop_reason": run.trace.stop_reason,
        "error": run.trace.error,
        "prompted": run.prompt is not None,
    } == {
        "reward": outcome.reward,
        "status": outcome.status,
        "stop_reason": outcome.stop_reason,
        "error": outcome.error,
        "prompted": outcome.started,
    }
    assert (job.errors == [run]) is not outcome.graded
    assert job.reward == (outcome.reward if outcome.graded else 0.0)
    assert run.job_id == job.id


UNSERVED = {
    "an env that exits at once": "broken",
    "an env that never finishes starting": "stalls",
}


@pytest.mark.parametrize("placement", rows("startup"))
@pytest.mark.parametrize("fixture", UNSERVED.values(), ids=UNSERVED.keys())
async def test_an_env_that_cannot_serve_fails_the_run_before_the_prompt_and_is_cleaned_up(
    fixture: str,
    placement: str,
    services: FakeServices,
    hud_env: HudEnv,
    request: pytest.FixtureRequest,
) -> None:
    world = World(source(fixture), services, hud_env, request, startup_limit=5)

    async with PLACEMENTS[placement].open(world) as runtime:
        job = await Task(env=fixture, id="never").run(ScriptedAgent("x"), runtime=runtime)
        await eventually(lambda: PLACEMENTS[placement].clean(world))

    (run,) = job.runs
    assert (run.trace.status, run.prompt, job.errors) == ("error", None, [run])
    assert run.trace.error == IsStr(regex=r"(?s)\[(provisioning|starting task)\] .+")


# ─── failures that depend on the provider's shape, not the runtime ─────

HEX_ID = IsStr(regex=r"[0-9a-f]{32}")
LOOPBACK = IsStr(regex=r"tcp://127\.0\.0\.1:\d+")


def broken_provider(error: BaseException) -> Provider:
    @asynccontextmanager
    async def provider(task: Task) -> AsyncIterator[Runtime]:
        raise error
        yield Runtime("tcp://127.0.0.1:1")

    return provider


def hanging_provider(task: Task) -> Any:
    @asynccontextmanager
    async def provider() -> AsyncIterator[Runtime]:
        await asyncio.Event().wait()
        yield Runtime("tcp://127.0.0.1:1")

    return provider()


def noted(message: str, note: str) -> BaseException:
    error = EOFError(message)
    error.add_note(note)
    return error


def failed(error: Any, *, stop_reason: str | None = None) -> dict[str, Any]:
    """A run that never went live: no prompt, no placement, no grade."""
    return {
        "status": "error",
        "stop_reason": stop_reason,
        "reward": 0.0,
        "raw": {},
        "error": error,
        "prompt": None,
        "runtime": None,
        "spans": ["system"],
    }


ADD = Task(env="lab", id="add", args={"a": 2, "b": 3})
SHAPES: dict[str, tuple[str, dict[str, Any], list[str], float | None]] = {
    "a constructor source serves the row": (
        "constructor",
        {
            "status": "completed",
            "stop_reason": None,
            "reward": 1.0,
            "raw": {"score": 1.0},
            "error": None,
            "prompt": "add 2 3",
            "runtime": LOOPBACK,
            "spans": ["task", "user", "task"],
        },
        ["initialize", "start add 2 3", "grade 5", "end add", "shutdown"],
        None,
    ),
    "the rollout deadline covers provisioning": (
        "hanging",
        failed("rollout timed out after 0.2s during provisioning", stop_reason="timeout"),
        [],
        0.2,
    ),
    "a provider that fails reports its notes": (
        "broken",
        failed("[provisioning] EOFError: sandbox closed\nenv output: ImportError: bugs"),
        [],
        None,
    ),
}


@pytest.mark.parametrize(
    ("provider", "expected", "hooks", "rollout_timeout"), SHAPES.values(), ids=SHAPES.keys()
)
async def test_a_rollout_always_returns_a_run_that_says_where_it_failed(
    provider: str, expected: dict[str, Any], hooks: list[str], rollout_timeout: float | None
) -> None:
    events: list[str] = []
    env = lab(events)
    providers: dict[str, Provider] = {
        "constructor": LocalRuntime(lambda task: env),
        "hanging": hanging_provider,
        "broken": broken_provider(noted("sandbox closed", "env output: ImportError: bugs")),
    }

    run = await rollout(
        ADD, ScriptedAgent(solve), runtime=providers[provider], rollout_timeout=rollout_timeout
    )
    await eventually(lambda: events[-1:] in ([], ["shutdown"]))

    assert {
        "status": run.trace.status,
        "stop_reason": run.trace.stop_reason,
        "reward": run.reward,
        "raw": run.grade.raw,
        "error": run.trace.error,
        "prompt": run.prompt_text if run.prompt is not None else None,
        "runtime": run.runtime,
        "spans": [step["source"] for step in steps(run.trace_id)],
    } == expected
    assert events == hooks
    assert (run.trace_id, run.job_id, run.slug) == (HEX_ID, HEX_ID, ADD.slug)
