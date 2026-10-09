"""A rollout, wherever it runs: what the Run and the Job say for each way it can end.

Each outcome runs under every placement, against a fixture environment served
from source: in this process, in a child process, in a container started through
the fake docker, and attached by address. Containers and attached addresses are
served from a child process, as they would be in production. The same row must end the same way in
all of them.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import pytest
from dirty_equals import IsStr

from hud.eval import DockerRuntime, LocalRuntime, Runtime, SubprocessRuntime, Task
from tests.fixtures.envs import source
from tests.harness import ScriptedAgent

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable
    from pathlib import Path

    from hud.agents.base import Agent
    from hud.eval import Provider
    from tests.harness import FakeDocker


@asynccontextmanager
async def in_process(
    path: Path, fake_docker: FakeDocker, tmp_path: Path
) -> AsyncIterator[Provider]:
    yield LocalRuntime(path)


@asynccontextmanager
async def child_process(
    path: Path, fake_docker: FakeDocker, tmp_path: Path
) -> AsyncIterator[Provider]:
    yield SubprocessRuntime(path)


@asynccontextmanager
async def served_apart(path: Path) -> AsyncIterator[str]:
    """Serve ``path`` in a child process and yield its address, so no test shares its env."""
    async with SubprocessRuntime(path)(Task(env=path.stem, id="serve")) as served:
        yield served.url.removeprefix("tcp://")


@asynccontextmanager
async def container(path: Path, fake_docker: FakeDocker, tmp_path: Path) -> AsyncIterator[Provider]:
    async with served_apart(path) as address:
        fake_docker.on(r"^run .* fixture:1$", stdout="fixture-1\n")
        fake_docker.on(r"^port fixture-1 8765$", stdout=f"{address}\n")
        yield DockerRuntime("fixture:1")


@asynccontextmanager
async def attached(path: Path, fake_docker: FakeDocker, tmp_path: Path) -> AsyncIterator[Provider]:
    async with served_apart(path) as address:
        yield Runtime(f"tcp://{address}")


PLACEMENTS = {
    "in-process": in_process,
    "child-process": child_process,
    "container": container,
    "attached": attached,
}


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
    rollout_timeout: float | None = None


def echo(agent: Callable[[], Agent], **outcome: Any) -> Outcome:
    return Outcome("echo", "echo", agent, args={"word": "tangerine"}, **outcome)


OUTCOMES = {
    "the right answer earns the reward": echo(
        lambda: ScriptedAgent("tangerine"), reward=1.0, status="completed"
    ),
    "a wrong answer is graded zero": echo(
        lambda: ScriptedAgent("lime"), reward=0.0, status="completed"
    ),
    "a typed answer is parsed before grading": Outcome(
        "echo",
        "count",
        lambda: ScriptedAgent("9"),
        args={"word": "tangerine"},
        reward=1.0,
        status="completed",
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
    "the agent deadline stops the agent and still grades": echo(
        lambda: ScriptedAgent("tangerine", linger=30),
        agent_config={"timeout_seconds": 0.3},
        reward=1.0,
        status="error",
        stop_reason="timeout",
        error="agent timed out after 0.3s",
    ),
    "the rollout deadline during the agent leaves the run ungraded": echo(
        lambda: ScriptedAgent("tangerine", linger=30),
        rollout_timeout=1.0,
        reward=0.0,
        status="error",
        stop_reason="timeout",
        error="rollout timed out after 1s during agent loop",
        graded=False,
    ),
    "a grader that raises leaves the run ungraded": Outcome(
        "faulty",
        "grade_raises",
        lambda: ScriptedAgent("x"),
        reward=0.0,
        status="error",
        error=IsStr(regex=r"\[grading\] .*grader exploded.*"),
        graded=False,
    ),
    "a grade without a score leaves the run ungraded": Outcome(
        "faulty",
        "scoreless",
        lambda: ScriptedAgent("x"),
        reward=0.0,
        status="error",
        error=IsStr(regex=r"\[grading\] .*missing a numeric 'score'.*"),
        graded=False,
    ),
    "the rollout deadline during grading leaves the run ungraded": Outcome(
        "faulty",
        "hang_grading",
        lambda: ScriptedAgent("x"),
        rollout_timeout=1.0,
        reward=0.0,
        status="error",
        stop_reason="timeout",
        error="rollout timed out after 1s during grading",
        graded=False,
    ),
    "a start that raises fails the run before the agent sees a prompt": Outcome(
        "faulty",
        "start_raises",
        lambda: ScriptedAgent("x"),
        reward=0.0,
        status="error",
        error=IsStr(regex=r"\[starting task\] .*start exploded.*"),
        graded=False,
        started=False,
    ),
}


@pytest.mark.parametrize("placement", PLACEMENTS)
@pytest.mark.parametrize("outcome", OUTCOMES.values(), ids=OUTCOMES.keys())
async def test_a_rollout_ends_the_same_way_wherever_it_runs(
    outcome: Outcome, placement: str, fake_docker: FakeDocker, tmp_path: Path
) -> None:
    row = Task(
        env=outcome.fixture,
        id=outcome.task,
        args=outcome.args,
        agent_config=outcome.agent_config,
    )

    async with PLACEMENTS[placement](source(outcome.fixture), fake_docker, tmp_path) as runtime:
        job = await row.run(
            outcome.agent(), runtime=runtime, rollout_timeout=outcome.rollout_timeout
        )

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
    assert (run.slug, run.job_id) == (row.slug, job.id)
