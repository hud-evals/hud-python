"""One rollout: what the agent is handed, and how what the agent, the grader or a
deadline did decides the run's reward, status and error."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

import pytest

from hud import Environment, Task
from hud.agents.base import Agent
from hud.agents.types import AgentConfig
from hud.capabilities import Connection
from hud.eval.run import rollout
from hud.runtime import LocalRuntime, Runtime
from hud.types import text_of
from tests.conftest import eventually
from tests.fixtures.agents import ScriptedAgent
from tests.fixtures.environments import checked
from tests.serving import serving

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path

    from hud import Run


@pytest.mark.parametrize(
    ("prompt", "turns", "text"),
    [
        pytest.param(
            [
                {"role": "system", "content": "be brief"},
                {"role": "assistant", "content": {"type": "text", "text": "hi"}},
                {"role": "user", "content": "add 2 3"},
            ],
            [("user", "be brief"), ("assistant", "hi"), ("user", "add 2 3")],
            "be brief\n\nhi\n\nadd 2 3",
            id="chat-turns-keep-their-roles-and-system-becomes-user",
        ),
        pytest.param(
            [
                {"role": "user", "content": "first"},
                {"role": "user", "content": {"type": "image", "data": "aW1n", "mimeType": "x"}},
                {"role": "user", "content": ""},
                "second",
            ],
            [("user", "first"), ("user", None), ("user", ""), ("user", "second")],
            "first\n\nsecond",
            id="non-text-and-empty-turns-are-left-out-of-the-text",
        ),
        pytest.param(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "what is this?"},
                        {"type": "image", "data": "aW1n", "mimeType": "image/png"},
                    ],
                }
            ],
            [("user", "what is this?"), ("user", None)],
            "what is this?",
            id="a-turn-of-several-blocks-is-one-message-per-block",
        ),
    ],
)
async def test_the_agent_is_handed_the_prompt_as_turns_and_as_text(
    prompt: Any, turns: list[tuple[str, str | None]], text: str
) -> None:
    env = Environment("prompts")

    @env.template()
    async def ask():
        yield prompt
        yield 1.0

    handed = []

    class Reader(Agent):
        async def __call__(self, run: Run) -> None:
            roles = [(m.role, getattr(m.content, "text", None)) for m in run.prompt]
            handed.append((roles, text_of(run.prompt)))

    await ask().run(Reader())

    assert handed == [(turns, text)]


@pytest.mark.e2e
@pytest.mark.sandbox
async def test_the_agent_holds_its_connections_only_while_it_runs(tmp_path: Path) -> None:
    env = Environment("searching")
    env.workspace(tmp_path / "ws")

    @env.template()
    async def look():
        yield "go"
        yield 1.0

    search = Connection(
        name="search",
        capability="shell",
        url="https://search.example/v1",
        headers={"Authorization": "Bearer secret"},
    )
    held = []

    class Reader(Agent):
        async def __call__(self, run: Run) -> None:
            held.append(sorted(run.connections))

    trace = await rollout(look(), Reader(), runtime=LocalRuntime(env), connections=[search])

    assert (held, trace.reward) == ([["search"]], 1.0)


@pytest.mark.parametrize(
    ("grade", "reward", "content", "status", "error"),
    [
        pytest.param(
            {"score": 0.0, "content": "the env broke", "isError": True},
            0.0,
            "the env broke",
            "completed",
            "the env broke",
            id="a-frame-marked-as-an-error",
        ),
        pytest.param(
            {"score": True},
            0.0,
            "not graded",
            "error",
            "'score'",
            id="a-boolean-score",
        ),
        pytest.param(
            {"done": True},
            0.0,
            "not graded",
            "error",
            "'score'",
            id="a-frame-without-a-score",
        ),
    ],
)
async def test_what_the_task_grades_with_is_the_runs_reward_or_its_error(
    grade: Any, reward: float, content: str | None, status: str, error: str | None
) -> None:
    env = Environment("grades")

    @env.template()
    async def grade_it():
        yield "go"
        yield grade

    job = await grade_it().run(ScriptedAgent("x"))

    (trace,) = job.traces
    assert (trace.reward, trace.grade.content, trace.status) == (reward, content, status)
    assert (trace in job.errors) is (error is not None)
    if error is None:
        assert trace.error is None
    else:
        assert error in (trace.error or "")
    assert (trace.error or "").startswith("[grading] ") is (status == "error")


async def test_an_agent_that_raises_before_answering_is_graded_on_no_answer() -> None:
    class Raises(Agent):
        async def __call__(self, run: Run) -> None:
            raise RuntimeError("agent exploded")

    job = await Task(env="checked", id="check").run(Raises(), runtime=LocalRuntime(checked()))

    (trace,) = job.traces
    assert (trace.reward, trace.status, job.errors) == (0.0, "error", [])
    assert (trace.error or "").startswith("[agent loop] ")
    assert "agent exploded" in (trace.error or "")


async def test_an_agent_that_raises_after_answering_keeps_its_grade() -> None:
    class RaisesAfter(Agent):
        async def __call__(self, run: Run) -> None:
            run.trace.content = "yes"
            raise RuntimeError("agent exploded")

    job = await Task(env="checked", id="check").run(RaisesAfter(), runtime=LocalRuntime(checked()))

    (trace,) = job.traces
    assert (trace.reward, trace.status, job.reward) == (1.0, "error", 1.0)
    assert (trace.error or "").startswith("[agent loop] ")
    assert "agent exploded" in (trace.error or "")


@pytest.mark.parametrize(
    ("agent_timeout", "task_timeout"),
    [(0.3, None), (30, 0.3)],
    ids=["the-agents-own", "the-tasks-over-the-agents"],
)
async def test_the_agent_timeout_stops_the_agent_and_still_grades(
    agent_timeout: float, task_timeout: float | None
) -> None:
    class AnswersThenWaits(Agent):
        async def __call__(self, run: Run) -> None:
            run.trace.content = "yes"
            await asyncio.sleep(30)

    task = Task(env="checked", id="check")
    if task_timeout is not None:
        task.agent_config = {"timeout_seconds": task_timeout}
    agent = AnswersThenWaits(AgentConfig(timeout_seconds=agent_timeout))
    job = await task.run(agent, runtime=LocalRuntime(checked()))

    (trace,) = job.traces
    assert (trace.reward, trace.status, trace.stop_reason) == (1.0, "error", "timeout")
    assert "agent timed out" in (trace.error or "")


async def test_a_grader_that_raises_leaves_the_run_ungraded() -> None:
    env = Environment("broken")

    @env.template()
    async def grade_raises():
        yield "go"
        raise RuntimeError("grader exploded")
        yield 0.0

    job = await grade_raises().run(ScriptedAgent("x"))

    (trace,) = job.traces
    assert (trace.reward, trace.status, job.errors) == (0.0, "error", [trace])
    assert (trace.error or "").startswith("[grading] ")
    assert "grader exploded" in (trace.error or "")


@pytest.mark.parametrize(
    ("template", "phase"),
    [
        ("slow_start", "starting task"),
        ("slow_agent", "agent loop"),
        ("slow_grade", "grading"),
    ],
)
async def test_the_rollout_timeout_ends_the_run_naming_its_phase_and_stops_the_env(
    template: str, phase: str
) -> None:
    events = []
    env = Environment("slow")

    @env.shutdown
    async def stop() -> None:
        events.append("stopped")

    @env.template()
    async def slow_start():
        await asyncio.sleep(30)
        yield "never"
        yield 1.0

    @env.template()
    async def slow_agent():
        yield "take your time"
        yield 1.0

    @env.template()
    async def slow_grade():
        yield "go"
        await asyncio.sleep(30)
        yield 1.0

    class WaitsWhenAsked(Agent):
        async def __call__(self, run: Run) -> None:
            run.trace.content = "done"
            if text_of(run.prompt) == "take your time":
                await asyncio.sleep(30)

    job = await Task(env="slow", id=template).run(
        WaitsWhenAsked(), runtime=LocalRuntime(env), rollout_timeout=0.3
    )

    (trace,) = job.traces
    assert (trace.reward, trace.status, trace.stop_reason, trace.error) == (
        0.0,
        "error",
        "timeout",
        f"rollout timed out after 0.3s during {phase}",
    )
    assert job.errors == [trace]
    await eventually(lambda: events == ["stopped"])


async def test_a_rollout_cancelled_on_an_env_served_elsewhere_closes_its_task_there() -> None:
    events = []
    env = Environment("served")

    @env.template()
    async def wait():
        try:
            yield "go"
            yield 1.0
        finally:
            events.append("task closed")

    class Waits(Agent):
        async def __call__(self, run: Run) -> None:
            events.append("prompted")
            await asyncio.sleep(30)

    async with serving(env) as served:
        pending = asyncio.create_task(wait().run(Waits(), runtime=Runtime(served.url)))
        await eventually(lambda: events == ["prompted"])
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending

        await eventually(lambda: events == ["prompted", "task closed"])


@pytest.mark.parametrize(("template", "placed"), [("wait", True), ("slow_start", False)])
async def test_a_rollout_timed_out_on_an_env_served_elsewhere_closes_its_task_there(
    template: str, placed: bool
) -> None:
    events = []
    env = Environment("served")

    @env.template()
    async def wait():
        try:
            yield "go"
            yield 1.0
        finally:
            events.append("task closed")

    @env.template()
    async def slow_start():
        try:
            await asyncio.sleep(30)
            yield "never"
            yield 1.0
        finally:
            events.append("task closed")

    class Waits(Agent):
        async def __call__(self, run: Run) -> None:
            await asyncio.sleep(30)

    task = Task(env="served", id=template)
    async with serving(env) as served:
        job = await task.run(Waits(), runtime=Runtime(served.url), rollout_timeout=0.3)

        await eventually(lambda: events == ["task closed"])
    (trace,) = job.traces
    assert (trace.stop_reason, trace.runtime) == ("timeout", served.url if placed else None)


async def test_the_rollout_timeout_ends_a_run_still_being_provisioned() -> None:
    @asynccontextmanager
    async def never_ready(task: Task) -> AsyncIterator[Runtime]:
        await asyncio.sleep(30)
        yield Runtime("tcp://127.0.0.1:1")

    task = Task(env="checked", id="check")
    job = await task.run(ScriptedAgent("yes"), runtime=never_ready, rollout_timeout=0.3)

    (trace,) = job.traces
    assert (trace.status, trace.stop_reason, trace.runtime) == ("error", "timeout", None)
    assert "during provisioning" in (trace.error or "")


async def test_a_rollout_timeout_in_teardown_keeps_the_grade_and_lets_teardown_finish() -> None:
    release = asyncio.Event()
    events = []

    @asynccontextmanager
    async def slow_to_stop(task: Task) -> AsyncIterator[Runtime]:
        async with LocalRuntime(checked())(task) as runtime:
            yield runtime
        await release.wait()
        events.append("torn down")

    job = await Task(env="checked", id="check").run(
        ScriptedAgent("yes"), runtime=slow_to_stop, rollout_timeout=0.3
    )
    release.set()

    (trace,) = job.traces
    assert (trace.reward, trace.stop_reason) == (1.0, "timeout")
    assert "during actor cleanup" in (trace.error or "")
    await eventually(lambda: events == ["torn down"])
