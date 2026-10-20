"""One rollout: what the agent is handed, and how what the agent, the grader or a
deadline did decides the run's reward, status and error."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

import pytest

from hud import Environment
from hud.agents.base import Agent
from hud.agents.types import AgentConfig
from hud.capabilities import Connection
from hud.eval import LocalRuntime, Runtime, Task, rollout
from hud.telemetry.context import get_current_trace_id
from tests.conftest import eventually
from tests.fixtures.agents import ScriptedAgent

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path

    from hud.eval import Run


@pytest.mark.parametrize(
    ("prompt", "turns", "text"),
    [
        pytest.param("add 2 3", [("user", "add 2 3")], "add 2 3", id="text-is-one-user-turn"),
        pytest.param(None, [("user", "")], "", id="no-prompt-is-one-empty-user-turn"),
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
            roles = [(m.role, getattr(m.content, "text", None)) for m in run.prompt_messages]
            handed.append((roles, run.prompt_text))

    await ask().run(Reader())

    assert handed == [(turns, text)]


async def test_the_agent_is_handed_its_episodes_bindings_inside_its_trace() -> None:
    env = Environment("robots")

    @env.template()
    async def claim():
        yield {"prompt": "go", "bindings": {"robot": {"token": "slot-2"}}}
        yield 1.0

    handed = []

    class Reader(Agent):
        async def __call__(self, run: Run) -> None:
            handed.append((run.bindings, get_current_trace_id()))

    job = await claim().run(Reader())

    assert handed == [({"robot": {"token": "slot-2"}}, job.runs[0].trace_id)]


async def test_bindings_that_are_not_objects_fail_the_start() -> None:
    env = Environment("robots")

    @env.template()
    async def claim():
        yield {"prompt": "go", "bindings": {"robot": "slot-2"}}
        yield 1.0

    job = await claim().run(ScriptedAgent("x"))

    (run,) = job.runs
    assert (run.trace.status, run.prompt) == ("error", None)
    assert (run.trace.error or "").startswith(
        "[starting task] TypeError: task start frame 'bindings' must map capability name -> object"
    )


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

    run = await rollout(look(), Reader(), runtime=LocalRuntime(env), connections=[search])

    assert (held, run.connections, run.reward) == ([["search"]], {}, 1.0)


@pytest.mark.parametrize(
    ("grade", "reward", "content", "status", "error"),
    [
        pytest.param(0.5, 0.5, None, "completed", None, id="a-number"),
        pytest.param(
            {"score": 0.5, "content": "half right", "info": {"checked": 2}},
            0.5,
            "half right",
            "completed",
            None,
            id="a-frame-with-a-score",
        ),
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
            None,
            "error",
            "result must include a numeric 'score'",
            id="a-boolean-score",
        ),
        pytest.param(
            {"done": True},
            0.0,
            None,
            "error",
            "graded with a dict missing a numeric 'score' (keys: ['done'])",
            id="a-frame-without-a-score",
        ),
        pytest.param(
            "high",
            0.0,
            None,
            "error",
            "task graded with str",
            id="neither",
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

    (run,) = job.runs
    assert (run.reward, run.grade.content, run.trace.status) == (reward, content, status)
    assert (run in job.errors) is (error is not None)
    if error is None:
        assert run.trace.error is None
    else:
        assert error in (run.trace.error or "")
    assert (run.trace.error or "").startswith("[grading] ") is (status == "error")


async def test_a_grade_breakdown_reaches_the_evaluation_without_its_grader_details() -> None:
    env = Environment("grades")

    @env.template()
    async def grade_it():
        yield "go"
        yield {
            "score": 0.5,
            "subscores": [{"name": "tests", "value": 0.5, "info": {"stdout": "2 failed"}}],
        }

    job = await grade_it().run(ScriptedAgent("x"))

    assert job.runs[0].evaluation["subscores"] == [
        {"name": "tests", "weight": 1.0, "value": 0.5, "children": None}
    ]


async def test_an_agent_that_raises_before_answering_is_graded_on_no_answer() -> None:
    env = Environment("checked")

    @env.template()
    async def check():
        answer = yield "say yes"
        yield 1.0 if answer == "yes" else 0.0

    class Raises(Agent):
        async def __call__(self, run: Run) -> None:
            raise RuntimeError("agent exploded")

    job = await check().run(Raises())

    (run,) = job.runs
    assert (run.reward, run.trace.status, run.trace.error, job.errors) == (
        0.0,
        "error",
        "[agent loop] RuntimeError: agent exploded",
        [],
    )


async def test_an_agent_that_raises_after_answering_keeps_its_grade() -> None:
    env = Environment("checked")

    @env.template()
    async def check():
        answer = yield "say yes"
        yield 1.0 if answer == "yes" else 0.0

    class RaisesAfter(Agent):
        async def __call__(self, run: Run) -> None:
            run.trace.content = "yes"
            raise RuntimeError("agent exploded")

    job = await check().run(RaisesAfter())

    (run,) = job.runs
    assert (run.reward, run.trace.status, run.trace.error, job.reward) == (
        1.0,
        "error",
        "[agent loop] RuntimeError: agent exploded",
        1.0,
    )


@pytest.mark.parametrize(
    ("agent_timeout", "task_timeout"),
    [(0.3, None), (30, 0.3)],
    ids=["the-agents-own", "the-tasks-over-the-agents"],
)
async def test_the_agent_timeout_stops_the_agent_and_still_grades(
    agent_timeout: float, task_timeout: float | None
) -> None:
    env = Environment("checked")

    @env.template()
    async def check():
        answer = yield "say yes"
        yield 1.0 if answer == "yes" else 0.0

    class AnswersThenWaits(Agent):
        async def __call__(self, run: Run) -> None:
            run.trace.content = "yes"
            await asyncio.sleep(30)

    task = check()
    if task_timeout is not None:
        task.agent_config = {"timeout_seconds": task_timeout}
    job = await task.run(AnswersThenWaits(AgentConfig(timeout_seconds=agent_timeout)))

    (run,) = job.runs
    assert (run.reward, run.trace.status, run.trace.stop_reason, run.trace.error) == (
        1.0,
        "error",
        "timeout",
        "agent timed out after 0.3s",
    )


async def test_a_timeout_the_agent_raises_itself_is_an_ordinary_failure() -> None:
    env = Environment("checked")

    @env.template()
    async def check():
        answer = yield "say yes"
        yield 1.0 if answer == "yes" else 0.0

    class ProviderTimesOut(Agent):
        async def __call__(self, run: Run) -> None:
            run.trace.content = "yes"
            raise TimeoutError("provider timed out")

    job = await check().run(ProviderTimesOut(AgentConfig(timeout_seconds=30)))

    (run,) = job.runs
    assert (run.reward, run.trace.stop_reason, run.trace.error) == (
        1.0,
        None,
        "[agent loop] TimeoutError: provider timed out",
    )


async def test_a_task_whose_start_raises_fails_before_the_agent_is_prompted() -> None:
    env = Environment("broken")

    @env.template()
    async def start_raises():
        raise RuntimeError("start exploded")
        yield "never"

    job = await start_raises().run(ScriptedAgent("x"))

    (run,) = job.runs
    assert (run.trace.status, run.prompt, job.errors) == ("error", None, [run])
    assert (run.trace.error or "").startswith("[starting task] ")
    assert "start exploded" in (run.trace.error or "")


async def test_a_grader_that_raises_leaves_the_run_ungraded() -> None:
    env = Environment("broken")

    @env.template()
    async def grade_raises():
        yield "go"
        raise RuntimeError("grader exploded")
        yield 0.0

    job = await grade_raises().run(ScriptedAgent("x"))

    (run,) = job.runs
    assert (run.reward, run.trace.status, job.errors) == (0.0, "error", [run])
    assert (run.trace.error or "").startswith("[grading] ")
    assert "grader exploded" in (run.trace.error or "")


async def test_cancelling_a_rollout_cancels_its_task_and_stops_its_env() -> None:
    events = []
    env = Environment("cancelled")

    @env.shutdown
    async def stop() -> None:
        events.append("stopped")

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

    pending = asyncio.create_task(wait().run(Waits()))
    await eventually(lambda: events == ["prompted"])
    pending.cancel()

    with pytest.raises(asyncio.CancelledError):
        await pending
    await eventually(lambda: events == ["prompted", "task closed", "stopped"])


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
            if run.prompt_text == "take your time":
                await asyncio.sleep(30)

    job = await Task(env="slow", id=template).run(
        WaitsWhenAsked(), runtime=LocalRuntime(env), rollout_timeout=0.3
    )

    (run,) = job.runs
    assert (run.reward, run.trace.status, run.trace.stop_reason, run.trace.error) == (
        0.0,
        "error",
        "timeout",
        f"rollout timed out after 0.3s during {phase}",
    )
    assert job.errors == [run]
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

    async with LocalRuntime(env)(wait()) as served:
        pending = asyncio.create_task(wait().run(Waits(), runtime=Runtime(served.url)))
        await eventually(lambda: events == ["prompted"])
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending

        await eventually(lambda: events == ["prompted", "task closed"])


async def test_a_rollout_timed_out_on_an_env_served_elsewhere_closes_its_task_there() -> None:
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
            await asyncio.sleep(30)

    async with LocalRuntime(env)(wait()) as served:
        job = await wait().run(Waits(), runtime=Runtime(served.url), rollout_timeout=0.3)

        await eventually(lambda: events == ["task closed"])
    assert (job.runs[0].trace.stop_reason, job.runs[0].runtime) == ("timeout", served.url)


async def test_the_rollout_timeout_ends_a_run_still_being_provisioned() -> None:
    @asynccontextmanager
    async def never_ready(task: Task) -> AsyncIterator[Runtime]:
        await asyncio.sleep(30)
        yield Runtime("tcp://127.0.0.1:1")

    task = Task(env="checked", id="check")
    job = await task.run(ScriptedAgent("yes"), runtime=never_ready, rollout_timeout=0.3)

    (run,) = job.runs
    assert (run.trace.status, run.trace.stop_reason, run.trace.error, run.runtime) == (
        "error",
        "timeout",
        "rollout timed out after 0.3s during provisioning",
        None,
    )


async def test_a_rollout_timeout_in_teardown_keeps_the_grade_and_lets_teardown_finish() -> None:
    env = Environment("checked")

    @env.template()
    async def check():
        answer = yield "say yes"
        yield 1.0 if answer == "yes" else 0.0

    release = asyncio.Event()
    events = []

    @asynccontextmanager
    async def slow_to_stop(task: Task) -> AsyncIterator[Runtime]:
        async with LocalRuntime(env)(task) as runtime:
            yield runtime
        await release.wait()
        events.append("torn down")

    job = await check().run(ScriptedAgent("yes"), runtime=slow_to_stop, rollout_timeout=0.3)
    release.set()

    (run,) = job.runs
    assert (run.reward, run.trace.stop_reason, run.trace.error) == (
        1.0,
        "timeout",
        "rollout timed out after 0.3s during actor cleanup",
    )
    await eventually(lambda: events == ["torn down"])


async def test_a_provider_that_fails_returns_a_run_with_what_it_reported() -> None:
    @asynccontextmanager
    async def fails_with_notes(task: Task) -> AsyncIterator[Runtime]:
        error = EOFError("sandbox closed")
        error.add_note("env output: ImportError: bugs")
        raise error
        yield Runtime("tcp://127.0.0.1:1")

    job = await Task(env="checked", id="check").run(ScriptedAgent("yes"), runtime=fails_with_notes)

    (run,) = job.runs
    assert (run.reward, run.trace.status, run.prompt, run.runtime) == (0.0, "error", None, None)
    assert run.trace.error == (
        "[provisioning] EOFError: sandbox closed\nenv output: ImportError: bugs"
    )
