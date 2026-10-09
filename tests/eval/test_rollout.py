"""The rollout atom: one task driven to a graded ``Run``, whatever fails along the way."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import pytest
from dirty_equals import IsStr

from hud.agents.base import Agent
from hud.capabilities import Connection
from hud.eval import LocalRuntime, Run, Runtime, SubprocessRuntime, Task, rollout
from hud.telemetry.context import get_current_trace_id
from tests.eval.envs import SUMS_SOURCE, eventually, lab, solve
from tests.harness import ScriptedAgent, steps

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path

    from hud.environment import Environment
    from hud.eval import Provider

HEX_ID = IsStr(regex=r"[0-9a-f]{32}")
LOOPBACK = IsStr(regex=r"tcp://127\.0\.0\.1:\d+")
GRADED_SPANS = ["task", "user", "task"]


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


@dataclass
class Case:
    task: Task
    agent: ScriptedAgent
    expected: dict[str, Any]
    events: list[str] = field(default_factory=list)
    provider: str = "live"
    rollout_timeout: float | None = None


def add(a: int = 2, b: int = 3, **row: Any) -> Task:
    return Task(env="lab", id="add", args={"a": a, "b": b}, **row)


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


def errored(error: Any, *, stop_reason: str | None = None) -> dict[str, Any]:
    """A run whose grading failed: the prompt and placement survive, the grade does not."""
    return {
        "status": "error",
        "stop_reason": stop_reason,
        "reward": 0.0,
        "raw": {},
        "error": error,
        "prompt": "go",
        "runtime": LOOPBACK,
        "spans": ["task", "user", "system"],
    }


COMPLETED = {
    "status": "completed",
    "stop_reason": None,
    "reward": 1.0,
    "raw": {"score": 1.0},
    "error": None,
    "prompt": "add 2 3",
    "runtime": LOOPBACK,
    "spans": GRADED_SPANS,
}
HOOKS = ["initialize", "start add 2 3", "grade 5", "end add", "shutdown"]

CASES = {
    "an answer is graded": Case(add(), ScriptedAgent(solve), COMPLETED, HOOKS),
    "a constructor source serves the row": Case(
        add(), ScriptedAgent(solve), COMPLETED, HOOKS, provider="constructor"
    ),
    # The child process logs its hooks in its own memory.
    "a subprocess source serves the row": Case(
        add(), ScriptedAgent(solve), COMPLETED, provider="subprocess"
    ),
    "an agent raising before it answers is graded on no answer": Case(
        add(),
        ScriptedAgent(fail_before=RuntimeError("agent exploded")),
        COMPLETED
        | {
            "status": "error",
            "reward": 0.0,
            "raw": {"score": 0.0},
            "error": "[agent loop] RuntimeError: agent exploded",
            "spans": [*GRADED_SPANS, "system"],
        },
        ["initialize", "start add 2 3", "grade None", "end add", "shutdown"],
    ),
    "an agent raising after it answers keeps its grade": Case(
        add(),
        ScriptedAgent(solve, fail_after=RuntimeError("agent exploded")),
        COMPLETED
        | {
            "status": "error",
            "error": "[agent loop] RuntimeError: agent exploded",
            "spans": [*GRADED_SPANS, "system"],
        },
        HOOKS,
    ),
    "an agent's own TimeoutError is an ordinary failure": Case(
        add(),
        ScriptedAgent(solve, fail_after=TimeoutError("provider timed out")),
        COMPLETED
        | {
            "status": "error",
            "error": "[agent loop] TimeoutError: provider timed out",
            "spans": [*GRADED_SPANS, "system"],
        },
        HOOKS,
    ),
    "an agent's own TimeoutError under an unexpired deadline is an ordinary failure": Case(
        add(agent_config={"timeout_seconds": 10.0}),
        ScriptedAgent(solve, fail_after=TimeoutError("provider timed out")),
        COMPLETED
        | {
            "status": "error",
            "error": "[agent loop] TimeoutError: provider timed out",
            "spans": [*GRADED_SPANS, "system"],
        },
        HOOKS,
    ),
    "the agent deadline stops the agent and still grades": Case(
        add(agent_config={"timeout_seconds": 0.1}),
        ScriptedAgent(solve, linger=30),
        COMPLETED
        | {
            "status": "error",
            "stop_reason": "timeout",
            "error": "agent timed out after 0.1s",
            "spans": ["task", "user", "system", "task"],
        },
        HOOKS,
    ),
    "the rollout deadline during the agent loop leaves the run ungraded": Case(
        add(),
        ScriptedAgent(solve, linger=30),
        COMPLETED
        | {
            "status": "error",
            "stop_reason": "timeout",
            "raw": {},
            "reward": 0.0,
            "error": "rollout timed out after 0.2s during agent loop",
            "spans": ["task", "user", "system"],
        },
        ["initialize", "start add 2 3", "end add", "shutdown"],
        rollout_timeout=0.2,
    ),
    "the rollout deadline covers grading": Case(
        Task(env="lab", id="hang_grading"),
        ScriptedAgent("x"),
        errored("rollout timed out after 0.2s during grading", stop_reason="timeout"),
        ["initialize", "grading cancelled", "shutdown"],
        rollout_timeout=0.2,
    ),
    "the rollout deadline covers provisioning": Case(
        add(),
        ScriptedAgent(solve),
        failed("rollout timed out after 0.2s during provisioning", stop_reason="timeout"),
        provider="hanging",
        rollout_timeout=0.2,
    ),
    "a grader that raises fails grading": Case(
        Task(env="lab", id="grade_raises"),
        ScriptedAgent("x"),
        errored(IsStr(regex=r"\[grading\] .*HudProtocolError: .*grader exploded")),
        ["initialize", "shutdown"],
    ),
    "a grade frame without a score fails grading": Case(
        Task(env="lab", id="frame", args={"result": {"done": True}}),
        ScriptedAgent("x"),
        errored(IsStr(regex=r"\[grading\] .*missing a numeric 'score' \(keys: \['done'\]\)")),
        ["initialize", "shutdown"],
    ),
    "a boolean score fails grading": Case(
        Task(env="lab", id="frame", args={"result": {"score": True}}),
        ScriptedAgent("x"),
        errored(IsStr(regex=r"\[grading\] .*result must include a numeric 'score'")),
        ["initialize", "shutdown"],
    ),
    "a subscore without a value fails grading": Case(
        Task(env="lab", id="frame", args={"result": {"score": 1.0, "subscores": [{"name": "m"}]}}),
        ScriptedAgent("x"),
        errored(IsStr(regex=r"(?s)\[grading\] .*ValidationError: .*SubScore\nvalue\n.*")),
        ["initialize", "shutdown"],
    ),
    "malformed bindings fail the start": Case(
        Task(env="lab", id="claim", args={"published": {"robot": "slot-2"}}),
        ScriptedAgent("x"),
        failed(
            "[starting task] TypeError: task start frame 'bindings' must map capability "
            "name -> object, got {'robot': 'slot-2'}"
        ),
        ["initialize", "shutdown"],
    ),
    "args over the frame limit fail the start before the template runs": Case(
        Task(env="lab", id="large", args={"criteria": "x" * (16 * 1024 * 1024)}),
        ScriptedAgent("x"),
        failed(
            IsStr(
                regex=r"(?s)\[starting task\] .*'tasks\.start' request.*"
                r"limit is 16777216 bytes.*file ID.*"
            )
        ),
        ["initialize", "shutdown"],
    ),
    "args over the frame limit once JSON-escaped fail the start": Case(
        Task(env="lab", id="large", args={"criteria": "é" * (16 * 1024 * 1024 // 6 + 1)}),
        ScriptedAgent("x"),
        failed(IsStr(regex=r"(?s)\[starting task\] .*limit is 16777216 bytes.*")),
        ["initialize", "shutdown"],
    ),
    "a provider that fails reports its notes": Case(
        add(),
        ScriptedAgent(solve),
        failed("[provisioning] EOFError: sandbox closed\nenv output: ImportError: bugs"),
        provider="broken",
    ),
}


@pytest.mark.parametrize("case", CASES.values(), ids=CASES.keys())
async def test_a_rollout_always_returns_a_run_that_says_where_it_failed(
    case: Case, tmp_path: Path
) -> None:
    events: list[str] = []
    env = lab(events)
    source = tmp_path / "env.py"
    source.write_text(SUMS_SOURCE)
    providers: dict[str, Provider] = {
        "live": LocalRuntime(env),
        "constructor": LocalRuntime(lambda task: env),
        "subprocess": SubprocessRuntime(source),
        "hanging": hanging_provider,
        "broken": broken_provider(noted("sandbox closed", "env output: ImportError: bugs")),
    }

    run = await rollout(
        case.task,
        case.agent,
        runtime=providers[case.provider],
        rollout_timeout=case.rollout_timeout,
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
    } == case.expected
    assert events == case.events
    assert (run.trace_id, run.job_id, run.slug) == (HEX_ID, HEX_ID, case.task.slug)


def chat_row(messages: list[Any]) -> Task:
    return Task(env="lab", id="chat", args={"messages": messages})


PROMPTS = {
    "text is one user turn": (add(), [("user", "add 2 3")], "add 2 3"),
    "no prompt is one empty user turn": (Task(env="lab", id="silent"), [("user", "")], ""),
    "chat turns keep their roles and system becomes user": (
        chat_row(
            [
                {"role": "system", "content": "be brief"},
                {"role": "assistant", "content": {"type": "text", "text": "hi"}},
                {"role": "user", "content": "add 2 3"},
            ]
        ),
        [("user", "be brief"), ("assistant", "hi"), ("user", "add 2 3")],
        "be brief\n\nhi\n\nadd 2 3",
    ),
    "non-text turns are dropped from the text": (
        chat_row(
            [
                {"role": "user", "content": "first"},
                {"role": "user", "content": {"type": "image", "data": "aW1n", "mimeType": "x"}},
                "second",
            ]
        ),
        [("user", "first"), ("user", None), ("user", "second")],
        "first\n\nsecond",
    ),
    "a turn of several blocks is one message per block": (
        chat_row(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "what is this?"},
                        {"type": "image", "data": "aW1n", "mimeType": "image/png"},
                    ],
                }
            ]
        ),
        [("user", "what is this?"), ("user", None)],
        "what is this?",
    ),
}


@pytest.mark.parametrize(("row", "messages", "text"), PROMPTS.values(), ids=PROMPTS.keys())
async def test_the_agent_sees_the_prompt_as_turns_and_as_text(
    row: Task, messages: list[tuple[str, str | None]], text: str
) -> None:
    seen: list[tuple[list[tuple[str, str | None]], str]] = []

    class Reader(Agent):
        async def __call__(self, run: Run) -> None:
            turns: list[tuple[str, str | None]] = [
                (message.role, getattr(message.content, "text", None))
                for message in run.prompt_messages
            ]
            seen.append((turns, run.prompt_text))

    run = await rollout(row, Reader(), runtime=LocalRuntime(lab()))

    assert seen == [(messages, text)]
    assert run.trace.status == "completed"


async def test_episode_bindings_from_the_start_frame_reach_the_agent() -> None:
    seen: list[tuple[dict[str, Any], str | None]] = []

    class Inspector(Agent):
        async def __call__(self, run: Run) -> None:
            seen.append((run.bindings, get_current_trace_id()))

    run = await rollout(
        Task(env="lab", id="claim", args={"published": {"robot": {"token": "slot-2"}}}),
        Inspector(),
        runtime=LocalRuntime(lab()),
    )

    assert seen == [({"robot": {"token": "slot-2"}}, run.trace_id)]
    assert run.bindings == {"robot": {"token": "slot-2"}}


@pytest.mark.e2e
@pytest.mark.sandbox
async def test_connections_reach_the_agent_only_while_it_runs(tmp_path: Path) -> None:
    env = lab()
    env.workspace(tmp_path / "ws")
    search = Connection(
        name="search",
        capability="shell",
        url="https://search.example/v1",
        headers={"Authorization": "Bearer secret"},
    )
    seen: list[list[str]] = []

    class Inspector(Agent):
        async def __call__(self, run: Run) -> None:
            seen.append(sorted(run.connections))
            run.trace.content = "5"

    run = await rollout(add(), Inspector(), runtime=LocalRuntime(env), connections=[search])

    assert seen == [["search"]]
    assert (run.reward, run.connections) == (1.0, {})


TRACE = "0" * 31 + "1"
IDENTITIES = {
    "a bare rollout mints its job and trace": ((None, None, None), (HEX_ID, None, HEX_ID)),
    "threaded ids are kept": (("j1", "g1", TRACE), ("j1", "g1", TRACE)),
}


@pytest.mark.parametrize(("ids", "expected"), IDENTITIES.values(), ids=IDENTITIES.keys())
async def test_a_run_carries_its_job_group_and_trace(
    ids: tuple[str | None, str | None, str | None], expected: tuple[Any, ...]
) -> None:
    job_id, group_id, trace_id = ids

    run = await rollout(
        add(),
        ScriptedAgent(solve),
        runtime=LocalRuntime(lab()),
        job_id=job_id,
        group_id=group_id,
        trace_id=trace_id,
    )

    assert (run.job_id, run.group_id, run.trace_id) == expected
    assert run.reward == 1.0


def blocking_teardown(env: Environment, release: asyncio.Event, stopped: list[str]) -> Provider:
    """Places rows on ``env`` and holds teardown until ``release`` is set."""

    @asynccontextmanager
    async def provider(task: Task) -> AsyncIterator[Runtime]:
        try:
            async with LocalRuntime(env)(task) as runtime:
                yield runtime
        finally:
            await release.wait()
            stopped.append(task.env)

    return provider


async def test_a_rollout_deadline_returns_without_waiting_for_teardown() -> None:
    release = asyncio.Event()
    stopped: list[str] = []

    run = await rollout(
        add(),
        ScriptedAgent(solve),
        runtime=blocking_teardown(lab(), release, stopped),
        rollout_timeout=0.2,
    )
    assert stopped == []
    release.set()
    await eventually(lambda: stopped == ["lab"])

    assert (run.reward, run.trace.status, run.trace.stop_reason) == (1.0, "error", "timeout")
    assert run.trace.error == "rollout timed out after 0.2s during actor cleanup"


async def test_cancelling_a_rollout_cancels_the_task_and_propagates() -> None:
    events: list[str] = []
    agent = ScriptedAgent(solve, delay=30)
    pending = asyncio.create_task(rollout(add(), agent, runtime=LocalRuntime(lab(events))))
    await eventually(lambda: agent.prompts != [])

    pending.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pending
    await eventually(lambda: events[-1:] == ["shutdown"])

    assert events == ["initialize", "start add 2 3", "end add", "shutdown"]
