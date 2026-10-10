"""A rollout, wherever it runs: what the Run and the Job say for each way it can end.

Each outcome runs under every placement, against a fixture environment served
from source: in this process, in a child process, in a container started through
the fake docker, attached by address, over the HUD runtime's tunnel, and in
Modal and Daytona sandboxes through their fake SDKs (all but the first two served
from a child process, as in production). The same row must
end the same way in all of them. Rollout contracts that do not depend on
placement follow the matrix.
"""

from __future__ import annotations

import asyncio
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import pytest
from dirty_equals import IsStr

from hud.agents.base import Agent
from hud.capabilities import Connection
from hud.eval import (
    DaytonaRuntime,
    DockerRuntime,
    HUDRuntime,
    LocalRuntime,
    ModalRuntime,
    Run,
    Runtime,
    SubprocessRuntime,
    Task,
    rollout,
)
from hud.telemetry.context import get_current_trace_id
from tests.eval.envs import SUMS_SOURCE, eventually, lab, solve
from tests.fixtures.envs import source
from tests.harness import ScriptedAgent, steps
from tests.harness.cloud import FakeDaytona, FakeModal
from tests.harness.runtime import host_on_runtime

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable
    from pathlib import Path

    from hud.environment import Environment
    from hud.eval import Provider
    from tests.harness import FakeDocker, FakeServices, HudEnv


@dataclass(frozen=True)
class World:
    """What a placement may stand up around the fixture: fake docker and fake HUD services."""

    path: Path
    fake_docker: FakeDocker
    services: FakeServices
    hud_env: HudEnv
    monkeypatch: pytest.MonkeyPatch


@asynccontextmanager
async def served_apart(path: Path) -> AsyncIterator[str]:
    """Serve ``path`` in a child process and yield its address, so no test shares its env."""
    async with SubprocessRuntime(path)(Task(env=path.stem, id="serve")) as served:
        yield served.url.removeprefix("tcp://")


@asynccontextmanager
async def in_process(world: World) -> AsyncIterator[Provider]:
    yield LocalRuntime(world.path)


@asynccontextmanager
async def child_process(world: World) -> AsyncIterator[Provider]:
    yield SubprocessRuntime(world.path)


@asynccontextmanager
async def container(world: World) -> AsyncIterator[Provider]:
    async with served_apart(world.path) as address:
        world.fake_docker.on(r"^run .* fixture:1$", stdout="fixture-1\n")
        world.fake_docker.on(r"^port fixture-1 8765$", stdout=f"{address}\n")
        yield DockerRuntime("fixture:1")


@asynccontextmanager
async def attached(world: World) -> AsyncIterator[Provider]:
    async with served_apart(world.path) as address:
        yield Runtime(f"tcp://{address}")


@asynccontextmanager
async def hud_tunnel(world: World) -> AsyncIterator[Provider]:
    world.hud_env.set(HUD_API_KEY="k")
    async with served_apart(world.path) as address:
        host_on_runtime(world.services, int(address.rsplit(":", 1)[1]))
        yield HUDRuntime()


@asynccontextmanager
async def modal(world: World) -> AsyncIterator[Provider]:
    async with served_apart(world.path) as address:
        host, port = address.rsplit(":", 1)
        FakeModal({"hud-fixture": (host, int(port))}).install(world.monkeypatch)
        yield ModalRuntime("hud-fixture")


@asynccontextmanager
async def daytona(world: World) -> AsyncIterator[Provider]:
    async with served_apart(world.path) as address:
        FakeDaytona(int(address.rsplit(":", 1)[1])).install(world.monkeypatch)
        yield DaytonaRuntime("hud-fixture")


PLACEMENTS: dict[str, Callable[[World], AbstractAsyncContextManager[Provider]]] = {
    "in-process": in_process,
    "child-process": child_process,
    "container": container,
    "attached": attached,
    "hud-tunnel": hud_tunnel,
    "modal": modal,
    "daytona": daytona,
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
        rollout_timeout=5.0,
        reward=0.0,
        status="error",
        stop_reason="timeout",
        error="rollout timed out after 5s during agent loop",
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
        rollout_timeout=5.0,
        reward=0.0,
        status="error",
        stop_reason="timeout",
        error="rollout timed out after 5s during grading",
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
    outcome: Outcome,
    placement: str,
    fake_docker: FakeDocker,
    services: FakeServices,
    hud_env: HudEnv,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    row = Task(
        env=outcome.fixture,
        id=outcome.task,
        args=outcome.args,
        agent_config=outcome.agent_config,
    )

    world = World(source(outcome.fixture), fake_docker, services, hud_env, monkeypatch)
    async with PLACEMENTS[placement](world) as runtime:
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


# ─── rollout contracts that do not depend on placement ─────────────────

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
    "a constructor source serves the row": Case(
        add(), ScriptedAgent(solve), COMPLETED, HOOKS, provider="constructor"
    ),
    # The child process logs its hooks in its own memory.
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
    "the rollout deadline covers provisioning": Case(
        add(),
        ScriptedAgent(solve),
        failed("rollout timed out after 0.2s during provisioning", stop_reason="timeout"),
        provider="hanging",
        rollout_timeout=0.2,
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
