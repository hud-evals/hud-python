"""What every runtime owes a rollout: how it ends, how a deadline or a cancel stops it, what
it refuses to provision, and the session files it moves to a verifier.

Every runtime ``hud.eval`` exports opens below as a row, around a fixture environment
from ``tests.fixtures.envs`` whose reward proves the path ran, and says how to tell,
afterwards, that it left its placement clean. Rows on a real daemon or account carry
the marks of the lane that has one. Each contract names the runtimes it does not
apply to, with the reason, and ``test_every_exported_runtime_is_a_row_of_every_contract``
holds the tables to the exported list.
"""

from __future__ import annotations

import asyncio
import contextlib
import importlib.util
import os
import re
import socket
import subprocess
import textwrap
import typing
import uuid
from collections import Counter
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import pytest
from dirty_equals import IsStr

from hud.agents.openai_compatible import OpenAIChatAgent
from hud.agents.types import OpenAIChatConfig
from hud.eval import (
    ComposeProject,
    DaytonaRuntime,
    DockerRuntime,
    HostedRuntime,
    HUDRuntime,
    LocalRuntime,
    ModalRuntime,
    Runtime,
    RuntimeConfig,
    RuntimeGPU,
    RuntimeLimits,
    RuntimeResources,
    Shared,
    SubprocessRuntime,
    Task,
    rollout,
)
from hud.eval import runtime as runtime_module
from tests.fixtures.envs import source
from tests.support import Reply, ScriptedAgent, actor, eventually, judge, solve, steps
from tests.support.runtime import SESSION, SESSIONS, host_on_runtime

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Awaitable, Callable
    from pathlib import Path

    from hud.agents.base import Agent
    from hud.eval import Provider
    from tests.support import FakeServices, HudEnv

ROLLOUTS = "/v2/rollouts/submit"


def serves(url: str) -> bool:
    host, port = url.removeprefix("tcp://").rsplit(":", 1)
    try:
        socket.create_connection((host, int(port)), timeout=1).close()
    except OSError:
        return False
    return True


def gone(pid: int) -> bool:
    """Whether process ``pid`` has exited (an unreaped zombie counts as exited)."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    status = subprocess.run(
        ["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True, check=False
    )
    return status.stdout.strip().startswith("Z")


def lifecycle(request: pytest.FixtureRequest) -> Callable[[], Awaitable[bool]]:
    """Have the fixture env log its lifecycle; clean once every env started in this process
    has stopped and every process an env ran in or started is gone. A child killed before
    it finished starting exits without its shutdown hooks, so a child and the daemon it
    started are held to exiting."""
    log = request.getfixturevalue("tmp_path") / "lifecycle.log"
    request.getfixturevalue("hud_env").set(HUD_FIXTURE_LIFECYCLE=str(log))

    async def clean() -> bool:
        events = log.read_text().split("\n") if log.exists() else []
        started = Counter(line.split()[1] for line in events if line.startswith("start "))
        stopped = Counter(line.split()[1] for line in events if line.startswith("stop "))
        here = str(os.getpid())
        daemons = {int(line.split()[1]) for line in events if line.startswith("daemon ")}
        children = {int(pid) for pid in started if pid != here} | daemons
        return started[here] == stopped[here] and all(gone(pid) for pid in children)

    return clean


#: What a runtime row opens: a provider (or ``HostedRuntime``) and how to tell it was left
#: clean.
Opened = tuple["Provider | HostedRuntime", "Callable[[], Awaitable[bool]]"]


@asynccontextmanager
async def in_process(
    path: Path, request: pytest.FixtureRequest, **limits: Any
) -> AsyncIterator[Opened]:
    clean = lifecycle(request)
    yield LocalRuntime(path, ready_timeout=limits.get("startup_limit") or 120), clean


@asynccontextmanager
async def child_process(
    path: Path, request: pytest.FixtureRequest, **limits: Any
) -> AsyncIterator[Opened]:
    clean = lifecycle(request)
    yield SubprocessRuntime(path, ready_timeout=limits.get("startup_limit") or 120), clean


@asynccontextmanager
async def shared_pool(
    path: Path, request: pytest.FixtureRequest, **limits: Any
) -> AsyncIterator[Opened]:
    clean = lifecycle(request)
    inner = LocalRuntime(path, ready_timeout=limits.get("startup_limit") or 120)
    yield Shared(inner, width=2), clean


@asynccontextmanager
async def attached(
    path: Path, request: pytest.FixtureRequest, **limits: Any
) -> AsyncIterator[Opened]:
    async with SubprocessRuntime(path)(Task(env=path.stem, id="serve")) as served:

        async def still_serving() -> bool:
            return serves(served.url)

        yield Runtime(served.url), still_serving


@asynccontextmanager
async def container(
    path: Path, request: pytest.FixtureRequest, **limits: Any
) -> AsyncIterator[Opened]:
    label = uuid.uuid4().hex
    image = request.getfixturevalue("fixture_image")(path.stem)
    config = (
        RuntimeConfig(limits=RuntimeLimits(startup_timeout_s=limits["startup_limit"]))
        if limits.get("startup_limit")
        else None
    )

    async def removed() -> bool:
        listing = await asyncio.create_subprocess_exec(
            *("docker", "ps", "--all", "--quiet", "--filter", f"label=hud-test={label}"),
            stdout=asyncio.subprocess.PIPE,
        )
        listed, _ = await listing.communicate()
        return listed.split() == []

    yield (
        DockerRuntime(image, run_args=["--label", f"hud-test={label}"], runtime_config=config),
        removed,
    )


@asynccontextmanager
async def hud_tunnel(
    path: Path, request: pytest.FixtureRequest, **limits: Any
) -> AsyncIterator[Opened]:
    services: FakeServices = request.getfixturevalue("services")
    request.getfixturevalue("hud_env").set(HUD_API_KEY="k")
    async with SubprocessRuntime(path)(Task(env=path.stem, id="serve")) as served:
        host_on_runtime(services, int(served.url.rsplit(":", 1)[1]))

        async def released() -> bool:
            created = len(services.requests("runtime", "POST", SESSIONS))
            return created > 0 and len(services.requests("runtime", "DELETE", SESSION)) == created

        yield HUDRuntime(run_timeout=limits.get("run_timeout")), released


@asynccontextmanager
async def modal_sandbox(
    path: Path, request: pytest.FixtureRequest, **limits: Any
) -> AsyncIterator[Opened]:
    import modal

    request.getfixturevalue("modal_account")
    image = request.getfixturevalue("modal_image")(path.stem)
    config = (
        RuntimeConfig(limits=RuntimeLimits(startup_timeout_s=limits["startup_limit"]))
        if limits.get("startup_limit")
        else None
    )
    inner = ModalRuntime(image=image, workdir="/hud", runtime_config=config)
    sandboxes: list[str] = []

    @asynccontextmanager
    async def placed(task: Task) -> AsyncIterator[Runtime]:
        async with inner(task) as runtime:
            sandboxes.append(runtime.params["instance_id"])
            yield runtime

    async def terminated() -> bool:
        for instance in sandboxes:
            sandbox = await modal.Sandbox.from_id.aio(instance)
            if await sandbox.poll.aio() is None:
                return False
        return True

    yield placed, terminated


@asynccontextmanager
async def hosted(
    path: Path, request: pytest.FixtureRequest, **limits: Any
) -> AsyncIterator[Opened]:
    """The platform runs the rollout and reports it running until it is cancelled."""
    platform = request.getfixturevalue("platform")
    platform.next_rollout = [{"status": "running"}]

    async def cancelled() -> bool:
        traces = platform.traces
        return traces != {} and all(trace.cancelled for trace in traces.values())

    yield HostedRuntime(poll_interval=0, run_timeout=limits.get("run_timeout")), cancelled


@dataclass(frozen=True)
class Placement:
    kind: type
    open: Callable[..., contextlib.AbstractAsyncContextManager[Opened]]
    marks: tuple[pytest.MarkDecorator, ...] = ()
    #: How long provisioning may take inside a rollout deadline.
    patience: float = 5.0


RUNTIMES: dict[str, Placement] = {
    "in-process": Placement(LocalRuntime, in_process),
    "child-process": Placement(SubprocessRuntime, child_process),
    "shared-pool": Placement(Shared, shared_pool),
    "attached": Placement(Runtime, attached),
    "container": Placement(DockerRuntime, container, (pytest.mark.e2e, pytest.mark.docker)),
    "hud-tunnel": Placement(HUDRuntime, hud_tunnel),
    "modal": Placement(ModalRuntime, modal_sandbox, (pytest.mark.e2e, pytest.mark.modal), 90.0),
    "hosted": Placement(HostedRuntime, hosted, patience=1.0),
}
UNPLACED: dict[type, str] = {
    DaytonaRuntime: "needs a Daytona account, and no lane has one; its refusals are rows below"
}
HOSTED = "the platform runs the environment and the agent; test_hosted folds its report"
NOT_APPLICABLE: dict[str, dict[type, str]] = {
    "outcomes": UNPLACED | {HostedRuntime: HOSTED},
    "unserved": UNPLACED
    | {
        HostedRuntime: HOSTED,
        Runtime: "a borrowed address is served before the rollout starts",
        HUDRuntime: "the platform serves the environment before the rollout starts",
    },
    "deadline": UNPLACED
    | dict.fromkeys(
        (SubprocessRuntime, Shared, Runtime, HUDRuntime, DockerRuntime, ModalRuntime),
        "a deadline tears the placement down the way a cancel does, which the cancel "
        "contract holds on every runtime; rollout decides what the run says",
    ),
    "cancel": UNPLACED,
    "session files": UNPLACED
    | {
        LocalRuntime: "rejects a runtime_config, so a verifier cannot ask for its own runtime",
        SubprocessRuntime: "rejects a runtime_config, so a verifier cannot ask for its own runtime",
        Shared: "transports what its inner provider transports",
        Runtime: "cannot export a session; the verifier runs beside the actor (test_verifier)",
        HUDRuntime: "cannot export a session; the verifier runs beside the actor (test_verifier)",
        HostedRuntime: "the platform places the verifier from the submitted row (test_hosted)",
    },
}


def rows(contract: str) -> list[Any]:
    """The runtimes ``contract`` runs over, with their lane marks."""
    return [
        pytest.param(name, id=name, marks=placement.marks)
        for name, placement in RUNTIMES.items()
        if placement.kind not in NOT_APPLICABLE[contract]
    ]


def exported_runtimes() -> set[type]:
    """Every runtime ``hud.eval.runtime`` exports: the classes a rollout can be placed with."""
    exported = (getattr(runtime_module, name) for name in runtime_module.__all__)
    return {
        kind
        for kind in exported
        if isinstance(kind, type)
        and typing.Protocol not in kind.__mro__
        and any("__call__" in vars(base) or "run" in vars(base) for base in kind.__mro__[:-1])
    }


@pytest.mark.parametrize("contract", [*NOT_APPLICABLE, "refusals"])
def test_every_exported_runtime_is_a_row_of_every_contract(contract: str) -> None:
    if contract == "refusals":
        assert set(REFUSALS) | set(UNREFUSING) == exported_runtimes()
    else:
        placed = {placement.kind for placement in RUNTIMES.values()}
        assert placed | set(NOT_APPLICABLE[contract]) == exported_runtimes()


# ─── how a rollout ends ─────────────────────────────────────────────────


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
    #: Whether the outcome depends on the runtime: placing the env and reaching it, the
    #: byte path a large frame takes, and tearing down after a start that failed. The
    #: rest is decided by ``rollout`` or inside the env, the same on every runtime.
    everywhere: bool = False


def echo(agent: Callable[[], Agent], word: str = "tangerine", **outcome: Any) -> Outcome:
    return Outcome("echo", "echo", agent, args={"word": word}, **outcome)


def faulty(task: str, error: Any, **outcome: Any) -> Outcome:
    return Outcome(
        "faulty", task, lambda: ScriptedAgent("x"), 0.0, "error", error, graded=False, **outcome
    )


#: Over a WebSocket's default 1 MiB message limit, under the 16 MiB frame limit.
LARGE = "x" * (4 * 1024 * 1024)
OUTCOMES = {
    "the right answer earns the reward": echo(
        lambda: ScriptedAgent("tangerine"), reward=1.0, status="completed", everywhere=True
    ),
    "a frame of several megabytes crosses to the env and back": echo(
        lambda: ScriptedAgent(LARGE),
        word=LARGE,
        reward=1.0,
        status="completed",
        everywhere=True,
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
    "a boolean score leaves the run ungraded": faulty(
        "bool_score", IsStr(regex=r"\[grading\] .*must include a numeric 'score'.*")
    ),
    "a start that raises fails the run before the agent sees a prompt": faulty(
        "start_raises",
        IsStr(regex=r"\[starting task\] .*start exploded.*"),
        started=False,
        everywhere=True,
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


@pytest.mark.parametrize(
    ("outcome", "runtime"),
    [
        pytest.param(outcome, row.values[0], id=f"{name}-{row.id}", marks=row.marks)
        for name, outcome in OUTCOMES.items()
        for row in rows("outcomes")
        if outcome.everywhere or row.id == "in-process"
    ],
)
async def test_a_rollout_ends_the_same_way_wherever_it_runs(
    outcome: Outcome, runtime: str, request: pytest.FixtureRequest
) -> None:
    row = Task(
        env=outcome.fixture, id=outcome.task, args=outcome.args, agent_config=outcome.agent_config
    )

    async with RUNTIMES[runtime].open(source(outcome.fixture), request) as (provider, clean):
        job = await row.run(outcome.agent(), runtime=provider)
        await eventually(clean)

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


@pytest.mark.parametrize("runtime", rows("unserved"))
@pytest.mark.parametrize("fixture", UNSERVED.values(), ids=UNSERVED.keys())
async def test_an_env_that_cannot_serve_fails_the_run_before_the_prompt_and_is_cleaned_up(
    fixture: str, runtime: str, request: pytest.FixtureRequest
) -> None:
    opened = RUNTIMES[runtime].open(source(fixture), request, startup_limit=5)

    async with opened as (provider, clean):
        job = await Task(env=fixture, id="never").run(ScriptedAgent("x"), runtime=provider)
        await eventually(clean)

    (run,) = job.runs
    assert (run.trace.status, run.prompt, job.errors) == ("error", None, [run])
    assert run.trace.error == IsStr(regex=r"(?s)\[(provisioning|starting task)\] .+")


# ─── deadlines and cancellation ────────────────────────────────────────

ECHO = Task(env="echo", id="echo", args={"word": "tangerine"})
# The row, the phase the deadline lands in, and how long the agent keeps running after it answers.
MOMENTS: dict[str, tuple[Task, str | None, float]] = {
    "during the agent loop": (ECHO, "agent loop", 3600),
    "during grading": (Task(env="faulty", id="hang_grading"), "grading", 0),
    "before the platform accepts it": (ECHO, None, 0),
}


def agent_for(runtime: str, linger: float = 3600) -> Agent:
    """An agent that answers, then keeps running for ``linger`` seconds; the platform runs
    only the agent types it knows."""
    if RUNTIMES[runtime].kind is HostedRuntime:
        return OpenAIChatAgent(
            OpenAIChatConfig(model="test-model", api_key="secret", base_url="http://model.test")
        )
    return ScriptedAgent("tangerine", linger=linger)


DEADLINES = [
    *(
        pytest.param(
            row.values[0], moment, "rollout_timeout", id=f"{moment}-{row.id}", marks=row.marks
        )
        for row in rows("deadline")
        for moment in ("during the agent loop", "during grading")
        if not (row.id == "hosted" and moment == "during grading")
    ),
    pytest.param("hosted", "before the platform accepts it", "rollout_timeout", id="unsubmitted"),
    pytest.param(
        "hud-tunnel", "during the agent loop", "constructor", id="deprecated-run_timeout-hud"
    ),
    pytest.param(
        "hosted", "during the agent loop", "constructor", id="deprecated-run_timeout-hosted"
    ),
]


@pytest.mark.parametrize(("runtime", "moment", "api"), DEADLINES)
async def test_a_deadline_ends_the_run_as_timed_out_and_leaves_its_placement_clean(
    runtime: str, moment: str, api: str, services: FakeServices, request: pytest.FixtureRequest
) -> None:
    row, phase, linger = MOMENTS[moment]
    within = RUNTIMES[runtime].patience
    deprecated = api == "constructor"
    opened = RUNTIMES[runtime].open(
        source(row.env), request, run_timeout=within if deprecated else None
    )
    warned = (
        pytest.warns(DeprecationWarning, match="rollout_timeout=... to Task.run")
        if deprecated
        else contextlib.nullcontext()
    )

    with warned:
        async with opened as (provider, clean):
            if phase is None:
                services.route("api", "POST", ROLLOUTS, Reply(json={}, delay=5))
            job = await row.run(
                agent_for(runtime, linger),
                runtime=provider,
                rollout_timeout=None if deprecated else within,
            )
            await eventually(clean, within=max(10, within))

    (run,) = job.runs
    error = (
        IsStr(regex=rf"hosted rollout [0-9a-f]{{32}} did not finish within {within:g}s")
        if RUNTIMES[runtime].kind is HostedRuntime
        else f"rollout timed out after {within:g}s during {phase}"
    )
    assert (run.trace.status, run.trace.stop_reason, run.reward, run.trace.error) == (
        "error",
        "timeout",
        0.0,
        error,
    )
    assert job.errors == [run]


@pytest.mark.parametrize("runtime", rows("cancel"))
async def test_cancelling_a_rollout_propagates_and_leaves_its_placement_clean(
    runtime: str, services: FakeServices, request: pytest.FixtureRequest
) -> None:
    agent = agent_for(runtime)

    def underway() -> bool:
        prompted = isinstance(agent, ScriptedAgent) and agent.prompts != []
        return prompted or services.requests("api", "POST", ROLLOUTS) != []

    async with RUNTIMES[runtime].open(source("echo"), request) as (provider, clean):
        pending = asyncio.create_task(ECHO.run(agent, runtime=provider))
        await eventually(underway, within=RUNTIMES[runtime].patience)
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending

        await eventually(clean, within=max(10, RUNTIMES[runtime].patience))


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


# ─── a provider that never places the row ──────────────────────────────

HEX_ID = IsStr(regex=r"[0-9a-f]{32}")


@asynccontextmanager
async def broken_provider(task: Task) -> AsyncIterator[Runtime]:
    error = EOFError("sandbox closed")
    error.add_note("env output: ImportError: bugs")
    raise error
    yield Runtime("tcp://127.0.0.1:1")


@asynccontextmanager
async def hanging_provider(task: Task) -> AsyncIterator[Runtime]:
    await asyncio.Event().wait()
    yield Runtime("tcp://127.0.0.1:1")


ADD = Task(env="lab", id="add", args={"a": 2, "b": 3})
UNPLACED_ROWS: dict[str, tuple[Provider, float | None, str, str | None]] = {
    "the rollout deadline covers provisioning": (
        hanging_provider,
        0.2,
        "rollout timed out after 0.2s during provisioning",
        "timeout",
    ),
    "a provider that fails reports its notes": (
        broken_provider,
        None,
        "[provisioning] EOFError: sandbox closed\nenv output: ImportError: bugs",
        None,
    ),
}


@pytest.mark.parametrize(
    ("provider", "rollout_timeout", "error", "stop_reason"),
    UNPLACED_ROWS.values(),
    ids=UNPLACED_ROWS.keys(),
)
async def test_a_row_that_is_never_placed_still_returns_a_run_that_says_why(
    provider: Provider, rollout_timeout: float | None, error: str, stop_reason: str | None
) -> None:
    run = await rollout(
        ADD, ScriptedAgent(solve), runtime=provider, rollout_timeout=rollout_timeout
    )

    assert (run.trace.status, run.trace.stop_reason, run.reward, run.grade.raw) == (
        "error",
        stop_reason,
        0.0,
        {},
    )
    assert (run.trace.error, run.prompt, run.runtime) == (error, None, None)
    assert [step["source"] for step in steps(run.trace_id)] == ["system"]
    assert (run.trace_id, run.job_id, run.slug) == (HEX_ID, HEX_ID, ADD.slug)


# ─── what each runtime refuses to provision ────────────────────────────

NO_DOCKER = {"PATH": "/nonexistent"}
NO_DAYTONA = {"DAYTONA_API_KEY": "k", "DAYTONA_API_URL": "http://127.0.0.1:9"}


def compose(document: str, **extra: Any) -> Callable[[Path], RuntimeConfig]:
    def build(root: Path) -> RuntimeConfig:
        (root / ".env").write_text("")
        path = root / "compose.yaml"
        path.write_text(textwrap.dedent(document))
        return RuntimeConfig(compose=ComposeProject(document=path), **extra)

    return build


def config(**fields: Any) -> Callable[[Path], RuntimeConfig]:
    return lambda root: RuntimeConfig(**fields)


def gpu(**fields: Any) -> RuntimeResources:
    return RuntimeResources(gpu=RuntimeGPU(**fields))


def chat_agent() -> Agent:
    return OpenAIChatAgent(
        OpenAIChatConfig(model="test-model", api_key="secret", base_url="http://model.test")
    )


def needs(extra: str) -> pytest.MarkDecorator:
    return pytest.mark.skipif(
        importlib.util.find_spec(extra) is None, reason=f"needs the {extra} extra"
    )


def with_a_secret() -> ModalRuntime:
    import modal

    return ModalRuntime(sandbox_secrets=[modal.Secret.from_dict({"API": "x"})])


@dataclass(frozen=True)
class Refusal:
    runtime: Callable[[], Any]
    message: str
    config: Callable[[Path], RuntimeConfig | None] = lambda root: None
    variables: dict[str, str | None] = field(default_factory=dict)
    agent: Callable[[], Agent] = lambda: ScriptedAgent("x")


SIMPLE = "services:\n  main: {image: lab}\n"
CHAINED = """
services:
  main: {image: lab, network_mode: "service:gateway"}
  gateway: {image: gateway}
"""
REFUSALS: dict[type, dict[str, Refusal]] = {
    LocalRuntime: {
        "a runtime_config in process": Refusal(
            lambda: LocalRuntime(source("echo")),
            "LocalRuntime does not support task runtime_config",
            config(image="lab"),
        ),
    },
    SubprocessRuntime: {
        "a runtime_config in a child process": Refusal(
            lambda: SubprocessRuntime(source("echo")),
            "SubprocessRuntime does not support task runtime_config",
            config(image="lab"),
        ),
    },
    DockerRuntime: {
        "a GPU type": Refusal(
            DockerRuntime,
            "cannot select GPUs by type",
            config(image="lab:1", resources=gpu(type="H100")),
            NO_DOCKER,
        ),
        "run_args for a Compose project": Refusal(
            lambda: DockerRuntime(run_args=["--privileged"]),
            "run_args apply only to image environments",
            compose(CHAINED),
            NO_DOCKER,
        ),
        "a network cycle": Refusal(
            DockerRuntime,
            "network_mode service cycle includes 'main'",
            compose(
                "services:\n  main: {image: lab, network_mode: 'service:relay'}\n"
                "  relay: {image: relay, network_mode: 'service:main'}\n"
            ),
            NO_DOCKER,
        ),
        "a Compose include": Refusal(
            DockerRuntime,
            "does not support Compose include or extends",
            compose("include: [other.yaml]\n" + SIMPLE),
            NO_DOCKER,
        ),
        "a variable only the host sets": Refusal(
            DockerRuntime,
            "variable 'HOST_ONLY' is not set by the project .env",
            compose("services:\n  main: {image: $HOST_ONLY}\n"),
            NO_DOCKER,
        ),
    },
    ModalRuntime: {
        "alternative GPU types": Refusal(
            lambda: ModalRuntime("hud-lab"),
            "alternative GPU types",
            config(resources=gpu(type=["A100", "H100"])),
        ),
        "an OS requirement": Refusal(
            lambda: ModalRuntime("hud-lab"),
            "resources.os",
            config(resources=RuntimeResources(os="windows")),
        ),
        "a GPU for a Compose project": Refusal(
            ModalRuntime, "cannot attach GPUs", compose(SIMPLE, resources=gpu())
        ),
        "sandbox secrets for a Compose project": Refusal(
            with_a_secret,
            "sandbox secrets require an image runtime",
            compose(SIMPLE),
        ),
    },
    DaytonaRuntime: {
        "a Compose project": Refusal(
            lambda: DaytonaRuntime("snap"),
            "does not support runtime_config.compose",
            compose(SIMPLE),
            NO_DAYTONA,
        ),
        "a run limit": Refusal(
            lambda: DaytonaRuntime("snap"),
            "does not support runtime_config.run_timeout_s",
            config(limits=RuntimeLimits(run_timeout_s=60)),
            NO_DAYTONA,
        ),
        "an OS requirement": Refusal(
            DaytonaRuntime,
            "does not support runtime_config.resources.os",
            config(image="lab", resources=RuntimeResources(os="linux")),
            NO_DAYTONA,
        ),
        "a fraction of a CPU": Refusal(
            DaytonaRuntime,
            "needs a whole number of CPUs, got 1.5",
            config(image="lab", resources=RuntimeResources(cpu=1.5)),
            NO_DAYTONA,
        ),
        "neither a snapshot nor an image": Refusal(
            DaytonaRuntime, "requires snapshot_name or runtime_config.image", variables=NO_DAYTONA
        ),
        "resources for a snapshot already built": Refusal(
            lambda: DaytonaRuntime("snap"),
            "cannot resize an already-built snapshot",
            config(resources=RuntimeResources(cpu=2)),
            NO_DAYTONA,
        ),
    },
    HUDRuntime: {
        "a GPU the deployed env cannot have": Refusal(
            HUDRuntime,
            "HUDRuntime cannot honor this task's declared placement requirements or limits",
            config(resources=gpu(type="H100")),
        ),
        "no API key": Refusal(
            HUDRuntime, "HUD runtime tunnel requires HUD_API_KEY", variables={"HUD_API_KEY": None}
        ),
    },
    HostedRuntime: {
        "an agent the platform cannot run": Refusal(
            HostedRuntime, "hosted execution supports the registered agent types"
        ),
        "no API key": Refusal(
            HostedRuntime,
            "HUD_API_KEY is required",
            variables={"HUD_API_KEY": None},
            agent=chat_agent,
        ),
    },
}
UNREFUSING = {
    Runtime: "a borrowed runtime provisions nothing, so it has nothing to refuse",
    Shared: "refuses what its inner provider refuses",
}
EXTRAS: dict[type, str] = {ModalRuntime: "modal", DaytonaRuntime: "daytona"}


@pytest.mark.parametrize(
    "refusal",
    [
        pytest.param(
            refusal,
            id=f"{kind.__name__}-{name}",
            marks=[needs(EXTRAS[kind])] if kind in EXTRAS else [],
        )
        for kind, refusals in REFUSALS.items()
        for name, refusal in refusals.items()
    ],
)
async def test_a_row_a_runtime_cannot_provision_fails_before_the_prompt_and_asks_nothing(
    refusal: Refusal, services: FakeServices, hud_env: HudEnv, tmp_path: Path
) -> None:
    hud_env.set(**({"HUD_API_KEY": "k"} | refusal.variables))
    row = Task(env="echo", id="echo", args={"word": "x"}, runtime_config=refusal.config(tmp_path))

    job = await row.run(refusal.agent(), runtime=refusal.runtime())

    (run,) = job.runs
    assert (run.trace.status, run.prompt, job.errors) == ("error", None, [run])
    assert re.search(re.escape(refusal.message), run.trace.error or "")
    assert services.requests() == []


# ─── session files for a verifier in its own sandbox ───────────────────


@pytest.mark.parametrize("runtime", rows("session files"))
@pytest.mark.parametrize(("link", "reward"), [(False, 1.0), (True, 0.0)], ids=["files", "a-link"])
async def test_a_verifier_in_its_own_sandbox_receives_the_actors_session_files(
    link: bool, reward: float, runtime: str, request: pytest.FixtureRequest
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

    async with RUNTIMES[runtime].open(source("sessions"), request) as (provider, clean):
        run = (await row.run(ScriptedAgent("secret"), runtime=provider)).runs[0]
        await eventually(clean)

    assert run.reward == reward
    assert ("runtime session contains" in (run.trace.error or "")) is link
