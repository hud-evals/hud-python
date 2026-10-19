"""What every runtime owes a rollout: it places the environment, carries the control
channel's frames to it, and tears it down, whether the rollout ends, fails to start,
or is cancelled; it fails an environment that cannot serve; it moves a session's files
to a verifier with a sandbox of its own; and it refuses what it cannot provision.

``rollout`` itself decides how a run ends (``test_rollout``), the same on every
runtime, so these contracts hold only what a runtime adds. Each runtime opens below
around a fixture environment from ``tests.fixtures.envs`` and says how to tell,
afterwards, that it left nothing behind. Rows on a real daemon or account carry the
marks of the lane that has one.
"""

from __future__ import annotations

import asyncio
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
)
from hud.eval import runtime as runtime_module
from tests.fixtures.envs import source
from tests.support import ScriptedAgent, eventually
from tests.support.runtime import SESSION, SESSIONS, host_on_runtime

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Awaitable, Callable
    from contextlib import AbstractAsyncContextManager
    from pathlib import Path

    from hud.agents.base import Agent
    from hud.eval import Provider
    from tests.support import FakeServices, HudEnv

#: A placed runtime: the provider a row runs on, and whether it has left nothing behind.
Placed = tuple["Provider", "Callable[[], Awaitable[bool]]"]


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


def nothing_left(request: pytest.FixtureRequest) -> Callable[[], Awaitable[bool]]:
    """Have the fixture env log its lifecycle, and tell whether it left nothing behind.

    Nothing is left once every env started in this process has stopped and every
    process an env ran in or started is gone. A child killed before it finished
    starting exits without running its shutdown hooks, so a child, and the daemon it
    started, are held to exiting rather than to stopping.
    """
    log = request.getfixturevalue("tmp_path") / "lifecycle.log"
    request.getfixturevalue("hud_env").set(HUD_FIXTURE_LIFECYCLE=str(log))

    async def clean() -> bool:
        lines = log.read_text().splitlines() if log.exists() else []
        events = [line.split()[:2] for line in lines]
        started = Counter(pid for event, pid in events if event == "start")
        stopped = Counter(pid for event, pid in events if event == "stop")
        here = str(os.getpid())
        processes = {int(pid) for event, pid in events if event != "stop" and pid != here}
        return started[here] == stopped[here] and all(gone(pid) for pid in processes)

    return clean


def startup(limit: float | None) -> RuntimeConfig | None:
    return RuntimeConfig(limits=RuntimeLimits(startup_timeout_s=limit)) if limit else None


@asynccontextmanager
async def in_process(
    path: Path, request: pytest.FixtureRequest, startup_limit: float | None
) -> AsyncIterator[Placed]:
    yield LocalRuntime(path, ready_timeout=startup_limit or 120), nothing_left(request)


@asynccontextmanager
async def child_process(
    path: Path, request: pytest.FixtureRequest, startup_limit: float | None
) -> AsyncIterator[Placed]:
    yield SubprocessRuntime(path, ready_timeout=startup_limit or 120), nothing_left(request)


@asynccontextmanager
async def shared_pool(
    path: Path, request: pytest.FixtureRequest, startup_limit: float | None
) -> AsyncIterator[Placed]:
    inner = LocalRuntime(path, ready_timeout=startup_limit or 120)
    yield Shared(inner, width=2), nothing_left(request)


@asynccontextmanager
async def attached(
    path: Path, request: pytest.FixtureRequest, startup_limit: float | None
) -> AsyncIterator[Placed]:
    """A runtime borrowing an env served elsewhere, which must still serve after the run."""
    async with SubprocessRuntime(path)(Task(env=path.stem, id="serve")) as served:

        async def still_serving() -> bool:
            return serves(served.url)

        yield Runtime(served.url), still_serving


@asynccontextmanager
async def container(
    path: Path, request: pytest.FixtureRequest, startup_limit: float | None
) -> AsyncIterator[Placed]:
    label = uuid.uuid4().hex
    image = request.getfixturevalue("fixture_image")(path.stem)

    async def removed() -> bool:
        listing = await asyncio.create_subprocess_exec(
            *("docker", "ps", "--all", "--quiet", "--filter", f"label=hud-test={label}"),
            stdout=asyncio.subprocess.PIPE,
        )
        listed, _ = await listing.communicate()
        return listed.split() == []

    labelled = ["--label", f"hud-test={label}"]
    yield DockerRuntime(image, run_args=labelled, runtime_config=startup(startup_limit)), removed


@asynccontextmanager
async def hud_tunnel(
    path: Path, request: pytest.FixtureRequest, startup_limit: float | None
) -> AsyncIterator[Placed]:
    """The fake HUD runtime service tunnels each session to the env served here."""
    services: FakeServices = request.getfixturevalue("services")
    request.getfixturevalue("hud_env").set(HUD_API_KEY="k")
    async with SubprocessRuntime(path)(Task(env=path.stem, id="serve")) as served:
        host_on_runtime(services, int(served.url.rsplit(":", 1)[1]))

        async def released() -> bool:
            created = len(services.requests("runtime", "POST", SESSIONS))
            return created > 0 and len(services.requests("runtime", "DELETE", SESSION)) == created

        yield HUDRuntime(), released


@asynccontextmanager
async def modal_sandbox(
    path: Path, request: pytest.FixtureRequest, startup_limit: float | None
) -> AsyncIterator[Placed]:
    import modal

    request.getfixturevalue("modal_account")
    image = request.getfixturevalue("modal_image")(path.stem)
    inner = ModalRuntime(image=image, workdir="/hud", runtime_config=startup(startup_limit))
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


@dataclass(frozen=True)
class Placement:
    kind: type
    open: Callable[[Path, pytest.FixtureRequest, float | None], AbstractAsyncContextManager[Placed]]
    marks: tuple[pytest.MarkDecorator, ...] = ()
    #: How long placing the env may take before the row is underway.
    patience: float = 5.0


RUNTIMES: dict[str, Placement] = {
    "in-process": Placement(LocalRuntime, in_process),
    "child-process": Placement(SubprocessRuntime, child_process),
    "shared-pool": Placement(Shared, shared_pool),
    "attached": Placement(Runtime, attached),
    "container": Placement(DockerRuntime, container, (pytest.mark.e2e, pytest.mark.docker)),
    "hud-tunnel": Placement(HUDRuntime, hud_tunnel),
    "modal": Placement(ModalRuntime, modal_sandbox, (pytest.mark.e2e, pytest.mark.modal), 90.0),
}
#: Runtimes no placement contract runs on.
UNPLACED: dict[type, str] = {
    DaytonaRuntime: "needs a Daytona account, and no lane has one; its refusals are rows below",
    HostedRuntime: "the platform runs the whole rollout; test_hosted holds what it owes",
}
NOT_APPLICABLE: dict[str, dict[type, str]] = {
    "placing": UNPLACED,
    "unserved": UNPLACED
    | {
        Runtime: "a borrowed address is served before the rollout starts",
        HUDRuntime: "the platform serves the environment before the rollout starts",
    },
    "cancel": UNPLACED,
    "session files": UNPLACED
    | {
        LocalRuntime: "rejects a runtime_config, so a verifier cannot ask for its own runtime",
        SubprocessRuntime: "rejects a runtime_config, so a verifier cannot ask for its own runtime",
        Shared: "transports what its inner provider transports",
        Runtime: "cannot export a session; the verifier runs beside the actor (test_verifier)",
        HUDRuntime: "cannot export a session; the verifier runs beside the actor (test_verifier)",
    },
}


def rows(contract: str) -> list[Any]:
    """The runtimes ``contract`` runs on, with their lane marks."""
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


@pytest.mark.parametrize("contract", NOT_APPLICABLE)
def test_every_exported_runtime_is_a_row_of_every_contract(contract: str) -> None:
    placed = {placement.kind for placement in RUNTIMES.values()}

    assert placed | set(NOT_APPLICABLE[contract]) == exported_runtimes()


def test_every_exported_runtime_has_its_refusals_in_the_table() -> None:
    assert set(REFUSALS) | set(UNREFUSING) == exported_runtimes()


# ─── placing, carrying and tearing down ────────────────────────────────

#: Over a WebSocket's default 1 MiB message limit, and under the 16 MiB frame limit.
LARGE = "x" * (4 * 1024 * 1024)
PLACINGS: dict[str, tuple[Task, str, tuple[float, str, bool]]] = {
    "a row runs to its grade": (
        Task(env="echo", id="echo", args={"word": "tangerine"}),
        "tangerine",
        (1.0, "completed", True),
    ),
    "a frame of several megabytes crosses to the env and back": (
        Task(env="echo", id="echo", args={"word": LARGE}),
        LARGE,
        (1.0, "completed", True),
    ),
    "a start that fails still tears the placement down": (
        Task(env="faulty", id="start_raises"),
        "x",
        (0.0, "error", False),
    ),
}


@pytest.mark.parametrize("runtime", rows("placing"))
@pytest.mark.parametrize(("row", "answer", "expected"), PLACINGS.values(), ids=PLACINGS.keys())
async def test_a_runtime_places_the_env_carries_its_frames_and_leaves_nothing_behind(
    row: Task,
    answer: str,
    expected: tuple[float, str, bool],
    runtime: str,
    request: pytest.FixtureRequest,
) -> None:
    async with RUNTIMES[runtime].open(source(row.env), request, None) as (provider, clean):
        job = await row.run(ScriptedAgent(answer), runtime=provider)
        await eventually(clean)

    (run,) = job.runs
    assert (run.reward, run.trace.status, run.prompt is not None) == expected


@pytest.mark.parametrize("runtime", rows("unserved"))
@pytest.mark.parametrize(
    "fixture",
    ["broken", "stalls"],
    ids=["an-env-that-exits-at-once", "an-env-that-never-finishes-starting"],
)
async def test_an_env_that_cannot_serve_fails_the_run_before_the_prompt_and_is_cleaned_up(
    fixture: str, runtime: str, request: pytest.FixtureRequest
) -> None:
    async with RUNTIMES[runtime].open(source(fixture), request, 5) as (provider, clean):
        job = await Task(env=fixture, id="never").run(ScriptedAgent("x"), runtime=provider)
        await eventually(clean)

    (run,) = job.runs
    assert (run.trace.status, run.prompt, job.errors) == ("error", None, [run])
    assert run.trace.error == IsStr(regex=r"(?s)\[(provisioning|starting task)\] .+")


@pytest.mark.parametrize("runtime", rows("cancel"))
async def test_cancelling_a_rollout_propagates_and_leaves_nothing_behind(
    runtime: str, request: pytest.FixtureRequest
) -> None:
    agent = ScriptedAgent("tangerine", linger=3600)
    row = Task(env="echo", id="echo", args={"word": "tangerine"})

    async with RUNTIMES[runtime].open(source("echo"), request, None) as (provider, clean):
        pending = asyncio.create_task(row.run(agent, runtime=provider))
        await eventually(lambda: agent.prompts != [], within=RUNTIMES[runtime].patience)
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending

        await eventually(clean, within=max(10, RUNTIMES[runtime].patience))


@pytest.mark.parametrize("runtime", rows("session files"))
@pytest.mark.parametrize(("link", "reward"), [(False, 1.0), (True, 0.0)], ids=["files", "a-link"])
async def test_a_verifier_in_its_own_sandbox_receives_the_actors_session_files(
    link: bool, reward: float, runtime: str, request: pytest.FixtureRequest
) -> None:
    """The verifier's own limits give it its own sandbox, so the actor's files must move;
    a session holding anything but files and directories does not."""
    row = Task(
        env="sessions",
        id="solve",
        args={"link": link},
        verifier=Task(env="sessions", id="verify", runtime_config=startup(600)),
    )

    async with RUNTIMES[runtime].open(source("sessions"), request, None) as (provider, clean):
        run = (await row.run(ScriptedAgent("secret"), runtime=provider)).runs[0]
        await eventually(clean)

    assert run.reward == reward
    assert ("runtime session contains" in (run.trace.error or "")) is link


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
