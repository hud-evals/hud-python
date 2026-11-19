"""What every runtime owes a rollout: it places the environment, carries the control
channel's frames to it and tears it down, whether the run ends, the env cannot serve,
or the rollout is cancelled.

Each runtime serves a fixture environment from ``tests.fixtures.envs``. Those envs log
their lifecycle and start a daemon when ``HUD_FIXTURE_LIFECYCLE`` names a file, so a
runtime is clean once every env it started here has stopped and every process an env
ran in or started is gone.
"""

from __future__ import annotations

import asyncio
import os
import socket
import uuid
from collections import Counter
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

import pytest

from hud import Task
from hud.agents.base import Agent
from hud.runtime import (
    DockerEndpoint,
    DockerRuntime,
    HUDRuntime,
    LocalRuntime,
    ModalRuntime,
    Provider,
    Runtime,
    RuntimeConfig,
    RuntimeData,
    RuntimeDataFile,
    Shared,
    SubprocessRuntime,
)
from tests.conftest import eventually
from tests.fixtures.agents import ScriptedAgent
from tests.fixtures.envs import source
from tests.runtime.conftest import SESSION, SESSIONS, gone, host_on_runtime

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Awaitable, Callable

    from hud import Run
    from hud.runtime import Provider
    from tests.fixtures.services import FakeServices

CONTAINER = pytest.param("DockerRuntime", marks=[pytest.mark.e2e, pytest.mark.docker])
MODAL = pytest.param("ModalRuntime", marks=[pytest.mark.e2e, pytest.mark.modal])
EVERY_RUNTIME = [
    "LocalRuntime",
    "SubprocessRuntime",
    "Shared",
    "Runtime",
    "HUDRuntime",
    CONTAINER,
    MODAL,
]


def serves(url: str) -> bool:
    host, port = url.removeprefix("tcp://").rsplit(":", 1)
    try:
        socket.create_connection((host, int(port)), timeout=1).close()
    except OSError:
        return False
    return True


@asynccontextmanager
async def placed(
    runtime: str, fixture: str, request: pytest.FixtureRequest, startup: int = 120
) -> AsyncGenerator[tuple[Provider, Callable[[], Awaitable[bool]]]]:
    """Place fixture env ``fixture`` on ``runtime``; yield it and its cleanliness check."""
    path = source(fixture)
    log = request.getfixturevalue("tmp_path") / "lifecycle.log"
    request.getfixturevalue("hud_env").set(HUD_FIXTURE_LIFECYCLE=str(log))

    async def nothing_left() -> bool:
        events = [
            line.split()[:2] for line in (log.read_text() if log.exists() else "").splitlines()
        ]
        here = str(os.getpid())
        started = Counter(pid for event, pid in events if event == "start")
        stopped = Counter(pid for event, pid in events if event == "stop")
        processes = {int(pid) for event, pid in events if event != "stop" and pid != here}
        return started[here] == stopped[here] and all(gone(pid) for pid in processes)

    match runtime:
        case "LocalRuntime":
            yield LocalRuntime(path, ready_timeout=startup), nothing_left
        case "SubprocessRuntime":
            yield SubprocessRuntime(path, ready_timeout=startup), nothing_left
        case "Shared":
            yield Shared(LocalRuntime(path, ready_timeout=startup), width=2), nothing_left
        case "Runtime":
            async with SubprocessRuntime(path)(Task(env=fixture, id="serve")) as served:

                async def still_serving() -> bool:
                    return serves(served.url)

                yield Runtime(served.url), still_serving
        case "HUDRuntime":
            services: FakeServices = request.getfixturevalue("services")
            request.getfixturevalue("hud_env").set(HUD_API_KEY="k")
            async with SubprocessRuntime(path)(Task(env=fixture, id="serve")) as served:
                host_on_runtime(services, served.url)

                async def released() -> bool:
                    created = len(services.requests("runtime", "POST", SESSIONS))
                    deleted = len(services.requests("runtime", "DELETE", SESSION))
                    return created == deleted > 0

                yield HUDRuntime(), released
        case "DockerRuntime":
            label = uuid.uuid4().hex
            image = request.getfixturevalue("fixture_image")(fixture)
            limits = RuntimeConfig.model_validate({"limits": {"startup_timeout_s": startup}})
            inner = DockerRuntime(
                image, run_args=["--label", f"hud-test={label}"], runtime_config=limits
            )
            volumes: set[str] = set()

            async def docker(*args: str) -> tuple[int, str]:
                process = await asyncio.create_subprocess_exec(
                    "docker",
                    *args,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.DEVNULL,
                )
                out, _ = await process.communicate()
                assert process.returncode is not None
                return process.returncode, out.decode()

            @asynccontextmanager
            async def recorded(task: Task) -> AsyncGenerator[Runtime]:
                async with inner(task) as placed_runtime:
                    assert isinstance(placed_runtime, DockerEndpoint)
                    code, mounts = await docker(
                        "inspect",
                        "--format",
                        "{{range .Mounts}}{{.Name}} {{end}}",
                        placed_runtime.container,
                    )
                    assert code == 0 and mounts.split(), placed_runtime.container
                    volumes.update(mounts.split())
                    yield placed_runtime

            # A cancelled rollout's teardown finishes in the background, removing the
            # container before its session volume; the test must outlast both.
            async def removed() -> bool:
                _, listed = await docker(
                    "ps", "--all", "--quiet", "--filter", f"label=hud-test={label}"
                )
                if listed.split():
                    return False
                for volume in volumes:
                    code, _ = await docker("volume", "inspect", volume)
                    if code == 0:
                        return False
                return True

            yield recorded, removed
        case "ModalRuntime":
            import modal

            request.getfixturevalue("modal_account")
            image = request.getfixturevalue("modal_image")(fixture)
            limits = RuntimeConfig.model_validate({"limits": {"startup_timeout_s": startup}})
            inner = ModalRuntime(image=image, workdir="/hud", runtime_config=limits)
            sandboxes = []

            @asynccontextmanager
            async def recorded(task: Task) -> AsyncGenerator[Runtime]:
                async with inner(task) as placed_runtime:
                    sandboxes.append(placed_runtime.params["instance_id"])
                    yield placed_runtime

            async def terminated() -> bool:
                for instance in sandboxes:
                    sandbox = await modal.Sandbox.from_id.aio(instance)
                    if await sandbox.poll.aio() is None:
                        return False
                return True

            yield recorded, terminated


# Over a WebSocket's default 1 MiB message limit, and under the 16 MiB frame limit.
LARGE = "x" * (4 * 1024 * 1024)


@pytest.mark.parametrize("runtime", EVERY_RUNTIME)
@pytest.mark.parametrize(
    ("fixture", "template", "word", "reward", "status"),
    [
        pytest.param("echo", "echo", "tangerine", 1.0, "completed", id="a-row-runs-to-its-grade"),
        pytest.param("echo", "echo", LARGE, 1.0, "completed", id="a-frame-of-megabytes-crosses"),
        pytest.param("faulty", "start_raises", None, 0.0, "error", id="a-start-that-fails"),
    ],
)
async def test_a_runtime_places_the_env_carries_its_frames_and_leaves_nothing_behind(
    fixture: str,
    template: str,
    word: str | None,
    reward: float,
    status: str,
    runtime: str,
    request: pytest.FixtureRequest,
) -> None:
    row = Task(env=fixture, id=template, args={"word": word} if word else {})

    async with placed(runtime, fixture, request) as (provider, nothing_left):
        job = await row.run(ScriptedAgent(word or "x"), runtime=provider)
        await eventually(nothing_left)

    (trace,) = job.traces
    assert (trace.reward, trace.status) == (reward, status)


@pytest.mark.parametrize(
    "runtime", ["LocalRuntime", "SubprocessRuntime", "Shared", CONTAINER, MODAL]
)
@pytest.mark.parametrize(
    "fixture",
    ["broken", "stalls"],
    ids=["an-env-that-exits-at-once", "an-env-that-never-finishes-starting"],
)
async def test_an_env_that_cannot_serve_fails_the_run_before_the_prompt_and_is_cleaned_up(
    fixture: str, runtime: str, request: pytest.FixtureRequest
) -> None:
    async with placed(runtime, fixture, request, startup=5) as (provider, nothing_left):
        job = await Task(env=fixture, id="never").run(ScriptedAgent("x"), runtime=provider)
        await eventually(nothing_left)

    (trace,) = job.traces
    assert (trace.status, trace.prompt_text, job.errors) == ("error", None, [trace])
    assert trace.error is not None
    assert trace.error.startswith(("[provisioning] ", "[starting task] "))


@pytest.mark.parametrize("runtime", EVERY_RUNTIME)
async def test_cancelling_a_rollout_propagates_and_leaves_nothing_behind(
    runtime: str, request: pytest.FixtureRequest
) -> None:
    prompted = asyncio.Event()

    class Waits(Agent):
        async def __call__(self, run: Run) -> None:
            prompted.set()
            await asyncio.sleep(3600)

    async with placed(runtime, "echo", request) as (provider, nothing_left):
        row = Task(env="echo", id="echo", args={"word": "tangerine"})
        pending = asyncio.create_task(row.run(Waits(), runtime=provider))
        await asyncio.wait_for(prompted.wait(), timeout=90)
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending

        await eventually(nothing_left, within=10)


@pytest.mark.parametrize("runtime", [CONTAINER, MODAL])
@pytest.mark.parametrize(("link", "reward"), [(False, 1.0), (True, 0.0)], ids=["files", "a-link"])
async def test_a_verifier_in_its_own_sandbox_receives_the_actors_session_files(
    link: bool, reward: float, runtime: str, request: pytest.FixtureRequest
) -> None:
    """The verifier's own limits give it its own sandbox, so the actor's files must move;
    a session holding anything but files and directories does not."""
    row = Task.model_validate(
        {
            "env": "sessions",
            "id": "solve",
            "args": {"link": link},
            "verifier": {
                "env": "sessions",
                "id": "verify",
                "runtime_config": {"limits": {"startup_timeout_s": 600}},
            },
        }
    )

    async with placed(runtime, "sessions", request) as (provider, nothing_left):
        job = await row.run(ScriptedAgent("secret"), runtime=provider)
        await eventually(nothing_left)

    (trace,) = job.traces
    assert trace.reward == reward
    assert ("runtime session contains" in (trace.error or "")) is link


@pytest.mark.parametrize("runtime", [DockerRuntime(), HUDRuntime()], ids=["docker", "hud"])
async def test_runtimes_that_cannot_mount_data_refuse_it(runtime: Provider) -> None:
    """A task that declares data never runs on a placement without it."""
    task = Task(
        env="echo",
        id="echo",
        runtime_config=RuntimeConfig(
            data=RuntimeData(files=[RuntimeDataFile(file_id=uuid.uuid4(), path="rows.csv")]),
        ),
    )

    with pytest.raises(ValueError, match="data"):
        async with runtime(task):
            pass
