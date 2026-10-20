"""What every runtime owes a rollout: it places the environment, carries the control
channel's frames to it and tears it down, whether the run ends, the env cannot serve,
or the rollout is cancelled; and it refuses a row it cannot provision before asking
anything of anyone.

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
from typing import TYPE_CHECKING, Any

import pytest

from hud.agents.base import Agent
from hud.eval import (
    DaytonaRuntime,
    DockerRuntime,
    HUDRuntime,
    LocalRuntime,
    ModalRuntime,
    Runtime,
    RuntimeConfig,
    Shared,
    SubprocessRuntime,
    Task,
)
from tests.conftest import CLOSED_PORT, eventually
from tests.eval.conftest import SESSION, SESSIONS, gone, host_on_runtime
from tests.fixtures.agents import ScriptedAgent
from tests.fixtures.envs import source

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Awaitable, Callable
    from pathlib import Path

    from hud.eval import Provider, Run
    from tests.conftest import HudEnv
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
) -> AsyncIterator[tuple[Provider, Callable[[], Awaitable[bool]]]]:
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

            async def removed() -> bool:
                listing = await asyncio.create_subprocess_exec(
                    *("docker", "ps", "--all", "--quiet", "--filter", f"label=hud-test={label}"),
                    stdout=asyncio.subprocess.PIPE,
                )
                listed, _ = await listing.communicate()
                return listed.split() == []

            labelled = ["--label", f"hud-test={label}"]
            yield DockerRuntime(image, run_args=labelled, runtime_config=limits), removed
        case "ModalRuntime":
            import modal

            request.getfixturevalue("modal_account")
            image = request.getfixturevalue("modal_image")(fixture)
            limits = RuntimeConfig.model_validate({"limits": {"startup_timeout_s": startup}})
            inner = ModalRuntime(image=image, workdir="/hud", runtime_config=limits)
            sandboxes = []

            @asynccontextmanager
            async def recorded(task: Task) -> AsyncIterator[Runtime]:
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


#: Over a WebSocket's default 1 MiB message limit, and under the 16 MiB frame limit.
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

    (run,) = job.runs
    assert (run.reward, run.trace.status) == (reward, status)


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

    (run,) = job.runs
    assert (run.trace.status, run.prompt, job.errors) == ("error", None, [run])
    assert (run.trace.error or "").startswith(("[provisioning] ", "[starting task] "))


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

    (run,) = job.runs
    assert run.reward == reward
    assert ("runtime session contains" in (run.trace.error or "")) is link


# ─── what each runtime refuses to provision ────────────────────────────


@pytest.mark.parametrize("runtime", [LocalRuntime, SubprocessRuntime])
async def test_a_local_runtime_refuses_a_row_that_asks_for_its_own_runtime(
    runtime: type[LocalRuntime | SubprocessRuntime], hud_env: HudEnv, tmp_path: Path
) -> None:
    lifecycle = tmp_path / "lifecycle.log"
    hud_env.set(HUD_FIXTURE_LIFECYCLE=str(lifecycle))
    row = Task(env="echo", id="echo", runtime_config={"image": "lab"})

    job = await row.run(ScriptedAgent("x"), runtime=runtime(source("echo")))

    (run,) = job.runs
    assert (run.trace.status, run.prompt, lifecycle.exists()) == ("error", None, False)
    assert (run.trace.error or "").startswith("[provisioning] ")
    assert "runtime_config" in (run.trace.error or "")


COMPOSE = {"compose": {"document": "compose.yaml"}}
SIMPLE = "services:\n  main: {image: lab}\n"


@pytest.mark.parametrize(
    ("runtime_config", "compose", "run_args", "message"),
    [
        pytest.param(
            {"image": "lab:1", "resources": {"gpu": {"type": "H100"}}},
            "",
            [],
            "cannot select GPUs by type",
            id="a-gpu-type",
        ),
        pytest.param(
            COMPOSE,
            'services:\n  main: {image: lab, network_mode: "service:gateway"}\n'
            "  gateway: {image: gateway}\n",
            ["--privileged"],
            "run_args apply only to image environments",
            id="run-args-for-a-compose-project",
        ),
        pytest.param(
            COMPOSE,
            "services:\n  main: {image: lab, network_mode: 'service:relay'}\n"
            "  relay: {image: relay, network_mode: 'service:main'}\n",
            [],
            "network_mode service cycle includes 'main'",
            id="a-network-cycle",
        ),
        pytest.param(
            COMPOSE,
            "include: [other.yaml]\n" + SIMPLE,
            [],
            "does not support Compose include or extends",
            id="a-compose-include",
        ),
        pytest.param(
            COMPOSE,
            "services:\n  main: {image: $HOST_ONLY}\n",
            [],
            "variable 'HOST_ONLY' is not set by the project .env",
            id="a-variable-only-the-host-sets",
        ),
    ],
)
async def test_docker_refuses_what_it_cannot_provision_before_running_anything(
    runtime_config: dict[str, Any],
    compose: str,
    run_args: list[str],
    message: str,
    hud_env: HudEnv,
    tmp_path: Path,
) -> None:
    hud_env.set(PATH="/nonexistent")
    (tmp_path / ".env").write_text("")
    (tmp_path / "compose.yaml").write_text(compose)
    row = Task.model_validate(
        {"env": "echo", "id": "echo", "runtime_config": runtime_config},
        context={"base_path": tmp_path},
    )

    job = await row.run(ScriptedAgent("x"), runtime=DockerRuntime(run_args=run_args))

    (run,) = job.runs
    assert (run.trace.status, run.prompt) == ("error", None)
    assert message in (run.trace.error or "")


@pytest.mark.parametrize(
    ("runtime_config", "message"),
    [
        pytest.param(
            {"resources": {"gpu": {"type": ["A100", "H100"]}}},
            "alternative GPU types",
            id="alternative-gpu-types",
        ),
        pytest.param(
            {"resources": {"os": "windows"}},
            "does not support runtime_config.resources.os",
            id="an-os",
        ),
        pytest.param(
            {**COMPOSE, "resources": {"gpu": {}}}, "cannot attach GPUs", id="a-gpu-for-compose"
        ),
    ],
)
async def test_modal_refuses_what_it_cannot_provision_before_creating_a_sandbox(
    runtime_config: dict[str, Any], message: str, hud_env: HudEnv, tmp_path: Path
) -> None:
    pytest.importorskip("modal", reason="needs the modal extra")
    hud_env.set(MODAL_SERVER_URL=CLOSED_PORT)
    (tmp_path / ".env").write_text("")
    (tmp_path / "compose.yaml").write_text(SIMPLE)
    row = Task.model_validate(
        {"env": "echo", "id": "echo", "runtime_config": runtime_config},
        context={"base_path": tmp_path},
    )

    job = await row.run(ScriptedAgent("x"), runtime=ModalRuntime("hud-lab"))

    (run,) = job.runs
    assert (run.trace.status, run.prompt) == ("error", None)
    assert message in (run.trace.error or "")


async def test_modal_refuses_sandbox_secrets_for_a_compose_project(
    hud_env: HudEnv, tmp_path: Path
) -> None:
    modal = pytest.importorskip("modal", reason="needs the modal extra")
    hud_env.set(MODAL_SERVER_URL=CLOSED_PORT)
    (tmp_path / ".env").write_text("")
    (tmp_path / "compose.yaml").write_text(SIMPLE)
    row = Task.model_validate(
        {"env": "echo", "id": "echo", "runtime_config": COMPOSE}, context={"base_path": tmp_path}
    )
    runtime = ModalRuntime(sandbox_secrets=[modal.Secret.from_dict({"API": "x"})])

    job = await row.run(ScriptedAgent("x"), runtime=runtime)

    (run,) = job.runs
    assert (run.trace.status, run.prompt) == ("error", None)
    assert "sandbox secrets" in (run.trace.error or "")


@pytest.mark.parametrize(
    ("snapshot", "runtime_config", "message"),
    [
        pytest.param("snap", COMPOSE, "does not support runtime_config.compose", id="compose"),
        pytest.param(
            "snap",
            {"limits": {"run_timeout_s": 60}},
            "does not support runtime_config.run_timeout_s",
            id="a-run-limit",
        ),
        pytest.param(
            None,
            {"image": "lab", "resources": {"os": "linux"}},
            "does not support runtime_config.resources.os",
            id="an-os",
        ),
        pytest.param(
            None,
            {"image": "lab", "resources": {"cpu": 1.5}},
            "needs a whole number of CPUs, got 1.5",
            id="a-fraction-of-a-cpu",
        ),
        pytest.param(
            None, None, "requires snapshot_name or runtime_config.image", id="nothing-to-boot"
        ),
        pytest.param(
            "snap",
            {"resources": {"cpu": 2}},
            "cannot resize an already-built snapshot",
            id="resources-for-a-built-snapshot",
        ),
    ],
)
async def test_daytona_refuses_what_it_cannot_provision_before_creating_a_sandbox(
    snapshot: str | None,
    runtime_config: dict[str, Any] | None,
    message: str,
    hud_env: HudEnv,
    services: FakeServices,
    tmp_path: Path,
) -> None:
    pytest.importorskip("daytona", reason="needs the daytona extra")
    hud_env.set(DAYTONA_API_KEY="k", DAYTONA_API_URL=f"http://127.0.0.1:{services.port}/daytona")
    (tmp_path / ".env").write_text("")
    (tmp_path / "compose.yaml").write_text(SIMPLE)
    row = Task.model_validate(
        {"env": "echo", "id": "echo", "runtime_config": runtime_config},
        context={"base_path": tmp_path},
    )

    job = await row.run(ScriptedAgent("x"), runtime=DaytonaRuntime(snapshot))

    (run,) = job.runs
    assert (run.trace.status, run.prompt, services.requests("daytona")) == ("error", None, [])
    assert message in (run.trace.error or "")
