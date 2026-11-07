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
import contextlib
import os
import socket
import tarfile
import uuid
from collections import Counter
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

import pytest

from hud.agents.base import Agent
from hud.eval import (
    DockerRuntime,
    HUDRuntime,
    LocalRuntime,
    ModalRuntime,
    Runtime,
    RuntimeConfig,
    Shared,
    SubprocessRuntime,
    Task,
    rollout,
)
from hud.eval.runtime import RootScope, RuntimeUnsupportedError, WorkspaceScope
from tests.conftest import eventually
from tests.eval.conftest import SESSION, SESSIONS, gone, host_on_runtime
from tests.fixtures.agents import ScriptedAgent
from tests.fixtures.envs import source

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Awaitable, Callable
    from pathlib import Path

    from hud.eval import Provider, Run
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
    assert ("runtime archive contains" in (run.trace.error or "")) is link


# ─── the filesystem interface: what a provisioned box lets the runner do with it ───

#: Providers that provision a box of their own and so own its filesystem operations.
BOXED = [CONTAINER, MODAL]
#: Operations a provider declines; the suite checks the typed refusal in their place.
DECLINED = {("ModalRuntime", "root checkpoint")}


@pytest.mark.parametrize("runtime", EVERY_RUNTIME)
async def test_a_runtime_exposes_filesystem_operations_or_declines_them_by_type(
    runtime: str, request: pytest.FixtureRequest
) -> None:
    boxed = runtime in {"DockerRuntime", "ModalRuntime"}

    async with placed(runtime, "echo", request) as (provider, nothing_left):
        scope = provider if isinstance(provider, Shared) else contextlib.nullcontext()
        async with scope, provider(Task(env="echo", id="echo")) as box:
            if boxed:
                result = await box.exec(["sh", "-c", "echo out; echo err >&2; exit 3"])
                assert (result.returncode, result.stdout, result.stderr) == (3, "out\n", "err\n")
            else:
                with pytest.raises(RuntimeUnsupportedError) as declined:
                    await box.exec(["true"])
                assert declined.value.operation == "exec"
        await eventually(nothing_left)


@pytest.mark.parametrize("runtime", BOXED)
async def test_collect_copies_files_and_directories_out_of_the_box(
    runtime: str, request: pytest.FixtureRequest, tmp_path: Path
) -> None:
    row = Task(env="box", id="write", args={"path": "/work/out/answer.txt", "text": "forty-two"})

    async with placed(runtime, "box", request) as (provider, nothing_left):
        async with provider(row) as box:
            run = await rollout(row, ScriptedAgent("forty-two"), runtime=box)
            await box.exec(["sh", "-c", "echo extra > /work/extra.txt; ln -s /etc /work/out/up"])
            with pytest.raises(tarfile.FilterError):
                await box.collect(["/work/out"], tmp_path / "linked")
            await box.exec(["rm", "/work/out/up"])
            copies = await box.collect(["/work/out", "/work/extra.txt"], tmp_path / "got")
        await eventually(nothing_left)

    assert run.reward == 1.0
    assert copies == [tmp_path / "got" / "out", tmp_path / "got" / "extra.txt"]
    assert (tmp_path / "got" / "out" / "answer.txt").read_text() == "forty-two"
    assert (tmp_path / "got" / "extra.txt").read_text() == "extra\n"


@pytest.mark.parametrize("runtime", BOXED)
async def test_a_workspace_checkpoint_restores_into_another_box(
    runtime: str, request: pytest.FixtureRequest
) -> None:
    writer = Task(env="box", id="write", args={"path": "/work/notes/a.txt", "text": "kept"})
    reader = Task(env="box", id="read", args={"path": "/work/notes/a.txt"})

    async with placed(runtime, "box", request) as (provider, nothing_left):
        async with provider(writer) as first:
            await rollout(writer, ScriptedAgent("kept"), runtime=first)
            with pytest.raises(RuntimeError, match="not a directory"):
                async with first.checkpoint(WorkspaceScope("/work/missing")):
                    pass
            async with (
                first.checkpoint(WorkspaceScope("/work/notes")) as notes,
                provider(reader) as second,
            ):
                await second.restore(notes, WorkspaceScope("/work/notes"))
                run = await rollout(reader, ScriptedAgent("kept"), runtime=second)
        await eventually(nothing_left)

    assert run.reward == 1.0


@pytest.mark.parametrize("runtime", BOXED)
async def test_a_root_checkpoint_boots_a_new_box_with_the_files(
    runtime: str, request: pytest.FixtureRequest
) -> None:
    writer = Task(env="box", id="write", args={"path": "/opt/mark.txt", "text": "stamped"})
    reader = Task(env="box", id="read", args={"path": "/opt/mark.txt"})

    async with placed(runtime, "box", request) as (provider, nothing_left):
        async with provider(writer) as first:
            await rollout(writer, ScriptedAgent("stamped"), runtime=first)
            if (runtime, "root checkpoint") in DECLINED:
                with pytest.raises(RuntimeUnsupportedError) as declined:
                    async with first.checkpoint(RootScope()):
                        pass
                assert declined.value.operation == "checkpoint the root filesystem"
                return
            async with first.checkpoint(RootScope()) as image:
                with pytest.raises(RuntimeUnsupportedError):
                    await first.restore(image, RootScope())
                restored = reader.model_copy(
                    update={"runtime_config": RuntimeConfig(image=image.ref)}
                )
                run = await restored.run(ScriptedAgent("stamped"), runtime=provider)
        await eventually(nothing_left)

    (graded,) = run.runs
    assert graded.reward == 1.0


@pytest.mark.parametrize("runtime", [CONTAINER])
async def test_a_box_past_its_memory_limit_dies_and_the_run_names_the_reason(
    runtime: str, request: pytest.FixtureRequest
) -> None:
    row = Task(
        env="box",
        id="hog",
        args={"mb": 1024},
        runtime_config=RuntimeConfig.model_validate({"resources": {"memory_mb": 256}}),
    )

    async with placed(runtime, "box", request) as (provider, nothing_left):
        job = await row.run(ScriptedAgent("x"), runtime=provider)
        await eventually(nothing_left)

    (run,) = job.runs
    assert run.trace.status == "error"
    assert "exit code 137" in (run.trace.error or "")
