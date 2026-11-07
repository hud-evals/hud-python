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
import json
import os
import socket
import tarfile
import textwrap
import uuid
from collections import Counter
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

import pytest

from hud import Task
from hud.agents.base import Agent
from hud.eval.run import rollout
from hud.runtime import (
    ComposeProject,
    DockerRuntime,
    HUDRuntime,
    LocalRuntime,
    ModalRuntime,
    RootScope,
    Runtime,
    RuntimeConfig,
    RuntimeGPU,
    RuntimeResources,
    RuntimeUnsupportedError,
    Shared,
    SubprocessRuntime,
    WorkspaceScope,
)
from hud.runtime.modal import ModalEndpoint
from tests.conftest import eventually
from tests.fixtures.agents import ScriptedAgent
from tests.fixtures.envs import source
from tests.runtime.conftest import SESSION, SESSIONS, gone, host_on_runtime

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Awaitable, Callable
    from pathlib import Path

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
            request.getfixturevalue("modal_account")
            image = request.getfixturevalue("modal_image")(fixture)
            limits = RuntimeConfig.model_validate({"limits": {"startup_timeout_s": startup}})
            yield modal_placement(ModalRuntime(image=image, workdir="/hud", runtime_config=limits))


def modal_placement(inner: ModalRuntime) -> tuple[Provider, Callable[[], Awaitable[bool]]]:
    """``inner`` recording every sandbox it places, and whether all of them have exited."""
    import modal

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

    return recorded, terminated


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
    # A provider that starts a box of its own names the failure; the others cannot.
    classified = ("container_start_failed", "user") if runtime in BOXED_NAMES else (None, None)
    assert (trace.error_code, trace.error_source) == classified


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
    assert ("runtime archive contains" in (trace.error or "")) is link


# Providers that provision a box of their own and so own its filesystem operations.
BOXED = [CONTAINER, MODAL]
BOXED_NAMES = {"DockerRuntime", "ModalRuntime"}


@pytest.mark.parametrize("runtime", EVERY_RUNTIME)
async def test_a_runtime_exposes_filesystem_operations_or_declines_them_by_type(
    runtime: str, request: pytest.FixtureRequest
) -> None:
    boxed = runtime in BOXED_NAMES

    async with placed(runtime, "echo", request) as (provider, nothing_left):
        scope = provider if isinstance(provider, Shared) else contextlib.nullcontext()
        async with scope, provider(Task(env="echo", id="echo")) as box:
            if boxed:
                result = await box.exec(["sh", "-c", "echo out; echo err >&2; exit 3"])
                assert (result.returncode, result.stdout, result.stderr) == (3, "out\n", "err\n")
                with pytest.raises(TimeoutError):
                    await box.exec(["sh", "-c", "sleep 60"], max_wait=1)
                # A command past max_wait is gone from the box, children included.
                left = await box.exec(
                    [
                        "sh",
                        "-c",
                        "for p in /proc/[0-9]*; do tr '\\0' ' ' < $p/cmdline 2>/dev/null; echo;"
                        " done | grep -c '^sleep 60 $' || true",
                    ]
                )
                assert left.stdout.strip() == "0", left.stdout
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
            trace = await rollout(row, ScriptedAgent("forty-two"), runtime=box)
            await box.exec(["sh", "-c", "echo extra > /work/extra.txt; ln -s /etc /work/out/up"])
            with pytest.raises(tarfile.FilterError):
                await box.collect(["/work/out"], tmp_path / "linked")
            await box.exec(["rm", "/work/out/up"])
            copies = await box.collect(["/work/out", "/work/extra.txt"], tmp_path / "got")
        await eventually(nothing_left)

    assert trace.reward == 1.0
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
                trace = await rollout(reader, ScriptedAgent("kept"), runtime=second)
        await eventually(nothing_left)

    assert trace.reward == 1.0


@pytest.mark.parametrize("runtime", BOXED)
async def test_a_root_checkpoint_boots_a_new_box_with_the_files(
    runtime: str, request: pytest.FixtureRequest
) -> None:
    writer = Task(env="box", id="write", args={"path": "/opt/mark.txt", "text": "stamped"})
    reader = Task(env="box", id="read", args={"path": "/opt/mark.txt"})

    async with placed(runtime, "box", request) as (provider, nothing_left):
        async with provider(writer) as first:
            await rollout(writer, ScriptedAgent("stamped"), runtime=first)
            if runtime == "ModalRuntime":  # its default runtime declines root checkpoints
                with pytest.raises(RuntimeUnsupportedError) as declined:
                    async with first.checkpoint(RootScope()):
                        pass
                assert declined.value.operation == "checkpoint the root filesystem"
                return
            async with first.checkpoint(RootScope()) as image:
                with pytest.raises(ValueError, match="starting a new box"):
                    await first.restore(image, RootScope())
                restored = reader.model_copy(
                    update={"runtime_config": RuntimeConfig(image=image.ref)}
                )
                job = await restored.run(ScriptedAgent("stamped"), runtime=provider)
        await eventually(nothing_left)

    (graded,) = job.traces
    assert graded.reward == 1.0


@pytest.mark.parametrize("runtime", BOXED)
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

    (trace,) = job.traces
    assert trace.status == "error"
    assert "exit code 137" in (trace.error or "")


@asynccontextmanager
async def placed_compose(
    runtime: str, request: pytest.FixtureRequest
) -> AsyncGenerator[tuple[Provider, Callable[[], Awaitable[bool]]]]:
    """A provider for Compose rows written by ``compose_row``, and its cleanliness check."""
    limits = RuntimeConfig.model_validate({"limits": {"startup_timeout_s": 120}})
    match runtime:
        case "DockerRuntime":
            working_dir = str(request.getfixturevalue("tmp_path"))

            async def removed() -> bool:
                listing = await asyncio.create_subprocess_exec(
                    "docker",
                    "ps",
                    "--all",
                    "--quiet",
                    "--filter",
                    f"label=com.docker.compose.project.working_dir={working_dir}",
                    stdout=asyncio.subprocess.PIPE,
                )
                listed, _ = await listing.communicate()
                return listed.split() == []

            yield DockerRuntime(runtime_config=limits), removed
        case "ModalRuntime":
            request.getfixturevalue("modal_account")
            yield modal_placement(ModalRuntime(workdir="/hud", runtime_config=limits))
        case _:
            raise AssertionError(runtime)


async def compose_row(
    runtime: str, request: pytest.FixtureRequest, services: str, *, docker: bool = False
) -> Task:
    """A ``box`` task whose environment is a Compose project of ``main`` and ``services``.

    ``main`` serves the box fixture; ``services`` is YAML for the services beside it.
    """
    match runtime:
        case "DockerRuntime":
            main = request.getfixturevalue("fixture_image")("box")
            source = "/hud/box.py"
        case "ModalRuntime":
            import modal

            request.getfixturevalue("modal_account")
            image = request.getfixturevalue("modal_image")("box")
            await image.build.aio(
                app=await modal.App.lookup.aio("hud-envs", create_if_missing=True)
            )
            main = f"modal://{image.object_id}"
            source = "/hud/env.py"
        case _:
            raise AssertionError(runtime)
    path = request.getfixturevalue("tmp_path") / "compose.yaml"
    path.write_text(
        "services:\n"
        f"  main:\n    image: {main}\n"
        f"    command: hud serve {source} --host 0.0.0.0 --port 8765\n"
        + "\n".join("  " + line for line in textwrap.dedent(services).strip("\n").splitlines())
        + "\n"
    )
    return Task(
        env="box",
        id="write",
        args={"path": "/work/a.txt", "text": "composed"},
        runtime_config=RuntimeConfig(
            compose=ComposeProject(document=path, service_access=docker or None),
            resources=RuntimeResources(memory_mb=1024) if docker else None,
        ),
    )


@pytest.mark.parametrize("runtime", BOXED)
async def test_sibling_services_are_reached_by_exec_and_collect(
    runtime: str, request: pytest.FixtureRequest, tmp_path: Path
) -> None:
    """``main`` starts once ``side`` is healthy, ``after`` once ``main`` is, and every
    service is reachable by name."""
    row = await compose_row(
        runtime,
        request,
        """
          depends_on:
            side:
              condition: service_healthy
          healthcheck:
            test:
              - CMD
              - python3
              - -c
              - import socket; socket.create_connection(('127.0.0.1', 8765))
            interval: 1s
            retries: 60
        side:
          image: python:3.12-slim
          command: sh -c "sleep 1; touch /tmp/ready; sleep infinity"
          healthcheck:
            test: ["CMD", "test", "-f", "/tmp/ready"]
            interval: 1s
            retries: 30
        after:
          image: python:3.12-slim
          command: sleep infinity
          depends_on:
            main:
              condition: service_healthy
        """,
    )

    async with placed_compose(runtime, request) as (provider, nothing_left):
        async with provider(row) as box:
            trace = await rollout(row, ScriptedAgent("composed"), runtime=box)
            noted = await box.exec(["sh", "-c", "hostname > /tmp/who; echo noted"], service="side")
            later = await box.exec(["true"], service="after")
            (tmp_path / "in").mkdir()
            (tmp_path / "in" / "note.txt").write_text("delivered")
            await box.put(tmp_path / "in", "/srv/in", service="side")
            delivered = await box.exec(["cat", "/srv/in/note.txt"], service="side")
            with pytest.raises(ValueError, match="'nobody'"):
                await box.exec(["true"], service="nobody")
            (who,) = await box.collect(["/tmp/who"], tmp_path / "side", service="side")
            (mine,) = await box.collect(["/work/a.txt"], tmp_path / "main")
        await eventually(nothing_left)

    assert trace.reward == 1.0
    assert noted.stdout == "noted\n"
    assert later.returncode == 0
    assert delivered.stdout == "delivered"
    assert who.read_text().strip() and who.read_text() != mine.read_text()
    assert mine.read_text() == "composed"


@pytest.mark.parametrize("runtime", BOXED)
async def test_a_project_that_cannot_come_up_fails_the_run_as_a_compose_failure(
    runtime: str, request: pytest.FixtureRequest
) -> None:
    row = await compose_row(
        runtime,
        request,
        """
          depends_on:
            side:
              condition: service_healthy
        side:
          image: python:3.12-slim
          command: sleep infinity
          healthcheck:
            test: ["CMD", "false"]
            interval: 1s
            retries: 2
        """,
    )

    async with placed_compose(runtime, request) as (provider, nothing_left):
        job = await row.run(ScriptedAgent("composed"), runtime=provider)
        await eventually(nothing_left)

    (trace,) = job.traces
    assert trace.status == "error"
    assert (trace.error_code, trace.error_source) == ("compose_execution_failed", "user")


@pytest.mark.parametrize("runtime", BOXED)
async def test_main_gets_a_docker_daemon_of_its_own(
    runtime: str, request: pytest.FixtureRequest
) -> None:
    row = await compose_row(runtime, request, "", docker=True)
    probe = (
        "import json, os, urllib.request\n"
        "base = os.environ['DOCKER_HOST'].replace('tcp://', 'http://')\n"
        "version = json.load(urllib.request.urlopen(base + '/version'))\n"
        "containers = json.load(urllib.request.urlopen(base + '/containers/json?all=1'))\n"
        "print(json.dumps([version['Version'], containers]))\n"
    )

    reader = Task(env="box", id="read", args={"path": "/work/a.txt"})

    async with placed_compose(runtime, request) as (provider, nothing_left):
        async with provider(row) as box:
            trace = await rollout(row, ScriptedAgent("composed"), runtime=box)
            seen = await box.exec(["python3", "-c", probe])
            # A box that hosts a daemon runs where root checkpoints are faithful.
            async with box.checkpoint(RootScope()) as image:
                restored = reader.model_copy(
                    update={"runtime_config": RuntimeConfig(image=image.ref)}
                )
                read_job = await restored.run(ScriptedAgent("composed"), runtime=provider)
        await eventually(nothing_left)

    assert trace.reward == 1.0
    assert seen.returncode == 0, seen.stderr
    version, containers = json.loads(seen.stdout)
    assert version.startswith("28.") and containers == []
    (graded,) = read_job.traces
    assert graded.reward == 1.0


@pytest.mark.parametrize("runtime", [MODAL])
async def test_a_gpu_request_attaches_a_gpu_to_the_environment(
    runtime: str, request: pytest.FixtureRequest
) -> None:
    """The Docker lane has no GPU; Docker's ``--gpus`` is the host's own capability."""
    row = Task(
        env="box",
        id="write",
        args={"path": "/work/a.txt", "text": "accelerated"},
        runtime_config=RuntimeConfig.model_validate({"resources": {"gpu": {"type": "T4"}}}),
    )

    async with placed(runtime, "box", request) as (provider, nothing_left):
        async with provider(row) as box:
            trace = await rollout(row, ScriptedAgent("accelerated"), runtime=box)
            listed = await box.exec(["nvidia-smi", "-L"])
        await eventually(nothing_left)

    assert trace.reward == 1.0
    assert "T4" in listed.stdout, listed.stderr


@pytest.mark.parametrize("runtime", [MODAL])
async def test_a_sidecar_with_a_restart_policy_is_restarted(
    runtime: str, request: pytest.FixtureRequest
) -> None:
    row = await compose_row(
        runtime,
        request,
        """
        side:
          image: python:3.12-slim
          command: sh -c "sleep 3; exit 1"
          restart: always
        """,
    )

    async with placed_compose(runtime, request) as (provider, nothing_left):
        async with provider(row) as box:
            assert isinstance(box, ModalEndpoint)
            first = box.sidecars["side"].object_id

            async def replaced() -> bool:
                return box.sidecars["side"].object_id != first

            await eventually(replaced, within=30)
            alive = await box.exec(["true"], service="side")
        await eventually(nothing_left)

    assert alive.returncode == 0


@pytest.mark.parametrize(
    ("services", "docker", "gpu", "memory", "reason"),
    [
        pytest.param(
            """
            worker:
              image: python:3.12-slim
              deploy:
                resources:
                  reservations:
                    devices: [{capabilities: [gpu]}]
            """,
            False,
            False,
            None,
            "sidecars are CPU-only",
            id="a-gpu-on-a-sidecar",
        ),
        pytest.param("", True, True, 1024, "has no GPUs", id="a-gpu-beside-a-docker-daemon"),
        pytest.param(
            "", True, False, None, "needs resources.memory_mb", id="a-docker-daemon-unsized"
        ),
        pytest.param(
            """
            worker:
              image: python:3.12-slim
            """,
            True,
            False,
            1024,
            "needs mem_limit on Compose service 'worker'",
            id="a-sidecar-unsized-beside-a-docker-daemon",
        ),
        pytest.param(
            """
            worker:
              build: ./worker
            """,
            False,
            False,
            None,
            "pre-built images only",
            id="a-service-to-build",
        ),
        pytest.param(
            """
            worker:
              image: python:3.12-slim
              network_mode: service:main
            """,
            False,
            False,
            None,
            "network_mode",
            id="a-shared-network-namespace",
        ),
        pytest.param(
            """
            worker:
              image: python:3.12-slim
              depends_on:
                main:
                  condition: service_completed_successfully
            """,
            False,
            False,
            None,
            "serves for the whole rollout",
            id="a-wait-for-main-to-complete",
        ),
    ],
)
async def test_modal_refuses_at_launch_what_a_sandbox_cannot_do(
    services: str, docker: bool, gpu: bool, memory: int | None, reason: str, tmp_path: Path
) -> None:
    pytest.importorskip("modal")
    path = tmp_path / "compose.yaml"
    path.write_text(
        "services:\n  main:\n    image: python:3.12-slim\n"
        + "\n".join("  " + line for line in textwrap.dedent(services).strip("\n").splitlines())
        + "\n"
    )
    row = Task(
        env="box",
        id="write",
        runtime_config=RuntimeConfig(
            compose=ComposeProject(document=path, service_access=docker or None),
            resources=RuntimeResources(
                gpu=RuntimeGPU(type="T4") if gpu else None, memory_mb=memory
            ),
        ),
    )

    with pytest.raises(ValueError, match=reason):
        async with ModalRuntime()(row):
            pytest.fail("the sandbox should not launch")
