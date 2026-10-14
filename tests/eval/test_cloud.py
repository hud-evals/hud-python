"""Cloud runtimes: ModalRuntime and DaytonaRuntime place a row on a sandbox that serves it.

The SDKs are faked at the module boundary (``tests/harness/cloud.py``); every
sandbox they create connects to an environment served in this process, so each
placement below also runs a rollout whose reward proves the sandbox served it.
"""

from __future__ import annotations

import asyncio
import logging
import textwrap
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import pytest

from hud.eval import (
    ComposeProject,
    DaytonaRuntime,
    LocalRuntime,
    ModalRuntime,
    RuntimeConfig,
    RuntimeGPU,
    RuntimeLimits,
    RuntimeResources,
    Task,
    rollout,
)
from tests.eval.envs import actor, judge, lab, solve
from tests.harness import ScriptedAgent
from tests.harness.cloud import BuiltImage, Exec, FakeDaytona, FakeModal, aio

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path

    from hud import Environment
    from hud.eval import Provider

ADD = Task(env="lab", id="add", args={"a": 2, "b": 3})
NOWHERE = ("127.0.0.1", 1)


@asynccontextmanager
async def serving(envs: dict[str, Environment]) -> AsyncIterator[dict[str, tuple[str, int]]]:
    """Serve each environment here; yield the address of each."""
    async with AsyncExitStack() as stack:
        addresses = {}
        for key, env in envs.items():
            runtime = await stack.enter_async_context(
                LocalRuntime(env)(Task(env=env.name, id="serve"))
            )
            host, port = runtime.url.removeprefix("tcp://").rsplit(":", 1)
            addresses[key] = (host, int(port))
        yield addresses


async def place(provider: Provider, row: Task) -> float:
    """Place ``row`` and run its rollout there; return the reward."""
    async with provider(row) as runtime:
        run = await rollout(row, ScriptedAgent(solve), runtime=runtime)
    return run.reward


def called(calls: list[tuple[str, Any]], name: str) -> list[Any]:
    return [detail for call, detail in calls if call == name]


def configured(config: RuntimeConfig | None = None, **agent: Any) -> Task:
    return ADD.model_copy(update={"runtime_config": config, "agent_config": agent or None})


def install(
    cloud: str, monkeypatch: pytest.MonkeyPatch, address: tuple[str, int] = NOWHERE, **fake: Any
) -> Any:
    """Install the fake SDK for ``cloud``, its ``hud-lab`` sandbox served at ``address``."""
    if cloud == "modal":
        return FakeModal({"hud-lab": address}, **fake).install(monkeypatch)
    return FakeDaytona(address[1], **fake).install(monkeypatch)


def provider_for(cloud: str, options: dict[str, Any]) -> Provider:
    return ModalRuntime(**options) if cloud == "modal" else DaytonaRuntime(**options)


# The row's own settings win over the provider's; a published image outlives the agent.
MODAL: dict[str, tuple[dict[str, Any], Task, dict[str, Any]]] = {
    "row image, resources and limits replace the provider's": (
        {"runtime_config": RuntimeConfig(resources=RuntimeResources(cpu=2, memory_mb=4096))},
        configured(
            RuntimeConfig(
                image="registry.test/lab:1",
                resources=RuntimeResources(gpu=RuntimeGPU(type="A10G", count=2)),
                limits=RuntimeLimits(startup_timeout_s=30, run_timeout_s=120),
            )
        ),
        {"image": "registry.test/lab:1", "gpu": "A10G:2", "timeout": 120, "ready": 30},
    ),
    "a published image by name outlives the agent's budget": (
        {"image_name": "hud-lab"},
        configured(timeout_seconds=90.5),
        {"image": "hud-lab", "gpu": None, "timeout": 3691, "ready": 600},
    ),
    "a modal image id with any GPU": (
        {},
        configured(
            RuntimeConfig(image="modal://im-123", resources=RuntimeResources(gpu=RuntimeGPU()))
        ),
        {"image": "im-123", "gpu": "any", "timeout": 3600, "ready": 600},
    ),
    "the row's image beats the published name": (
        {"image_name": "hud-lab"},
        configured(RuntimeConfig(image="registry.test/lab:1")),
        {"image": "registry.test/lab:1", "gpu": None, "timeout": 3600, "ready": 600},
    ),
}


@pytest.mark.parametrize(("options", "row", "sandbox"), MODAL.values(), ids=MODAL.keys())
async def test_a_modal_sandbox_serves_the_row_with_the_settings_that_win(
    options: dict[str, Any], row: Task, sandbox: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    images = ("registry.test/lab:1", "hud-lab", "im-123")

    async with serving({"lab": lab()}) as addresses:
        modal = FakeModal({image: addresses["lab"] for image in images}).install(monkeypatch)
        reward = await place(ModalRuntime(**options), row)

    (create,) = called(modal.calls, "Sandbox.create")
    (ready,) = called(modal.calls, "sb-1.wait_until_ready")
    assert reward == 1.0
    assert {
        "image": create["image"],
        "gpu": create.get("gpu"),
        "timeout": create["timeout"],
        "ready": ready["timeout"],
    } == sandbox
    assert modal.calls[-1] == ("sb-1.terminate", {})


@dataclass(frozen=True)
class UserImage:
    """A ``modal.Image`` the caller defines; the runtime builds it once."""

    builds: list[str] = field(default_factory=list, hash=False, compare=False)

    @property
    def build(self) -> Any:
        async def build(*, app: str) -> None:
            self.builds.append(app)

        return aio(build)


async def test_a_caller_image_is_built_once_in_the_callers_app(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Stand-ins for the SDK's own Image and App objects.
    image: Any = UserImage()
    app: Any = "app:mine"

    async with serving({"lab": lab()}) as addresses:
        modal = FakeModal({image: addresses["lab"]}).install(monkeypatch)
        provider = ModalRuntime(image=image, app=app)
        rewards = [await place(provider, ADD) for _ in range(2)]

    assert rewards == [1.0, 1.0]
    assert image.builds == ["app:mine"]
    assert called(modal.calls, "App.lookup") == []


def compose_project(tmp_path: Path) -> ComposeProject:
    root = tmp_path / "project"
    (root / "recipe").mkdir(parents=True)
    (root / "recipe" / ".env").write_text("TAG=1\n")
    path = root / "recipe" / "compose.yaml"
    path.write_text(
        textwrap.dedent(
            """
            services:
              main:
                image: lab:${TAG}
                network_mode: service:gateway
              gateway:
                image: gateway
                healthcheck: {test: [CMD, "true"]}
            """
        )
    )
    return ComposeProject(document=path, root=root, service_access=True)


async def test_a_modal_compose_project_runs_main_confined_and_sized_then_comes_down(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = RuntimeConfig(
        compose=compose_project(tmp_path), resources=RuntimeResources(cpu=2, memory_mb=16384)
    )

    async with serving({"lab": lab()}) as addresses:
        modal = FakeModal({"docker:28.3.3-dind": addresses["lab"]}).install(monkeypatch)
        reward = await place(ModalRuntime(), configured(config))

    (create,) = called(modal.calls, "Sandbox.create")
    (override,) = [
        upload["/hud/override.json"]
        for upload in called(modal.calls, "sb-1.copy_from_local")
        if "/hud/override.json" in upload
    ]
    main = override["services"]["main"]
    commands = [" ".join(run["args"]) for run in called(modal.calls, "sb-1.exec")]
    assert reward == 1.0
    assert create["experimental_options"] == {"vm_runtime": True}
    assert "seccomp=/hud/docker-seccomp.json" in main["security_opt"]
    assert (main["cpus"], main["mem_limit"]) == (2.0, "16384m")
    assert commands[-1].endswith("down --volumes --remove-orphans")
    assert modal.calls[-1] == ("sb-1.terminate", {})


async def test_a_verifier_on_modal_gets_the_actors_session_archive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    verifier = Task(env="judge", id="verify", runtime_config=RuntimeConfig(image="judge"))
    row = Task(
        env="actor", id="solve", runtime_config=RuntimeConfig(image="actor"), verifier=verifier
    )

    async with serving({"actor": actor(), "judge": judge()}) as addresses:
        modal = FakeModal(addresses, execs=[Exec("if [ -d", stdout="1")]).install(monkeypatch)
        run = await rollout(row, ScriptedAgent("secret"), runtime=ModalRuntime())

    moves = [
        name
        for name, detail in modal.calls
        if name.endswith(("copy_to_local", "copy_from_local"))
        and "/runtime/session.tar.gz" in str(detail)
    ]
    assert run.reward == 1.0
    assert moves == ["sb-1.copy_to_local", "sb-2.copy_from_local"]


COMPOSE_DOC = RuntimeConfig.model_validate({"compose": {"document": {"services": {"main": {}}}}})
REFUSALS: dict[str, tuple[str, dict[str, Any], Any, str]] = {
    "modal: an app and an app name": (
        "modal",
        {"app": "app:mine", "app_name": "other"},
        None,
        "either app or app_name",
    ),
    "modal: no image to boot": ("modal", {}, None, "requires image="),
    "modal: alternative GPU types": (
        "modal",
        {"image_name": "hud-lab"},
        RuntimeConfig(resources=RuntimeResources(gpu=RuntimeGPU(type=["A100", "H100"]))),
        "alternative GPU types",
    ),
    "modal: an OS requirement": (
        "modal",
        {"image_name": "hud-lab"},
        RuntimeConfig(resources=RuntimeResources(os="windows")),
        "resources.os",
    ),
    "modal: a GPU for a Compose project": ("modal", {}, "compose+gpu", "cannot attach GPUs"),
    "modal: sandbox secrets for a Compose project": (
        "modal",
        {"sandbox_secrets": ["secret:api"]},
        "compose",
        "sandbox secrets require an image runtime",
    ),
    "modal: a Compose document that is not a local file": (
        "modal",
        {},
        COMPOSE_DOC,
        "compose as a local file path",
    ),
    "daytona: a Compose project": (
        "daytona",
        {"snapshot_name": "hud-lab"},
        COMPOSE_DOC,
        "does not support runtime_config.compose",
    ),
    "daytona: a run limit": (
        "daytona",
        {"snapshot_name": "hud-lab"},
        RuntimeConfig(limits=RuntimeLimits(run_timeout_s=60)),
        "does not support runtime_config.run_timeout_s",
    ),
    "daytona: a fractional CPU": (
        "daytona",
        {},
        RuntimeConfig(image="registry.test/lab:1", resources=RuntimeResources(cpu=1.5)),
        "whole number of CPUs, got 1.5",
    ),
    "daytona: resources for a prebuilt snapshot": (
        "daytona",
        {"snapshot_name": "hud-lab"},
        RuntimeConfig(resources=RuntimeResources(cpu=2)),
        "cannot resize an already-built snapshot",
    ),
    "daytona: no snapshot and no image": ("daytona", {}, None, "requires snapshot_name"),
    "daytona: a TPU": (
        "daytona",
        {"snapshot_name": "hud-lab"},
        RuntimeConfig(resources=RuntimeResources(tpu={"type": "v5e", "topology": "2x2"})),
        "resources.tpu",
    ),
}


@pytest.mark.parametrize(
    ("cloud", "options", "config", "message"), REFUSALS.values(), ids=REFUSALS.keys()
)
async def test_a_cloud_refuses_what_it_cannot_provision_before_creating_a_sandbox(
    cloud: str,
    options: dict[str, Any],
    config: Any,
    message: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = install(cloud, monkeypatch)
    if isinstance(config, str):
        gpu = RuntimeResources(gpu=RuntimeGPU()) if config == "compose+gpu" else None
        config = RuntimeConfig(compose=compose_project(tmp_path), resources=gpu)

    with pytest.raises(ValueError, match=message):
        async with provider_for(cloud, options)(configured(config)):
            pytest.fail("the acquisition should fail")

    assert [name for name, _ in fake.calls if name in {"Sandbox.create", "create"}] == []


COMPOSE_STARTS = {
    "a startup slower than its limit": (
        Exec("up --detach", hang=True),
        TimeoutError,
        "startup timed out after 1 seconds",
    ),
    "a startup that fails": (
        Exec("up --detach", returncode=1, stderr="no such image: gateway\n"),
        RuntimeError,
        "startup failed: no such image: gateway",
    ),
}


@pytest.mark.parametrize(
    ("startup", "error", "message"), COMPOSE_STARTS.values(), ids=COMPOSE_STARTS.keys()
)
async def test_a_modal_compose_project_that_does_not_start_terminates_its_sandbox(
    startup: Exec,
    error: type[Exception],
    message: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = RuntimeConfig(
        compose=compose_project(tmp_path), limits=RuntimeLimits(startup_timeout_s=1)
    )
    modal = FakeModal({"docker:28.3.3-dind": NOWHERE}, execs=[startup]).install(monkeypatch)

    with pytest.raises(error, match=message):
        async with ModalRuntime()(configured(config)):
            pytest.fail("the acquisition should fail")

    assert modal.calls[-1] == ("sb-1.terminate", {})


PUBLISHED = {"modal": {"image_name": "hud-lab"}, "daytona": {"snapshot_name": "hud-lab"}}


@pytest.mark.parametrize("cloud", PUBLISHED)
async def test_a_sandbox_streams_its_output_and_teardown_does_not_wait_on_the_stream(
    cloud: str, monkeypatch: pytest.MonkeyPatch, capfd: pytest.CaptureFixture[str]
) -> None:
    async with serving({"lab": lab()}) as addresses:
        fake = install(
            cloud,
            monkeypatch,
            addresses["lab"],
            output=("booted\n", "warned\n"),
            hold=asyncio.Event(),
        )
        reward = await place(provider_for(cloud, PUBLISHED[cloud]), ADD)

    out, err = capfd.readouterr()
    assert (reward, out, err) == (1.0, "booted\n", "warned\n")
    assert fake.calls[-1][0] in {"sb-1.terminate", "close"}


DAYTONA_SIZES = {
    "megabytes round up to whole gibibytes and GPU types are prefixed": (
        RuntimeResources(
            cpu=2, memory_mb=3000, storage_mb=1025, gpu=RuntimeGPU(type=["H100", "A100"], count=2)
        ),
        {"cpu": 2, "memory": 3, "disk": 2, "gpu": 2, "gpu_type": ["gpu:H100", "gpu:A100"]},
    ),
    "unset resources are left to Daytona": (RuntimeResources(cpu=4), {"cpu": 4}),
}


@pytest.mark.parametrize(("resources", "sized"), DAYTONA_SIZES.values(), ids=DAYTONA_SIZES.keys())
async def test_a_daytona_sandbox_is_sized_in_daytonas_units(
    resources: RuntimeResources, sized: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    config = RuntimeConfig(image="registry.test/lab:1", resources=resources)

    async with serving({"lab": lab()}) as addresses:
        daytona = FakeDaytona(addresses["lab"][1]).install(monkeypatch)
        reward = await place(DaytonaRuntime(), configured(config))

    (create,) = called(daytona.calls, "create")
    assert reward == 1.0
    assert create["params"]["from_image"]["resources"] == sized


async def test_a_daytona_snapshot_is_reused_until_its_image_or_size_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = tmp_path / "context"
    context.mkdir()
    (context / "env.py").write_text("v1")
    built: Any = BuiltImage(context)  # stands in for the SDK's daytona.Image
    sized = RuntimeConfig(resources=RuntimeResources(cpu=2, memory_mb=4096))

    async with serving({"lab": lab()}) as addresses:
        daytona = FakeDaytona(addresses["lab"][1]).install(monkeypatch)
        rewards = [
            await place(DaytonaRuntime("ref", image="registry.test/lab:1"), ADD),
            await place(DaytonaRuntime("ref", image="registry.test/lab:1"), ADD),
            await place(DaytonaRuntime("ref", image="registry.test/lab:2"), ADD),
            await place(DaytonaRuntime("built", image=built), ADD),
            await place(DaytonaRuntime("built", image=built), configured(sized)),
        ]
        (context / "env.py").write_text("v2")
        rewards.append(await place(DaytonaRuntime("built", image=built), ADD))

    assert rewards == [1.0] * 6
    assert [
        (name.removeprefix("snapshot."), detail if isinstance(detail, str) else detail["name"])
        for name, detail in daytona.calls
        if name in {"snapshot.create", "snapshot.delete"}
    ] == [
        ("create", "ref"),
        ("delete", "ref"),
        ("create", "ref"),
        ("create", "built"),
        ("create", "built-2cpu-4gb"),
        ("delete", "built"),
        ("create", "built"),
    ]


async def test_a_daytona_sandbox_that_cannot_be_deleted_is_named_in_a_warning(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    async with serving({"lab": lab()}) as addresses:
        FakeDaytona(addresses["lab"][1], delete_fails=True).install(monkeypatch)
        reward = await place(DaytonaRuntime("hud-lab"), ADD)

    assert reward == 1.0
    assert [
        record.getMessage() for record in caplog.records if record.levelno == logging.WARNING
    ] == ["failed to delete Daytona sandbox sandbox-1; it may still be running"]


async def test_a_dropped_daytona_connection_carries_the_envs_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    FakeDaytona(1, env_output="ImportError: no module named bugs\n").install(monkeypatch)

    with pytest.raises(EOFError) as raised:
        async with DaytonaRuntime("hud-lab")(ADD):
            raise EOFError("handshake closed")

    assert raised.value.__notes__ == [
        "env output in sandbox sandbox-1:\nImportError: no module named bugs"
    ]
