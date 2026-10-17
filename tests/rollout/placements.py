"""Every runtime a rollout can be placed on, as the rows the rollout contracts run over.

A row opens its runtime around one fixture environment and says how to tell,
afterwards, that the rollout left its placement clean: no env left serving and
no container, sandbox or runtime session behind, while an env it only borrowed
keeps serving. Rows on a real daemon or account carry the marks of the lane that
has one.

Each contract names the runtimes it does not apply to, with the reason, in
``NOT_APPLICABLE``; every runtime ``hud.eval`` exports has a row in each contract
or an entry there, and ``test_every_exported_runtime_has_a_row_in_every_contract``
holds the tables to that.
"""

from __future__ import annotations

import socket
import subprocess
import typing
import uuid
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import pytest

from hud.eval import (
    DaytonaRuntime,
    DockerRuntime,
    HostedRuntime,
    HUDRuntime,
    LocalRuntime,
    ModalRuntime,
    Runtime,
    RuntimeConfig,
    RuntimeLimits,
    Shared,
    SubprocessRuntime,
    Task,
)
from hud.eval import runtime as runtime_module
from tests.support import RecordingProvider
from tests.support.runtime import SESSION, SESSIONS, host_on_runtime

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Awaitable, Callable
    from pathlib import Path

    from tests.support import FakeServices, HudEnv, Platform

ROLLOUTS = "/v2/rollouts/submit"


@dataclass(frozen=True)
class World:
    """What a row may stand up around the fixture at ``path``: the fake HUD services, the
    images and accounts it asks pytest for, a startup limit for the env, and the
    deprecated constructor deadline for the runtimes that still take one."""

    path: Path
    services: FakeServices
    hud_env: HudEnv
    fixtures: pytest.FixtureRequest
    startup_limit: int | None = None
    run_timeout: float | None = None
    label: str = field(default_factory=lambda: uuid.uuid4().hex)
    recorders: list[RecordingProvider] = field(default_factory=list)
    borrowed: list[str] = field(default_factory=list)

    def record(self, provider: Any) -> RecordingProvider:
        recording = RecordingProvider(provider)
        self.recorders.append(recording)
        return recording

    @property
    def platform(self) -> Platform:
        return self.fixtures.getfixturevalue("platform")

    def limits(self) -> RuntimeConfig | None:
        if self.startup_limit is None:
            return None
        return RuntimeConfig(limits=RuntimeLimits(startup_timeout_s=self.startup_limit))


@dataclass(frozen=True)
class Placement:
    """One runtime as a row: how to open it, how to tell it was left clean, and how long
    its provisioning may take inside a rollout deadline."""

    kind: type
    open: Callable[[World], AbstractAsyncContextManager[Any]]
    clean: Callable[[World], Awaitable[bool]]
    marks: tuple[pytest.MarkDecorator, ...] = ()
    patience: float = 5.0


def serves(url: str) -> bool:
    host, port = url.removeprefix("tcp://").rsplit(":", 1)
    try:
        socket.create_connection((host, int(port)), timeout=1).close()
    except OSError:
        return False
    return True


def containers(label: str) -> list[str]:
    out = subprocess.run(
        ["docker", "ps", "--all", "--quiet", "--filter", f"label=hud-test={label}"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return out.split()


async def stopped(world: World) -> bool:
    """Every placement the world recorded stopped, and none of its envs still serves."""
    return all(
        [event for event, _ in recording.events].count("start")
        == [event for event, _ in recording.events].count("stop")
        and not any(serves(runtime.url) for runtime in recording.runtimes)
        for recording in world.recorders
    )


async def left_serving(world: World) -> bool:
    return all(serves(f"tcp://{address}") for address in world.borrowed)


async def no_containers(world: World) -> bool:
    return containers(world.label) == []


async def sessions_released(world: World) -> bool:
    created = len(world.services.requests("runtime", "POST", SESSIONS))
    return created > 0 and len(world.services.requests("runtime", "DELETE", SESSION)) == created


async def sandboxes_terminated(world: World) -> bool:
    import modal

    ids = [runtime.params["instance_id"] for r in world.recorders for runtime in r.runtimes]
    for instance in ids:
        sandbox = await modal.Sandbox.from_id.aio(instance)
        if await sandbox.poll.aio() is None:
            return False
    return await stopped(world)


async def cancelled_remotely(world: World) -> bool:
    traces = world.platform.traces
    return traces != {} and all(trace.cancelled for trace in traces.values())


@asynccontextmanager
async def served_apart(path: Path) -> AsyncIterator[str]:
    """Serve ``path`` in a child process and yield its address, so no test shares its env."""
    async with SubprocessRuntime(path)(Task(env=path.stem, id="serve")) as served:
        yield served.url.removeprefix("tcp://")


def ready_timeout(world: World) -> float:
    return float(world.startup_limit or 120)


@asynccontextmanager
async def in_process(world: World) -> AsyncIterator[Any]:
    yield world.record(LocalRuntime(world.path, ready_timeout=ready_timeout(world)))


@asynccontextmanager
async def child_process(world: World) -> AsyncIterator[Any]:
    yield world.record(SubprocessRuntime(world.path, ready_timeout=ready_timeout(world)))


@asynccontextmanager
async def shared_pool(world: World) -> AsyncIterator[Any]:
    inner = world.record(LocalRuntime(world.path, ready_timeout=ready_timeout(world)))
    yield Shared(inner, width=2)


@asynccontextmanager
async def attached(world: World) -> AsyncIterator[Any]:
    async with served_apart(world.path) as address:
        world.borrowed.append(address)
        yield Runtime(f"tcp://{address}")


@asynccontextmanager
async def container(world: World) -> AsyncIterator[Any]:
    image = world.fixtures.getfixturevalue("fixture_image")(world.path.stem)
    yield DockerRuntime(
        image, run_args=["--label", f"hud-test={world.label}"], runtime_config=world.limits()
    )


@asynccontextmanager
async def hud_tunnel(world: World) -> AsyncIterator[Any]:
    world.hud_env.set(HUD_API_KEY="k")
    async with served_apart(world.path) as address:
        host_on_runtime(world.services, int(address.rsplit(":", 1)[1]))
        yield HUDRuntime(run_timeout=world.run_timeout)


@asynccontextmanager
async def modal_sandbox(world: World) -> AsyncIterator[Any]:
    world.fixtures.getfixturevalue("modal_account")
    image = world.fixtures.getfixturevalue("modal_image")(world.path.stem)
    yield world.record(ModalRuntime(image=image, workdir="/hud", runtime_config=world.limits()))


@asynccontextmanager
async def hosted(world: World) -> AsyncIterator[Any]:
    """The platform runs the rollout and reports it running until it is cancelled."""
    world.platform.next_rollout = [{"status": "running"}]
    yield HostedRuntime(poll_interval=0, run_timeout=world.run_timeout)


PLACEMENTS: dict[str, Placement] = {
    "in-process": Placement(LocalRuntime, in_process, stopped),
    "child-process": Placement(SubprocessRuntime, child_process, stopped),
    "shared-pool": Placement(Shared, shared_pool, stopped),
    "attached": Placement(Runtime, attached, left_serving),
    "container": Placement(
        DockerRuntime, container, no_containers, marks=(pytest.mark.e2e, pytest.mark.docker)
    ),
    "hud-tunnel": Placement(HUDRuntime, hud_tunnel, sessions_released),
    "modal": Placement(
        ModalRuntime,
        modal_sandbox,
        sandboxes_terminated,
        marks=(pytest.mark.e2e, pytest.mark.modal),
        patience=90.0,
    ),
    "hosted": Placement(HostedRuntime, hosted, cancelled_remotely, patience=1.0),
}

UNPLACED: dict[type, str] = {
    DaytonaRuntime: "needs a Daytona account, and no lane has one; its refusals are rows "
    "in test_refusals",
}
HOSTED = (
    "the platform runs the environment and the agent; what it reports folds into the Run "
    "(test_hosted)"
)
NOT_APPLICABLE: dict[str, dict[type, str]] = {
    "outcomes": UNPLACED | {HostedRuntime: HOSTED},
    "startup": UNPLACED
    | {
        HostedRuntime: HOSTED,
        Runtime: "a borrowed address is served before the rollout starts",
        HUDRuntime: "the platform serves the environment before the rollout starts",
    },
    "deadline": UNPLACED,
    "cancel": UNPLACED,
    "session-transport": UNPLACED
    | {
        LocalRuntime: "rejects a runtime_config, so a verifier cannot ask for its own runtime",
        SubprocessRuntime: "rejects a runtime_config, so a verifier cannot ask for its own runtime",
        Shared: "transports what its inner provider transports",
        Runtime: "cannot export a session; the verifier runs beside the actor (test_verifier)",
        HUDRuntime: "cannot export a session; the verifier runs beside the actor (test_verifier)",
        HostedRuntime: "the platform places the verifier from the submitted row (test_hosted)",
    },
}


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


def rows(contract: str) -> list[Any]:
    """The placements ``contract`` runs over, as ``pytest.param`` ids with their lane marks."""
    skipped = NOT_APPLICABLE[contract]
    return [
        pytest.param(name, id=name, marks=placement.marks)
        for name, placement in PLACEMENTS.items()
        if placement.kind not in skipped
    ]
