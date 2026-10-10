"""Cloud runtimes: what ModalRuntime and DaytonaRuntime ask their SDKs to do.

The SDKs are faked at the module boundary (``tests/harness/cloud.py``); every
sandbox they create connects to an environment served in this process, so each
acquisition below also runs a rollout whose reward proves the placement worked.
"""

# The snapshots hold whole sandbox command lines.
# ruff: noqa: E501

from __future__ import annotations

import asyncio
import json
import logging
import re
import textwrap
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import pytest
from inline_snapshot import snapshot

from hud import Environment
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

    from hud.eval import Provider

ADD = Task(env="lab", id="add", args={"a": 2, "b": 3})


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


async def place(provider: Provider, row: Task) -> tuple[Any, float]:
    """Acquire ``row``'s placement and run its rollout there."""
    async with provider(row) as runtime:
        run = await rollout(row, ScriptedAgent(solve), runtime=runtime)
    return runtime, run.reward


@dataclass(frozen=True)
class UserImage:
    """A ``modal.Image`` the caller defines; the runtime builds it once."""

    builds: list[str] = field(default_factory=list, hash=False, compare=False)

    @property
    def build(self) -> Any:
        async def build(*, app: str) -> None:
            self.builds.append(app)

        return aio(build)


SERVE_ARGS = [
    "sh",
    "-c",
    'mkdir -p /runtime/sessions /media/hud && rm -rf /media/hud/sessions && ln -s /runtime/sessions /media/hud/sessions && exec "$@"',
    "hud-runtime",
    *("hud", "serve", "env.py", "--host", "0.0.0.0", "--port", "8765"),
]


def modal_launch(calls: list[Any]) -> dict[str, Any]:
    """What a Modal launch asked for beyond what every launch asks for.

    Every launch looks up the ``hud-envs`` app, serves ``env.py`` on 8765 behind a
    TCP readiness probe, and terminates its sandbox; this checks that and returns
    the image source, the sandbox options that vary, and the ready timeout.
    """
    (image, app, (create_name, create), (ready_name, ready), terminate) = calls
    assert app == ("App.lookup", {"name": "hud-envs", "create_if_missing": True})
    assert (create_name, ready_name, terminate) == (
        "Sandbox.create",
        "sb-1.wait_until_ready",
        ("sb-1.terminate", {}),
    )
    options = dict(create)
    assert (
        options.pop("args"),
        options.pop("unencrypted_ports"),
        options.pop("readiness_probe"),
    ) == (
        SERVE_ARGS,
        [8765],
        "tcp:8765",
    )
    assert options.pop("app") == "app:hud-envs"
    return {"image": image, "sandbox": options, "ready_timeout": ready["timeout"]}


MODAL: dict[str, tuple[dict[str, Any], RuntimeConfig | None, dict[str, Any] | None, Any]] = {
    "a registry image whose row resources and limits replace the provider's": (
        {"runtime_config": RuntimeConfig(resources=RuntimeResources(cpu=2, memory_mb=4096))},
        RuntimeConfig(
            image="registry.test/lab:1",
            resources=RuntimeResources(gpu=RuntimeGPU(type="A10G", count=2)),
            limits=RuntimeLimits(startup_timeout_s=30, run_timeout_s=120),
        ),
        None,
        snapshot(
            {
                "image": ("Image.from_registry", ["registry.test/lab:1"]),
                "sandbox": {
                    "image": "registry.test/lab:1",
                    "workdir": None,
                    "timeout": 120,
                    "gpu": "A10G:2",
                },
                "ready_timeout": 30,
            }
        ),
    ),
    "a published image by name outlives the agent's budget": (
        {"image_name": "hud-lab"},
        None,
        {"timeout_seconds": 90.5},
        snapshot(
            {
                "image": ("Image.from_name", ["hud-lab"]),
                "sandbox": {"image": "hud-lab", "workdir": None, "timeout": 3691},
                "ready_timeout": 600,
            }
        ),
    ),
    "a modal image id with any GPU and a best-effort storage request": (
        {},
        RuntimeConfig(
            image="modal://im-123",
            resources=RuntimeResources(gpu=RuntimeGPU(), storage_mb=4096),
        ),
        None,
        snapshot(
            {
                "image": ("Image.from_id", ["im-123"]),
                "sandbox": {"image": "im-123", "workdir": None, "timeout": 3600, "gpu": "any"},
                "ready_timeout": 600,
            }
        ),
    ),
    "the row's image beats the published name": (
        {"image_name": "hud-lab"},
        RuntimeConfig(image="registry.test/lab:1"),
        None,
        snapshot(
            {
                "image": ("Image.from_registry", ["registry.test/lab:1"]),
                "sandbox": {"image": "registry.test/lab:1", "workdir": None, "timeout": 3600},
                "ready_timeout": 600,
            }
        ),
    ),
    "workdir, env vars, secrets and a registry secret reach the sandbox": (
        {
            "image_name": "hud-lab",
            "workdir": "/srv",
            "env_vars": {"MODE": "eval"},
            "sandbox_secrets": ["secret:api"],
            "registry_secret": "secret:registry",
            "runtime_config": {"image": "registry.test/lab:1"},
        },
        None,
        None,
        snapshot(
            {
                "image": ("Image.from_registry", ["registry.test/lab:1", "secret:registry"]),
                "sandbox": {
                    "image": "registry.test/lab:1",
                    "workdir": "/srv",
                    "timeout": 3600,
                    "env": {"MODE": "eval"},
                    "secrets": ("secret:api",),
                },
                "ready_timeout": 600,
            }
        ),
    ),
}


@pytest.mark.parametrize(
    ("options", "config", "agent_config", "transcript"), MODAL.values(), ids=MODAL.keys()
)
async def test_a_modal_sandbox_boots_from_the_image_the_row_resolves_to(
    options: dict[str, Any],
    config: RuntimeConfig | None,
    agent_config: dict[str, Any] | None,
    transcript: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    row = ADD.model_copy(update={"runtime_config": config, "agent_config": agent_config})
    images = ("registry.test/lab:1", "hud-lab", "im-123")

    async with serving({"lab": lab()}) as addresses:
        modal = FakeModal({image: addresses["lab"] for image in images}).install(monkeypatch)
        runtime, reward = await place(ModalRuntime(**options), row)

    assert reward == 1.0
    assert (runtime.url, runtime.params) == (
        "tcp://{}:{}".format(*addresses["lab"]),
        {"provider": "modal", "instance_id": "sb-1"},
    )
    assert modal_launch(modal.calls) == transcript


async def test_a_caller_image_is_built_once_in_the_callers_app(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Stand-ins for the SDK's own Image and App objects.
    image: Any = UserImage()
    app: Any = "app:mine"
    provider = ModalRuntime(image=image, app=app)

    async with serving({"lab": lab()}) as addresses:
        modal = FakeModal({image: addresses["lab"]}).install(monkeypatch)
        rewards = [(await place(provider, ADD))[1] for _ in range(2)]

    assert rewards == [1.0, 1.0]
    assert image.builds == ["app:mine"]
    assert [name for name, _ in modal.calls if name in {"App.lookup", "Sandbox.create"}] == [
        "Sandbox.create",
        "Sandbox.create",
    ]


def compose_project(tmp_path: Path, document: str) -> ComposeProject:
    root = tmp_path / "project"
    (root / "recipe").mkdir(parents=True)
    (root / "recipe" / ".env").write_text("TAG=1\n")
    (root / "data.txt").write_text("fixture data\n")
    path = root / "recipe" / "compose.yaml"
    path.write_text(textwrap.dedent(document))
    return ComposeProject(document=path, root=root, service_access=True)


COMPOSE = """
services:
  main:
    image: lab:${TAG}
    network_mode: service:gateway
  gateway:
    image: gateway
    healthcheck: {test: [CMD, "true"]}
"""


async def test_a_modal_compose_project_runs_inside_a_docker_vm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = RuntimeConfig(
        compose=compose_project(tmp_path, COMPOSE),
        resources=RuntimeResources(cpu=2, memory_mb=16384),
        limits=RuntimeLimits(startup_timeout_s=300),
    )
    row = ADD.model_copy(update={"runtime_config": config})

    async with serving({"lab": lab()}) as addresses:
        modal = FakeModal({"docker:28.3.3-dind": addresses["lab"]}).install(monkeypatch)
        runtime, reward = await place(ModalRuntime(env_vars={"MODE": "eval"}), row)

    assert reward == 1.0
    assert runtime.params == {"provider": "modal", "instance_id": "sb-1", "ready_timeout": 300}
    assert modal.calls == snapshot(
        [
            ("Image.from_registry", ["docker:28.3.3-dind"]),
            ("App.lookup", {"name": "hud-envs", "create_if_missing": True}),
            (
                "Sandbox.create",
                {
                    "args": [],
                    "app": "app:hud-envs",
                    "image": "docker:28.3.3-dind",
                    "workdir": None,
                    "unencrypted_ports": [8765],
                    "readiness_probe": None,
                    "timeout": 3600,
                    "experimental_options": {"vm_runtime": True},
                    "cpu": 4.0,
                    "memory": 16384,
                },
            ),
            (
                "sb-1.copy_from_local",
                {
                    "/hud/project.tar.gz": [
                        "data.txt",
                        "recipe",
                        "recipe/.env",
                        "recipe/compose.yaml",
                    ]
                },
            ),
            (
                "sb-1.copy_from_local",
                {
                    "/hud/override.json": {
                        "services": {
                            "main": {
                                "security_opt": [
                                    "seccomp=/hud/docker-seccomp.json",
                                    "systempaths=unconfined",
                                    "apparmor=unconfined",
                                ],
                                "volumes": [
                                    {
                                        "type": "volume",
                                        "source": "hud-runtime-sessions",
                                        "target": "/runtime/sessions",
                                    },
                                    {
                                        "type": "volume",
                                        "source": "hud-runtime-sessions",
                                        "target": "/media/hud/sessions",
                                    },
                                    {
                                        "type": "bind",
                                        "source": "/var/run/docker.sock",
                                        "target": "/var/run/docker.sock",
                                    },
                                    {
                                        "type": "bind",
                                        "source": "/var/run/docker.sock",
                                        "target": "/media/hud/docker.sock",
                                    },
                                ],
                                "environment": {"MODE": "eval"},
                                "cpus": 2.0,
                                "mem_limit": "16384m",
                            }
                        },
                        "volumes": {"hud-runtime-sessions": {}},
                    }
                },
            ),
            (
                "sb-1.copy_from_local",
                {
                    "/hud/ports.yaml": """\
services:
  gateway:
    ports: !override ["8765:8765"]
"""
                },
            ),
            ("sb-1.copy_from_local", {"/hud/docker-seccomp.json": "docker-seccomp.json"}),
            (
                "sb-1.exec",
                {
                    "args": [
                        "sh",
                        "-c",
                        "mkdir -p /hud/project /runtime && tar -xzf /hud/project.tar.gz -C /hud/project && until docker info >/dev/null 2>&1; do sleep 1; done && docker compose --project-directory /hud/project/recipe --file /hud/project/recipe/compose.yaml --file /hud/override.json --file /hud/ports.yaml up --detach --build --remove-orphans && docker compose --project-directory /hud/project/recipe --file /hud/project/recipe/compose.yaml --file /hud/override.json --file /hud/ports.yaml up --wait --no-deps --no-recreate --no-build gateway",
                    ],
                    "timeout": 300,
                },
            ),
            (
                "sb-1.exec",
                {
                    "args": [
                        "docker",
                        "compose",
                        "--project-directory",
                        "/hud/project/recipe",
                        "--file",
                        "/hud/project/recipe/compose.yaml",
                        "--file",
                        "/hud/override.json",
                        "--file",
                        "/hud/ports.yaml",
                        "logs",
                        "--follow",
                        "--no-color",
                    ]
                },
            ),
            (
                "sb-1.exec",
                {
                    "args": [
                        "docker",
                        "compose",
                        "--project-directory",
                        "/hud/project/recipe",
                        "--file",
                        "/hud/project/recipe/compose.yaml",
                        "--file",
                        "/hud/override.json",
                        "--file",
                        "/hud/ports.yaml",
                        "down",
                        "--volumes",
                        "--remove-orphans",
                    ],
                    "timeout": 30,
                },
            ),
            ("sb-1.terminate", {}),
        ]
    )


async def test_a_verifier_on_modal_receives_the_actors_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    verifier = Task(env="judge", id="verify", runtime_config=RuntimeConfig(image="judge"))
    row = Task(
        env="actor", id="solve", runtime_config=RuntimeConfig(image="actor"), verifier=verifier
    )

    async with serving({"actor": actor(), "judge": judge()}) as addresses:
        modal = FakeModal(addresses, execs=[Exec("if [ -d", stdout="1")]).install(monkeypatch)
        run = await rollout(row, ScriptedAgent("secret"), runtime=ModalRuntime())

    assert run.reward == 1.0
    transfers = [
        (name, detail)
        for name, detail in modal.calls
        if name.endswith(("exec", "copy_to_local", "copy_from_local"))
    ]
    assert json.loads(re.sub(r"sess-[0-9a-f]{8}", "<session>", json.dumps(transfers))) == snapshot(
        [
            [
                "sb-1.exec",
                {"args": ["sh", "-c", "if [ -d /runtime/sessions/<session> ]; then printf 1; fi"]},
            ],
            [
                "sb-1.exec",
                {
                    "args": [
                        "sh",
                        "-c",
                        "if find /runtime/sessions/<session> -mindepth 1 ! -type f ! -type d -print -quit | grep -q .; then echo 'runtime session contains an unsupported entry' >&2; exit 1; fi; tar -czf /runtime/session.tar.gz -C /runtime/sessions/<session> .",
                    ]
                },
            ],
            ["sb-1.copy_to_local", {"source": "/runtime/session.tar.gz"}],
            ["sb-2.copy_from_local", {"/runtime/session.tar.gz": []}],
            [
                "sb-2.exec",
                {
                    "args": [
                        "sh",
                        "-c",
                        "rm -rf /runtime/sessions/<session> && mkdir -p /runtime/sessions/<session> && tar -xzf /runtime/session.tar.gz -C /runtime/sessions/<session>",
                    ]
                },
            ],
        ]
    )


async def test_a_verifier_on_a_modal_compose_project_receives_the_actors_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = RuntimeConfig(compose=compose_project(tmp_path, COMPOSE))
    env = judge(actor(Environment("reviewed")))
    row = Task(
        env="reviewed",
        id="solve",
        runtime_config=config,
        verifier=Task(env="reviewed", id="verify", runtime_config=config),
    )

    async with serving({"docker:28.3.3-dind": env}) as addresses:
        modal = FakeModal(addresses, execs=[Exec("test -d", stdout="1")]).install(monkeypatch)
        run = await rollout(row, ScriptedAgent("secret"), runtime=ModalRuntime())

    assert run.reward == 1.0
    session_commands = [
        detail["args"][-1]
        for name, detail in modal.calls
        if name.endswith(".exec") and "/runtime/session" in detail["args"][-1]
    ]
    assert [
        re.sub(r"sess-[0-9a-f]{8}", "<session>", command) for command in session_commands
    ] == snapshot(
        [
            'CONTAINER=$(docker compose --project-directory /hud/project/recipe --file /hud/project/recipe/compose.yaml --file /hud/override.json --file /hud/ports.yaml ps --quiet main); docker exec "$CONTAINER" test -d /runtime/sessions/<session> && printf 1 || true',
            "CONTAINER=$(docker compose --project-directory /hud/project/recipe --file /hud/project/recipe/compose.yaml --file /hud/override.json --file /hud/ports.yaml ps --quiet main); rm -rf /runtime/session-export && mkdir -p /runtime/session-export && docker cp \"$CONTAINER\":/runtime/sessions/<session>/. /runtime/session-export && if find /runtime/session-export -mindepth 1 ! -type f ! -type d -print -quit | grep -q .; then echo 'runtime session contains an unsupported entry' >&2; exit 1; fi; tar -czf /runtime/session.tar.gz -C /runtime/session-export .",
            'CONTAINER=$(docker compose --project-directory /hud/project/recipe --file /hud/project/recipe/compose.yaml --file /hud/override.json --file /hud/ports.yaml ps --quiet main); rm -rf /runtime/session-import && mkdir -p /runtime/session-import && tar -xzf /runtime/session.tar.gz -C /runtime/session-import && docker exec "$CONTAINER" sh -c "rm -rf /runtime/sessions/<session> && mkdir -p /runtime/sessions/<session>" && docker cp /runtime/session-import/. "$CONTAINER":/runtime/sessions/<session>',
        ]
    )


MODAL_REFUSALS: dict[str, tuple[dict[str, Any], Any, type[Exception], str]] = {
    "an app and an app name": (
        {"app": "app:mine", "app_name": "other"},
        None,
        ValueError,
        "ModalRuntime accepts either app or app_name, not both",
    ),
    "no image to boot": (
        {},
        None,
        ValueError,
        "ModalRuntime requires image=, image_name=, runtime_config.image, or "
        "runtime_config.compose",
    ),
    "alternative GPU types": (
        {"image_name": "hud-lab"},
        RuntimeConfig(resources=RuntimeResources(gpu=RuntimeGPU(type=["A100", "H100"]))),
        ValueError,
        "ModalRuntime does not support alternative GPU types",
    ),
    "an OS requirement": (
        {"image_name": "hud-lab"},
        RuntimeConfig(resources=RuntimeResources(os="windows")),
        ValueError,
        "ModalRuntime does not support runtime_config.resources.os",
    ),
    "a GPU for a Compose project": (
        {},
        "compose+gpu",
        ValueError,
        "ModalRuntime cannot attach GPUs to services inside Docker-in-Docker; use a "
        "materialized image or omit runtime_config.compose",
    ),
    "sandbox secrets for a Compose project": (
        {"sandbox_secrets": ["secret:api"]},
        "compose",
        ValueError,
        "ModalRuntime sandbox secrets require an image runtime; attaching them to the outer "
        "Docker-in-Docker sandbox would not expose them to main",
    ),
    "a Compose document that is not a local file": (
        {},
        RuntimeConfig.model_validate({"compose": {"document": {"services": {"main": {}}}}}),
        ValueError,
        "ModalRuntime requires compose as a local file path",
    ),
}


@pytest.mark.parametrize(
    ("options", "config", "error", "message"), MODAL_REFUSALS.values(), ids=MODAL_REFUSALS.keys()
)
async def test_modal_refuses_what_it_cannot_provision_before_creating_a_sandbox(
    options: dict[str, Any],
    config: Any,
    error: type[Exception],
    message: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    modal = FakeModal({}).install(monkeypatch)
    if isinstance(config, str):
        gpu = RuntimeResources(gpu=RuntimeGPU()) if config == "compose+gpu" else None
        config = RuntimeConfig(compose=compose_project(tmp_path, COMPOSE), resources=gpu)

    with pytest.raises(error) as raised:
        async with ModalRuntime(**options)(ADD.model_copy(update={"runtime_config": config})):
            pytest.fail("the acquisition should fail")

    assert str(raised.value) == message
    assert [name for name, _ in modal.calls if name == "Sandbox.create"] == []


COMPOSE_STARTS = {
    "a startup slower than its limit": (
        Exec("up --detach", hang=True),
        TimeoutError,
        "Modal Compose startup timed out after 1 seconds",
    ),
    "a startup that fails": (
        Exec("up --detach", returncode=1, stderr="no such image: gateway\n"),
        RuntimeError,
        "Modal Compose startup failed: no such image: gateway",
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
        compose=compose_project(tmp_path, COMPOSE), limits=RuntimeLimits(startup_timeout_s=1)
    )
    modal = FakeModal({"docker:28.3.3-dind": ("127.0.0.1", 1)}, execs=[startup]).install(
        monkeypatch
    )

    with pytest.raises(error, match=message):
        async with ModalRuntime()(ADD.model_copy(update={"runtime_config": config})):
            pytest.fail("the acquisition should fail")

    assert [name for name, _ in modal.calls][-1] == "sb-1.terminate"


async def test_a_modal_sandbox_streams_its_output_and_exit_does_not_wait_on_it(
    monkeypatch: pytest.MonkeyPatch, capfd: pytest.CaptureFixture[str]
) -> None:
    async with serving({"lab": lab()}) as addresses:
        modal = FakeModal(
            {"hud-lab": addresses["lab"]}, output=("booted", "warned"), hold=asyncio.Event()
        ).install(monkeypatch)
        _, reward = await place(ModalRuntime("hud-lab"), ADD)

    out, err = capfd.readouterr()
    assert (reward, out, err) == (1.0, "booted", "warned")
    assert [name for name, _ in modal.calls][-1] == "sb-1.terminate"


def daytona_launch(calls: list[Any]) -> dict[str, Any]:
    """What a Daytona launch asked for beyond what every launch asks for.

    Every launch starts ``hud serve`` in a ``hud-serve`` session, forwards port
    its port over SSH, and deletes its sandbox; this checks that and returns the
    create request, the serve command, and the SSH access and port it asked for.
    """
    (create, session, (run_name, (run_session, run)), (_, expires), (_, ssh), forward, *rest) = (
        calls
    )
    assert session == ("sandbox-1.create_session", "hud-serve")
    assert (run_name, run_session, run["run_async"]) == (
        "sandbox-1.execute_session_command",
        "hud-serve",
        True,
    )
    (forward_name, (local_host, local_port, remote_host, remote_port)) = forward
    assert (forward_name, local_host, local_port, remote_host) == (
        "ssh.forward_local_port",
        "127.0.0.1",
        0,
        "127.0.0.1",
    )
    assert rest == [("ssh.close", {}), ("delete", "sandbox-1"), ("close", {})]
    assert ssh["username"] == "token-sandbox-1"
    return {
        "create": create,
        "command": run["command"],
        "ssh": {"host": ssh["host"], "expires_minutes": expires, "port": remote_port},
    }


DAYTONA: dict[str, tuple[dict[str, Any], RuntimeConfig | None, Any]] = {
    "an image row sized to whole cores and gibibytes": (
        {},
        RuntimeConfig(
            image="registry.test/lab:1",
            resources=RuntimeResources(
                cpu=2,
                memory_mb=3000,
                storage_mb=1025,
                gpu=RuntimeGPU(type=["H100", "A100"], count=2),
            ),
            limits=RuntimeLimits(startup_timeout_s=45),
        ),
        snapshot(
            {
                "create": (
                    "create",
                    {
                        "params": {
                            "from_image": {
                                "image": "base:registry.test/lab:1",
                                "ephemeral": True,
                                "auto_stop_interval": 0,
                                "resources": {
                                    "cpu": 2,
                                    "memory": 3,
                                    "disk": 2,
                                    "gpu": 2,
                                    "gpu_type": ["gpu:H100", "gpu:A100"],
                                },
                            }
                        },
                        "timeout": 45,
                    },
                ),
                "command": 'cd /app && PATH="$PWD/.venv/bin:$PATH" hud serve env.py --host 0.0.0.0 --port 8765',
                "ssh": {"host": "ssh.app.daytona.io", "expires_minutes": 1440, "port": 8765},
            }
        ),
    ),
    "a prebuilt snapshot": (
        {"snapshot_name": "hud-lab"},
        None,
        snapshot(
            {
                "create": (
                    "create",
                    {
                        "params": {
                            "from_snapshot": {
                                "snapshot": "hud-lab",
                                "ephemeral": True,
                                "auto_stop_interval": 0,
                            }
                        },
                        "timeout": 120,
                    },
                ),
                "command": 'cd /app && PATH="$PWD/.venv/bin:$PATH" hud serve env.py --host 0.0.0.0 --port 8765',
                "ssh": {"host": "ssh.app.daytona.io", "expires_minutes": 1440, "port": 8765},
            }
        ),
    ),
    "the row's image overlays the provider's sizing": (
        {"runtime_config": {"resources": {"cpu": 4}}, "workdir": None, "port": 9000},
        RuntimeConfig(image="registry.test/lab:1"),
        snapshot(
            {
                "create": (
                    "create",
                    {
                        "params": {
                            "from_image": {
                                "image": "base:registry.test/lab:1",
                                "ephemeral": True,
                                "auto_stop_interval": 0,
                                "resources": {"cpu": 4},
                            }
                        },
                        "timeout": 120,
                    },
                ),
                "command": 'PATH="$PWD/.venv/bin:$PATH" hud serve env.py --host 0.0.0.0 --port 9000',
                "ssh": {"host": "ssh.app.daytona.io", "expires_minutes": 1440, "port": 9000},
            }
        ),
    ),
}


@pytest.mark.parametrize(("options", "config", "transcript"), DAYTONA.values(), ids=DAYTONA.keys())
async def test_a_daytona_sandbox_serves_over_an_ssh_forward(
    options: dict[str, Any],
    config: RuntimeConfig | None,
    transcript: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with serving({"lab": lab()}) as addresses:
        daytona = FakeDaytona(addresses["lab"][1]).install(monkeypatch)
        runtime, reward = await place(
            DaytonaRuntime(**options), ADD.model_copy(update={"runtime_config": config})
        )

    assert reward == 1.0
    assert runtime.params == {"provider": "daytona", "instance_id": "sandbox-1"}
    assert daytona_launch(daytona.calls) == transcript


async def test_a_daytona_snapshot_follows_its_image(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = tmp_path / "context"
    context.mkdir()
    (context / "env.py").write_text("v1")
    built: Any = BuiltImage(context)  # stands in for the SDK's daytona.Image
    sized = RuntimeConfig(resources=RuntimeResources(cpu=2, memory_mb=4096))

    async with serving({"lab": lab()}) as addresses:
        daytona = FakeDaytona(addresses["lab"][1]).install(monkeypatch)
        steps = [
            ("registry ref", DaytonaRuntime("ref", image="registry.test/lab:1"), None),
            ("same ref", DaytonaRuntime("ref", image="registry.test/lab:1"), None),
            ("new ref", DaytonaRuntime("ref", image="registry.test/lab:2"), None),
            ("built", DaytonaRuntime("built", image=built), None),
            ("sized", DaytonaRuntime("built", image=built), sized),
        ]
        rewards = []
        for _, provider, config in steps:
            rewards.append(
                (await place(provider, ADD.model_copy(update={"runtime_config": config})))[1]
            )
        (context / "env.py").write_text("v2")
        rewards.append((await place(DaytonaRuntime("built", image=built), ADD))[1])

    assert rewards == [1.0] * 6
    assert [
        (name, detail)
        for name, detail in daytona.calls
        if name.startswith("snapshot.") or name == "create"
    ] == snapshot(
        [
            ("snapshot.get", "ref"),
            ("snapshot.create", {"name": "ref", "image": "registry.test/lab:1"}),
            (
                "create",
                {
                    "params": {
                        "from_snapshot": {
                            "snapshot": "ref",
                            "ephemeral": True,
                            "auto_stop_interval": 0,
                        }
                    },
                    "timeout": 120,
                },
            ),
            ("snapshot.get", "ref"),
            (
                "create",
                {
                    "params": {
                        "from_snapshot": {
                            "snapshot": "ref",
                            "ephemeral": True,
                            "auto_stop_interval": 0,
                        }
                    },
                    "timeout": 120,
                },
            ),
            ("snapshot.get", "ref"),
            ("snapshot.delete", "ref"),
            ("snapshot.get", "ref"),
            ("snapshot.get", "ref"),
            ("snapshot.create", {"name": "ref", "image": "registry.test/lab:2"}),
            (
                "create",
                {
                    "params": {
                        "from_snapshot": {
                            "snapshot": "ref",
                            "ephemeral": True,
                            "auto_stop_interval": 0,
                        }
                    },
                    "timeout": 120,
                },
            ),
            ("snapshot.get", "built"),
            ("snapshot.create", {"name": "built", "image": "built image"}),
            (
                "create",
                {
                    "params": {
                        "from_snapshot": {
                            "snapshot": "built",
                            "ephemeral": True,
                            "auto_stop_interval": 0,
                        }
                    },
                    "timeout": 120,
                },
            ),
            ("snapshot.get", "built-2cpu-4gb"),
            (
                "snapshot.create",
                {
                    "name": "built-2cpu-4gb",
                    "image": "built image",
                    "resources": {"cpu": 2, "memory": 4},
                },
            ),
            (
                "create",
                {
                    "params": {
                        "from_snapshot": {
                            "snapshot": "built-2cpu-4gb",
                            "ephemeral": True,
                            "auto_stop_interval": 0,
                        }
                    },
                    "timeout": 120,
                },
            ),
            ("snapshot.get", "built"),
            ("snapshot.delete", "built"),
            ("snapshot.get", "built"),
            ("snapshot.get", "built"),
            ("snapshot.create", {"name": "built", "image": "built image"}),
            (
                "create",
                {
                    "params": {
                        "from_snapshot": {
                            "snapshot": "built",
                            "ephemeral": True,
                            "auto_stop_interval": 0,
                        }
                    },
                    "timeout": 120,
                },
            ),
        ]
    )


DAYTONA_REFUSALS: dict[str, tuple[dict[str, Any], RuntimeConfig | None, str]] = {
    "a Compose project": (
        {"snapshot_name": "hud-lab"},
        RuntimeConfig.model_validate({"compose": {"document": {"services": {"main": {}}}}}),
        "DaytonaRuntime does not support runtime_config.compose",
    ),
    "a run limit": (
        {"snapshot_name": "hud-lab"},
        RuntimeConfig(limits=RuntimeLimits(run_timeout_s=60)),
        "DaytonaRuntime does not support runtime_config.run_timeout_s",
    ),
    "a fractional CPU": (
        {},
        RuntimeConfig(image="registry.test/lab:1", resources=RuntimeResources(cpu=1.5)),
        "DaytonaRuntime needs a whole number of CPUs, got 1.5",
    ),
    "resources for a prebuilt snapshot": (
        {"snapshot_name": "hud-lab"},
        RuntimeConfig(resources=RuntimeResources(cpu=2)),
        "DaytonaRuntime cannot resize an already-built snapshot: resources are fixed when it "
        "is built, so pass image= to build one",
    ),
    "no snapshot and no image": (
        {},
        None,
        "DaytonaRuntime requires snapshot_name or runtime_config.image",
    ),
    "a TPU": (
        {"snapshot_name": "hud-lab"},
        RuntimeConfig(resources=RuntimeResources(tpu={"type": "v5e", "topology": "2x2"})),
        "DaytonaRuntime does not support runtime_config.resources.tpu",
    ),
}


@pytest.mark.parametrize(
    ("options", "config", "message"), DAYTONA_REFUSALS.values(), ids=DAYTONA_REFUSALS.keys()
)
async def test_daytona_refuses_what_it_cannot_provision_before_creating_a_sandbox(
    options: dict[str, Any],
    config: RuntimeConfig | None,
    message: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daytona = FakeDaytona(1).install(monkeypatch)

    with pytest.raises(ValueError) as raised:
        async with DaytonaRuntime(**options)(ADD.model_copy(update={"runtime_config": config})):
            pytest.fail("the acquisition should fail")

    assert str(raised.value) == message
    assert [name for name, _ in daytona.calls] == ["close"]


async def test_a_daytona_sandbox_that_cannot_be_deleted_is_named_in_a_warning(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    async with serving({"lab": lab()}) as addresses:
        FakeDaytona(addresses["lab"][1], delete_fails=True).install(monkeypatch)
        _, reward = await place(DaytonaRuntime("hud-lab"), ADD)

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


async def test_a_daytona_sandbox_streams_its_output_and_exit_does_not_wait_on_it(
    monkeypatch: pytest.MonkeyPatch, capfd: pytest.CaptureFixture[str]
) -> None:
    async with serving({"lab": lab()}) as addresses:
        daytona = FakeDaytona(
            addresses["lab"][1], output=("booted\n", "warned\n"), hold=asyncio.Event()
        ).install(monkeypatch)
        _, reward = await place(DaytonaRuntime("hud-lab"), ADD)

    out, err = capfd.readouterr()
    assert (reward, out, err) == (1.0, "booted\n", "warned\n")
    assert [name for name, _ in daytona.calls][-2:] == ["delete", "close"]
