"""DockerRuntime: what it asks the docker CLI to do, and the runtime it yields.

The fake ``docker`` on ``PATH`` records every invocation; its ``port`` answer
points at an environment served in this process, so each started container
really serves a rollout whose reward proves the placement worked.
"""

# The snapshots hold whole docker command lines.
# ruff: noqa: E501

from __future__ import annotations

import asyncio
import io
import json
import re
import tarfile
import textwrap
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
from dirty_equals import IsStr
from inline_snapshot import snapshot

import hud
from hud import Environment
from hud.eval import (
    ComposeProject,
    DockerRuntime,
    RuntimeConfig,
    RuntimeGPU,
    RuntimeLimits,
    RuntimeResources,
    Task,
    rollout,
)
from hud.eval.runtime import DockerBindMount
from tests.eval.envs import DF_FREE, actor, containers, eventually, judge, lab, solve
from tests.harness import ScriptedAgent

if TYPE_CHECKING:
    from hud.eval import Runtime
    from tests.harness import FakeDocker

SECCOMP = Path(hud.__file__).parent / "eval" / "docker-seccomp.json"
ADD = Task(env="lab", id="add", args={"a": 2, "b": 3})


def scrub(text: str, tmp_path: Path) -> str:
    """Docker arguments with their run-specific names replaced by placeholders."""
    for pattern, placeholder in (
        (r"hud-runtime-sessions-[0-9a-f]{32}", "<volume>"),
        (r"hud-[0-9a-f]{12}\b", "<project>"),
        (r"[^\s\"]*/hud-compose-[^/\s\"]+", "<staged>"),
        (r"[^\s\"]*/hud-session(-import)?-[^/\s\"]+", "<local>"),
        (r"/runtime/session-export-[0-9a-f]{32}", "<export>"),
        (r"sess-[0-9a-f]{8}", "<session>"),
        (r"127\.0\.0\.1:\d{4,5}\b", "127.0.0.1:<port>"),
    ):
        text = re.sub(pattern, placeholder, text)
    return text.replace(str(SECCOMP), "<seccomp>").replace(str(tmp_path), "<tmp>")


def transcript(fake_docker: FakeDocker, tmp_path: Path) -> list[str]:
    """The docker commands run, less the log followers, which may be stopped before they log."""
    return [
        scrub(" ".join("<script>" if "\n" in arg else arg for arg in call.argv), tmp_path)
        for call in fake_docker.calls
        if "logs --follow" not in call.command
    ]


def staged(fake_docker: FakeDocker, tmp_path: Path, name: str) -> Any:
    """The staged Compose file ``name`` as ``compose up`` read it."""
    (text,) = {
        text
        for call in fake_docker.calls
        if " up --detach " in call.command
        for path, text in call.files.items()
        if path.endswith(name)
    }
    scrubbed = scrub(text, tmp_path)
    return json.loads(scrubbed) if name.endswith(".json") else scrubbed


async def run_on(provider: DockerRuntime, row: Task) -> tuple[Runtime, float]:
    """Acquire ``row``'s placement, run the rollout on it, and return both."""
    async with provider(row) as runtime:
        run = await rollout(row, ScriptedAgent(solve), runtime=runtime)
    return runtime, run.reward


def docker_run(fake_docker: FakeDocker) -> tuple[list[str], str]:
    """What ``docker run`` was asked for: the options before the sandbox profile, and the image."""
    (argv,) = [call.argv for call in fake_docker.calls if call.argv[:1] == ["run"]]
    return argv[2 : argv.index("--security-opt")], argv[-1]


async def test_an_image_row_runs_in_a_sandboxed_container_that_is_removed_after(
    fake_docker: FakeDocker, tmp_path: Path
) -> None:
    async with containers(fake_docker, tmp_path / "rootfs", {"lab:1": lab()}):
        runtime, reward = await run_on(DockerRuntime("lab:1"), ADD)

    assert reward == 1.0
    assert runtime.params == {}
    assert transcript(fake_docker, tmp_path) == snapshot(
        [
            "volume create <volume>",
            "run --detach --security-opt seccomp=<seccomp> --security-opt systempaths=unconfined --security-opt apparmor=unconfined --mount type=volume,source=<volume>,target=/runtime/sessions --mount type=volume,source=<volume>,target=/media/hud/sessions --publish 127.0.0.1::8765 lab:1",
            "port lab-1 8765",
            "rm --force lab-1",
            "volume rm --force <volume>",
        ]
    )


IMAGES: dict[str, tuple[dict[str, Any], RuntimeConfig | None, list[str], dict[str, Any]]] = {
    "run args, env vars, bind mounts and resources reach docker run": (
        {
            "image": "lab:1",
            "run_args": ["--network", "host"],
            "env_vars": {"MODE": "eval"},
            "bind_mounts": [DockerBindMount("/opt/data", "/data")],
            "runtime_config": RuntimeConfig(
                resources=RuntimeResources(cpu=2, memory_mb=4096, gpu=RuntimeGPU())
            ),
        },
        None,
        [
            *("--network", "host", "--env", "MODE=eval"),
            *("--cpus", "2", "--memory", "4096m", "--gpus", "1"),
            *("--mount", "type=bind,source=/opt/data,target=/data,readonly"),
        ],
        {},
    ),
    "the row's image replaces the provider's and keeps its resources": (
        {"image": "lab:old", "runtime_config": {"resources": {"cpu": 1.5}}},
        RuntimeConfig(image="lab:1"),
        ["--cpus", "1.5"],
        {},
    ),
    "the row's resources replace the provider's whole": (
        {"image": "lab:1", "runtime_config": {"resources": {"cpu": 2, "memory_mb": 4096}}},
        RuntimeConfig(resources=RuntimeResources(cpu=4)),
        ["--cpus", "4"],
        {},
    ),
    "a constructor runtime_config image beats the positional one": (
        {"image": "lab:old", "runtime_config": {"image": "lab:1"}},
        None,
        [],
        {},
    ),
    "a startup limit becomes the runtime's ready timeout": (
        {"image": "lab:1"},
        RuntimeConfig(limits=RuntimeLimits(startup_timeout_s=300)),
        [],
        {"ready_timeout": 300},
    ),
}


@pytest.mark.parametrize(
    ("options", "config", "run_options", "params"), IMAGES.values(), ids=IMAGES.keys()
)
async def test_a_row_and_its_provider_decide_what_docker_run_is_asked_for(
    options: dict[str, Any],
    config: RuntimeConfig | None,
    run_options: list[str],
    params: dict[str, Any],
    fake_docker: FakeDocker,
    tmp_path: Path,
) -> None:
    row = ADD.model_copy(update={"runtime_config": config})

    async with containers(fake_docker, tmp_path / "rootfs", {"lab:1": lab()}):
        runtime, reward = await run_on(DockerRuntime(**options), row)

    assert reward == 1.0
    assert docker_run(fake_docker) == (run_options, "lab:1")
    assert runtime.params == params


async def test_a_storage_request_is_admitted_against_the_containers_free_disk(
    fake_docker: FakeDocker, tmp_path: Path
) -> None:
    row = ADD.model_copy(
        update={"runtime_config": RuntimeConfig(resources=RuntimeResources(storage_mb=1024))}
    )

    async with containers(fake_docker, tmp_path / "rootfs", {"lab:1": lab()}):
        _, reward = await run_on(DockerRuntime("lab:1"), row)

    assert reward == 1.0
    assert fake_docker.commands("exec ") == ["exec lab-1 df -Pk /"]


async def test_a_container_streams_its_logs_to_the_terminal(
    fake_docker: FakeDocker, tmp_path: Path, capfd: pytest.CaptureFixture[str]
) -> None:
    fake_docker.on(r"^logs --follow ", stdout="env says hello\n", stderr="env warns\n")
    seen = {"out": "", "err": ""}

    def heard() -> bool:
        out, err = capfd.readouterr()
        seen["out"] += out
        seen["err"] += err
        return "env says hello" in seen["out"] and "env warns" in seen["err"]

    async with (
        containers(fake_docker, tmp_path / "rootfs", {"lab:1": lab()}),
        DockerRuntime("lab:1")(ADD),
    ):
        await eventually(heard)


LOW_DISK = "Filesystem 1024-blocks Used Available Capacity Mounted on\noverlay 0 0 4194304 0% /\n"


def volumes_left(fake_docker: FakeDocker) -> set[str]:
    """Session volumes created and never removed."""
    created = {call.argv[2] for call in fake_docker.calls if call.argv[:2] == ["volume", "create"]}
    removed = {call.argv[-1] for call in fake_docker.calls if call.argv[:2] == ["volume", "rm"]}
    return created - removed


IMAGE_FAILURES: dict[
    str, tuple[RuntimeConfig, list[tuple[str, dict[str, Any]]], str, list[str]]
] = {
    "a GPU type it cannot select": (
        RuntimeConfig(image="lab:1", resources=RuntimeResources(gpu=RuntimeGPU(type="H100"))),
        [],
        "DockerRuntime cannot select GPUs by type",
        [],
    ),
    "no image at all": (
        RuntimeConfig(resources=RuntimeResources(cpu=1)),
        [],
        "DockerRuntime requires runtime_config.image or runtime_config.compose",
        [],
    ),
    "a container that exits before serving": (
        RuntimeConfig(image="lab:1"),
        [(r"^logs --tail 40 ", {"stderr": "ImportError: boom\n"})],
        "container for image 'lab:1' exited before serving port 8765:\nImportError: boom",
        ["rm --force cid-1"],
    ),
    "less free disk than requested": (
        RuntimeConfig(image="lab:1", resources=RuntimeResources(storage_mb=8192)),
        [(r"^exec \S+ df -Pk /$", {"stdout": LOW_DISK})],
        "DockerRuntime requires 8192 MB of free disk; the environment has 4096 MB",
        ["rm --force cid-1"],
    ),
    "free disk it cannot read": (
        RuntimeConfig(image="lab:1", resources=RuntimeResources(storage_mb=8192)),
        [(r"^exec \S+ df -Pk /$", {"stdout": "df: not found\n"})],
        "DockerRuntime could not measure the environment's free disk",
        ["rm --force cid-1"],
    ),
    "a start slower than its startup limit": (
        RuntimeConfig(image="lab:1", limits=RuntimeLimits(startup_timeout_s=1)),
        [(r"^run ", {"stdout": "cid-1\n", "delay": 3})],
        IsStr(regex=r"docker run --detach .* lab:1 timed out after 1s"),
        [],
    ),
}


@pytest.mark.parametrize(
    ("config", "rules", "error", "removed"), IMAGE_FAILURES.values(), ids=IMAGE_FAILURES.keys()
)
async def test_an_image_that_cannot_serve_fails_its_acquisition_and_is_cleaned_up(
    config: RuntimeConfig,
    rules: list[tuple[str, dict[str, Any]]],
    error: Any,
    removed: list[str],
    fake_docker: FakeDocker,
    tmp_path: Path,
) -> None:
    for match, answer in rules:
        fake_docker.on(match, **answer)

    with pytest.raises((ValueError, RuntimeError)) as raised:
        async with DockerRuntime()(ADD.model_copy(update={"runtime_config": config})):
            pytest.fail("the acquisition should fail")

    assert scrub(str(raised.value), tmp_path) == error
    assert fake_docker.commands("rm --force ") == removed
    assert volumes_left(fake_docker) == set()
    if not rules:
        assert fake_docker.calls == []


async def test_a_container_past_its_run_limit_is_removed(
    fake_docker: FakeDocker, tmp_path: Path
) -> None:
    fake_docker.on(r"^port ", stdout="127.0.0.1:43210\n")
    row = ADD.model_copy(
        update={
            "runtime_config": RuntimeConfig(image="lab:1", limits=RuntimeLimits(run_timeout_s=1))
        }
    )

    async with DockerRuntime()(row):
        await asyncio.sleep(1.2)
        removed_while_held = fake_docker.commands("rm --force ")

    assert removed_while_held == ["rm --force cid-1"]
    assert fake_docker.commands("rm --force ") == ["rm --force cid-1", "rm --force cid-1"]


def test_the_sandbox_profile_allows_workspace_namespaces_and_denies_kernel_interfaces() -> None:
    profile = json.loads(SECCOMP.read_text())
    denied = {
        name
        for rule in profile["syscalls"]
        if rule["action"] != profile["defaultAction"] and not rule.get("includes", {}).get("caps")
        for name in rule["names"]
    }

    assert profile["defaultAction"] == "SCMP_ACT_ALLOW"
    assert {"mount", "pivot_root", "setns", "umount", "umount2", "unshare"}.isdisjoint(denied)
    assert {"bpf", "keyctl", "perf_event_open", "ptrace", "userfaultfd"} <= denied


def compose_file(tmp_path: Path, document: str, env_file: str = "TAG=1\n") -> Path:
    project = tmp_path / "project"
    project.mkdir()
    (project / ".env").write_text(env_file)
    path = project / "compose.yaml"
    path.write_text(textwrap.dedent(document))
    return path


HEALTHCHECKED = """
services:
  main:
    image: lab:${TAG}
    environment: [MODE=eval, EMPTY=]
    command: hud serve env.py --port 8765
    healthcheck: {test: [CMD, "true"]}
  db:
    image: postgres:16
    healthcheck: {test: [CMD, pg_isready], interval: 1s}
  cache:
    image: redis
    healthcheck: {test: [NONE]}
  worker:
    image: worker
    healthcheck: {test: [CMD, "true"], disable: true}
  tools:
    image: tools
    profiles: [debug]
    healthcheck: {test: [CMD, "true"]}
  idle:
    image: idle
    scale: 0
    healthcheck: {test: [CMD, "true"]}
  plain:
    image: plain
    ports: ["8080", "127.0.0.1:9000:9000/udp"]
    expose: [7000]
"""

CHAINED = """
services:
  main:
    image: lab
    network_mode: service:relay
  relay:
    image: relay
    network_mode: service:gateway
  gateway:
    image: gateway
"""

COMPOSE: dict[str, tuple[str, bool, dict[str, Any], RuntimeConfig | None, dict[str, str], Any]] = {
    "healthchecked services are awaited and resources land on main": (
        HEALTHCHECKED,
        False,
        {"env_vars": {"MODE": "override"}, "bind_mounts": [DockerBindMount("/opt/data", "/data")]},
        RuntimeConfig(resources=RuntimeResources(cpu=2, memory_mb=4096, gpu=RuntimeGPU(count=2))),
        {},
        snapshot(
            {
                "commands": [
                    "up --detach --build --remove-orphans",
                    "up --wait --no-deps --no-recreate --no-build main db",
                    "port main 8765",
                    "ps --quiet main",
                    "down --volumes --remove-orphans",
                ],
                "main": {
                    "environment": {"MODE": "override"},
                    "cpus": 2.0,
                    "mem_limit": "4096m",
                    "gpus": 2,
                    "volumes": [
                        {
                            "type": "bind",
                            "source": "/opt/data",
                            "target": "/data",
                            "read_only": True,
                        }
                    ],
                },
                "ports": """\
services:
  main:
    ports: !override ["127.0.0.1::8765"]
""",
            }
        ),
    ),
    "the published port belongs to the end of main's network chain": (
        CHAINED,
        False,
        {},
        None,
        {},
        snapshot(
            {
                "commands": [
                    "up --detach --build --remove-orphans",
                    "port gateway 8765",
                    "ps --quiet main",
                    "down --volumes --remove-orphans",
                ],
                "main": {},
                "ports": """\
services:
  gateway:
    ports: !override ["127.0.0.1::8765"]
""",
            }
        ),
    ),
    "service access mounts the local daemon's socket": (
        CHAINED,
        True,
        {},
        None,
        {"DOCKER_HOST": "unix:///run/user/docker.sock"},
        snapshot(
            {
                "commands": [
                    "up --detach --build --remove-orphans",
                    "port gateway 8765",
                    "ps --quiet main",
                    "down --volumes --remove-orphans",
                ],
                "main": {
                    "volumes": [
                        {
                            "type": "bind",
                            "source": "/run/user/docker.sock",
                            "target": "/var/run/docker.sock",
                        },
                        {
                            "type": "bind",
                            "source": "/run/user/docker.sock",
                            "target": "/media/hud/docker.sock",
                        },
                    ]
                },
                "ports": """\
services:
  gateway:
    ports: !override ["127.0.0.1::8765"]
""",
            }
        ),
    ),
    "service access asks docker for its socket when DOCKER_HOST is unset": (
        CHAINED,
        True,
        {},
        None,
        {},
        snapshot(
            {
                "commands": [
                    "context inspect --format {{.Endpoints.docker.Host}}",
                    "up --detach --build --remove-orphans",
                    "port gateway 8765",
                    "ps --quiet main",
                    "down --volumes --remove-orphans",
                ],
                "main": {
                    "volumes": [
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
                    ]
                },
                "ports": """\
services:
  gateway:
    ports: !override ["127.0.0.1::8765"]
""",
            }
        ),
    ),
    "service access through a remote daemon mounts the socket it names": (
        CHAINED,
        True,
        {"compose_service_socket": "/vm/run/docker.sock"},
        None,
        {"DOCKER_HOST": "tcp://docker.remote:2376"},
        snapshot(
            {
                "commands": [
                    "up --detach --build --remove-orphans",
                    "port gateway 8765",
                    "ps --quiet main",
                    "down --volumes --remove-orphans",
                ],
                "main": {
                    "volumes": [
                        {
                            "type": "bind",
                            "source": "/vm/run/docker.sock",
                            "target": "/var/run/docker.sock",
                        },
                        {
                            "type": "bind",
                            "source": "/vm/run/docker.sock",
                            "target": "/media/hud/docker.sock",
                        },
                    ]
                },
                "ports": """\
services:
  gateway:
    ports: !override ["127.0.0.1::8765"]
""",
            }
        ),
    ),
    "a storage request is admitted inside main": (
        CHAINED,
        False,
        {},
        RuntimeConfig(resources=RuntimeResources(storage_mb=1024)),
        {},
        snapshot(
            {
                "commands": [
                    "up --detach --build --remove-orphans",
                    "exec -T main df -Pk /",
                    "port gateway 8765",
                    "ps --quiet main",
                    "down --volumes --remove-orphans",
                ],
                "main": {},
                "ports": """\
services:
  gateway:
    ports: !override ["127.0.0.1::8765"]
""",
            }
        ),
    ),
}


SANDBOX_OPTIONS = ["seccomp=<seccomp>", "systempaths=unconfined", "apparmor=unconfined"]
SESSION_VOLUMES = [
    {"type": "volume", "source": "hud-runtime-sessions", "target": target}
    for target in ("/runtime/sessions", "/media/hud/sessions")
]
COMPOSE_PREFIX = re.compile(r"^compose --project-name \S+ --project-directory \S+ (?:--file \S+ )+")


def launch(fake_docker: FakeDocker, tmp_path: Path) -> dict[str, Any]:
    """What a Compose launch asked for beyond what every launch asks for.

    Every launch runs ``main`` under the sandbox profile with the session volumes
    mounted; this checks that and returns the compose verbs, the rest of the
    override, and the published ports.
    """
    override = staged(fake_docker, tmp_path, "override.json")
    services = dict(override["services"])
    main = dict(services.pop("main"))
    assert main.pop("security_opt") == SANDBOX_OPTIONS
    volumes = main.pop("volumes")
    assert volumes[:2] == SESSION_VOLUMES
    if volumes[2:]:
        main["volumes"] = volumes[2:]
    assert override["volumes"] == {"hud-runtime-sessions": {}}
    return {
        "commands": [
            COMPOSE_PREFIX.sub("", command) for command in transcript(fake_docker, tmp_path)
        ],
        "main": main,
        **({"services": services} if services else {}),
        "ports": staged(fake_docker, tmp_path, "ports.yaml"),
    }


@pytest.mark.parametrize(
    ("document", "service_access", "options", "config", "environment", "launched"),
    COMPOSE.values(),
    ids=COMPOSE.keys(),
)
async def test_a_compose_row_starts_its_project_with_a_provider_override(
    document: str,
    service_access: bool,
    options: dict[str, Any],
    config: RuntimeConfig | None,
    environment: dict[str, str],
    launched: Any,
    fake_docker: FakeDocker,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("DOCKER_HOST", raising=False)
    for name, value in environment.items():
        monkeypatch.setenv(name, value)
    project = ComposeProject(
        document=compose_file(tmp_path, document), service_access=service_access or None
    )
    row = ADD.model_copy(
        update={"runtime_config": RuntimeConfig(compose=project).with_overrides(config)}
    )
    fake_docker.on(r"^context inspect ", stdout="unix:///var/run/docker.sock\n")
    fake_docker.on(r"^compose .* ps --quiet main$", stdout="main-1\n")
    fake_docker.on(r"^compose .* exec -T main df -Pk /$", stdout=DF_FREE)

    async with containers(fake_docker, tmp_path / "rootfs", {"lab": lab()}) as addresses:
        fake_docker.on(r"^compose .* port \w+ 8765$", stdout=f"{addresses['lab']}\n")
        runtime, reward = await run_on(DockerRuntime(**options), row)

    assert reward == 1.0
    assert runtime.config == row.runtime_config
    assert launch(fake_docker, tmp_path) == launched


INTERPOLATED = """
services:
  main:
    image: ${REGISTRY}/lab:${TAG:-latest}
    environment:
      HOME_DIR: '$HOME'
      PRICE: ${PRICE}
      BASE: ${PREFIX}
      LITERAL: $$PATH
      FLAG: true
    entrypoint: python -m hud
    command: serve env.py
    expose: [8765]
    ports: ["127.0.0.1:9000:9000"]
"""

DOCUMENTS = {
    "short syntax is normalized and every service gets its defaults": (
        HEALTHCHECKED,
        "TAG=1\n",
        snapshot(
            {
                "services": {
                    "main": {
                        "image": "lab:1",
                        "environment": {"MODE": "eval", "EMPTY": ""},
                        "command": ["hud", "serve", "env.py", "--port", "8765"],
                        "healthcheck": {"test": ["CMD", "true"]},
                        "expose": [],
                        "ports": [],
                        "volumes": [],
                    },
                    "db": {
                        "image": "postgres:16",
                        "environment": {},
                        "healthcheck": {"test": ["CMD", "pg_isready"], "interval": "1s"},
                        "expose": [],
                        "ports": [],
                        "volumes": [],
                    },
                    "cache": {
                        "image": "redis",
                        "environment": {},
                        "healthcheck": {"test": ["NONE"]},
                        "expose": [],
                        "ports": [],
                        "volumes": [],
                    },
                    "worker": {
                        "image": "worker",
                        "environment": {},
                        "healthcheck": {"disable": True, "test": ["CMD", "true"]},
                        "expose": [],
                        "ports": [],
                        "volumes": [],
                    },
                    "tools": {
                        "image": "tools",
                        "environment": {},
                        "healthcheck": {"test": ["CMD", "true"]},
                        "expose": [],
                        "ports": [],
                        "volumes": [],
                        "profiles": ["debug"],
                    },
                    "idle": {
                        "image": "idle",
                        "environment": {},
                        "healthcheck": {"test": ["CMD", "true"]},
                        "expose": [],
                        "ports": [],
                        "volumes": [],
                        "scale": 0,
                    },
                    "plain": {
                        "image": "plain",
                        "environment": {},
                        "expose": ["7000"],
                        "ports": [
                            {"target": 8080, "protocol": "tcp"},
                            {
                                "target": 9000,
                                "protocol": "udp",
                                "published": 9000,
                                "host_ip": "127.0.0.1",
                            },
                        ],
                        "volumes": [],
                    },
                },
                "networks": {},
            }
        ),
    ),
    "variables come only from the project's .env and literal dollars stay literal": (
        INTERPOLATED,
        "REGISTRY=registry.test\nPRICE=cost$5\nPREFIX=${REGISTRY}/base\n",
        snapshot(
            {
                "services": {
                    "main": {
                        "image": "registry.test/lab:latest",
                        "environment": {
                            "HOME_DIR": "$$HOME",
                            "PRICE": "cost$$5",
                            "BASE": "registry.test/base",
                            "LITERAL": "$$PATH",
                            "FLAG": "true",
                        },
                        "entrypoint": ["python", "-m", "hud"],
                        "command": ["serve", "env.py"],
                        "expose": ["8765"],
                        "ports": [
                            {
                                "target": 9000,
                                "protocol": "tcp",
                                "published": 9000,
                                "host_ip": "127.0.0.1",
                            }
                        ],
                        "volumes": [],
                    }
                },
                "networks": {},
            }
        ),
    ),
}


@pytest.mark.parametrize(
    ("document", "env_file", "normalized"), DOCUMENTS.values(), ids=DOCUMENTS.keys()
)
async def test_a_compose_document_is_staged_normalized_and_self_contained(
    document: str,
    env_file: str,
    normalized: Any,
    fake_docker: FakeDocker,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TAG", "from-the-host")
    monkeypatch.setenv("HOME", "/home/host")
    project = ComposeProject(document=compose_file(tmp_path, document, env_file))
    row = ADD.model_copy(update={"runtime_config": RuntimeConfig(compose=project)})
    fake_docker.on(r"^compose .* port main 8765$", stdout="127.0.0.1:43210\n")

    async with DockerRuntime()(row):
        pass

    assert staged(fake_docker, tmp_path, "compose.json") == normalized


COMPOSE_FAILURES: dict[
    str, tuple[str, bool, dict[str, Any], RuntimeConfig | None, str, list[str]]
] = {
    "service access through a remote daemon without a socket": (
        CHAINED,
        True,
        {"DOCKER_HOST": "tcp://docker.remote:2376"},
        None,
        "DockerRuntime Compose service access through a remote daemon requires "
        "compose_service_socket",
        [],
    ),
    "a GPU type it cannot select": (
        CHAINED,
        False,
        {},
        RuntimeConfig(resources=RuntimeResources(gpu=RuntimeGPU(type="H100"))),
        "DockerRuntime cannot select Compose GPUs by type",
        [],
    ),
    "run args, which only apply to images": (
        CHAINED,
        False,
        {"run_args": ["--privileged"]},
        None,
        "DockerRuntime run_args apply only to image environments",
        [],
    ),
    "a network chain that loops": (
        "services:\n  main: {image: lab, network_mode: 'service:relay'}\n"
        "  relay: {image: relay, network_mode: 'service:main'}\n",
        False,
        {},
        None,
        "Compose network_mode service cycle includes 'main'",
        [],
    ),
    "a network chain to a missing service": (
        "services:\n  main: {image: lab, network_mode: 'service:gone'}\n",
        False,
        {},
        None,
        "Compose network_mode references unknown service 'gone'",
        [],
    ),
    "an include": (
        "include: [other.yaml]\nservices:\n  main: {image: lab}\n",
        False,
        {},
        None,
        "remote adaptation does not support Compose include or extends",
        [],
    ),
    "an extends": (
        "services:\n  main: {image: lab, extends: {file: base.yaml, service: base}}\n",
        False,
        {},
        None,
        "remote adaptation does not support Compose include or extends",
        [],
    ),
    "a variable only the host would set": (
        "services:\n  main: {image: $HOST_ONLY}\n",
        False,
        {},
        None,
        "Compose variable 'HOST_ONLY' is not set by the project .env",
        [],
    ),
    "a required variable left unset": (
        'services:\n  main: {image: "${IMAGE:?set IMAGE in .env}"}\n',
        False,
        {},
        None,
        "set IMAGE in .env",
        [],
    ),
    "a network owner that exits before serving": (
        CHAINED,
        False,
        {},
        None,
        "Compose gateway service exited before serving port 8765:\nImportError: boom",
        snapshot(
            [
                "compose --project-name <project> --project-directory <tmp>/project --file <staged>/compose.json --file <staged>/override.json --file <staged>/ports.yaml up --detach --build --remove-orphans",
                "compose --project-name <project> --project-directory <tmp>/project --file <staged>/compose.json --file <staged>/override.json --file <staged>/ports.yaml port gateway 8765",
                "compose --project-name <project> --project-directory <tmp>/project --file <staged>/compose.json --file <staged>/override.json --file <staged>/ports.yaml logs --tail 40 gateway",
                "compose --project-name <project> --project-directory <tmp>/project --file <staged>/compose.json --file <staged>/override.json --file <staged>/ports.yaml down --volumes --remove-orphans",
            ]
        ),
    ),
}


@pytest.mark.parametrize(
    ("document", "service_access", "change", "config", "error", "commands"),
    COMPOSE_FAILURES.values(),
    ids=COMPOSE_FAILURES.keys(),
)
async def test_a_compose_project_that_cannot_serve_fails_and_is_taken_down(
    document: str,
    service_access: bool,
    change: dict[str, Any],
    config: RuntimeConfig | None,
    error: str,
    commands: list[str],
    fake_docker: FakeDocker,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("DOCKER_HOST", raising=False)
    if "DOCKER_HOST" in change:
        monkeypatch.setenv("DOCKER_HOST", change["DOCKER_HOST"])
    project = ComposeProject(
        document=compose_file(tmp_path, document), service_access=service_access or None
    )
    row = ADD.model_copy(
        update={"runtime_config": RuntimeConfig(compose=project).with_overrides(config)}
    )
    fake_docker.on(r"^compose .* logs --tail 40 gateway$", stderr="ImportError: boom\n")
    provider = DockerRuntime(run_args=change.get("run_args", ()))

    with pytest.raises((ValueError, RuntimeError)) as raised:
        async with provider(row):
            pytest.fail("the acquisition should fail")

    assert str(raised.value) == error
    assert transcript(fake_docker, tmp_path) == commands


async def test_a_platform_compose_document_needs_a_local_file() -> None:
    config = RuntimeConfig.model_validate({"compose": {"document": {"services": {"main": {}}}}})

    with pytest.raises(ValueError, match="DockerRuntime requires compose as a local file path"):
        async with DockerRuntime()(ADD.model_copy(update={"runtime_config": config})):
            pass


async def test_a_container_session_moves_to_the_verifier_unless_it_holds_a_link(
    fake_docker: FakeDocker, tmp_path: Path
) -> None:
    rootfs = tmp_path / "rootfs"
    images = {
        "actor": actor(sessions=rootfs / "actor"),
        "linker": actor(Environment("linker"), sessions=rootfs / "linker", link=True),
        "judge": judge(sessions=rootfs / "judge"),
    }
    verifier = Task(env="judge", id="verify", runtime_config=RuntimeConfig(image="judge"))

    def row(env: str) -> Task:
        return Task(env=env, id="solve", runtime_config=RuntimeConfig(image=env), verifier=verifier)

    async with containers(fake_docker, rootfs, images):
        moved = await rollout(row("actor"), ScriptedAgent("secret"), runtime=DockerRuntime())
        linked = await rollout(row("linker"), ScriptedAgent("secret"), runtime=DockerRuntime())

    assert moved.reward == 1.0
    assert (linked.reward, linked.trace.error) == (
        0.0,
        IsStr(
            regex=r"(?s)\[snapshotting actor\] RuntimeError: docker exec --user 0 linker python3 .*"
            r"ValueError: runtime session contains a symbolic link"
        ),
    )
    assert [
        command
        for command in transcript(fake_docker, tmp_path)
        if command.startswith(("exec", "cp"))
    ] == snapshot(
        [
            'exec actor sh -c if [ -d "$1" ]; then printf 1; fi hud-session /runtime/sessions/<session>',
            "exec --user 0 actor python3 -c <script> /runtime/sessions/<session> <export>.tar.gz",
            "cp actor:<export>.tar.gz <local>/session.tar.gz",
            "exec --user 0 actor rm -f <export>.tar.gz",
            "exec --user 0 judge rm -rf /runtime/sessions/<session>",
            "exec --user 0 judge mkdir -p /runtime/sessions/<session>",
            "cp <local>/. judge:/runtime/sessions/<session>",
            'exec linker sh -c if [ -d "$1" ]; then printf 1; fi hud-session /runtime/sessions/<session>',
            "exec --user 0 linker python3 -c <script> /runtime/sessions/<session> <export>.tar.gz",
            "exec --user 0 linker rm -f <export>.tar.gz",
        ]
    )


async def test_a_restored_session_archive_may_hold_only_files_and_directories(
    fake_docker: FakeDocker, tmp_path: Path
) -> None:
    archive = tmp_path / "session.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        data = b"secret"
        work = tarfile.TarInfo("work.txt")
        work.size = len(data)
        tar.addfile(work, io.BytesIO(data))
        link = tarfile.TarInfo("escape")
        link.type = tarfile.SYMTYPE
        link.linkname = "/etc/passwd"
        tar.addfile(link)

    async with (
        containers(fake_docker, tmp_path / "rootfs", {"lab:1": lab()}),
        DockerRuntime("lab:1")(ADD) as runtime,
    ):
        with pytest.raises(
            ValueError, match="runtime session archive contains an unsupported entry"
        ):
            await runtime.restore_session("sess-12345678", archive)

    assert not any(command.startswith(("exec", "cp")) for command in fake_docker.commands())
