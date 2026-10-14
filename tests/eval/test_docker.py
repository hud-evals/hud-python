"""DockerRuntime: a row runs in a confined container or Compose project that is removed after.

The fake ``docker`` on ``PATH`` records every invocation and runs ``cp`` and
``exec`` against a directory per container; its ``port`` answer points at an
environment served in this process, so each placement really serves a rollout
whose reward proves the container worked.
"""

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
from tests.eval.envs import DF_FREE, actor, containers, judge, lab, solve
from tests.harness import ScriptedAgent, eventually

if TYPE_CHECKING:
    from hud.eval import Runtime
    from tests.harness import FakeDocker

SECCOMP = Path(hud.__file__).parent / "eval" / "docker-seccomp.json"
CONFINED = [f"seccomp={SECCOMP}", "systempaths=unconfined", "apparmor=unconfined"]
ADD = Task(env="lab", id="add", args={"a": 2, "b": 3})


def placed(config: RuntimeConfig | None) -> Task:
    return ADD.model_copy(update={"runtime_config": config})


async def run_on(provider: DockerRuntime, row: Task) -> tuple[Runtime, float]:
    """Acquire ``row``'s placement, run the rollout on it, and return both."""
    async with provider(row) as runtime:
        run = await rollout(row, ScriptedAgent(solve), runtime=runtime)
    return runtime, run.reward


def docker_run(fake_docker: FakeDocker) -> list[str]:
    (argv,) = [call.argv for call in fake_docker.calls if call.argv[:1] == ["run"]]
    return argv


def volumes_left(fake_docker: FakeDocker) -> set[str]:
    """Session volumes created and never removed."""
    created = {call.argv[2] for call in fake_docker.calls if call.argv[:2] == ["volume", "create"]}
    removed = {call.argv[-1] for call in fake_docker.calls if call.argv[:2] == ["volume", "rm"]}
    return created - removed


async def test_an_image_row_runs_in_a_confined_container_that_is_removed_after(
    fake_docker: FakeDocker, tmp_path: Path
) -> None:
    async with containers(fake_docker, tmp_path / "rootfs", {"lab:1": lab()}):
        _, reward = await run_on(DockerRuntime("lab:1"), ADD)

    argv = docker_run(fake_docker)
    assert reward == 1.0
    assert [argv[i + 1] for i, arg in enumerate(argv) if arg == "--security-opt"] == CONFINED
    assert "--privileged" not in argv
    assert fake_docker.commands("rm --force ") == ["rm --force lab-1"]
    assert volumes_left(fake_docker) == set()


# What the row and the provider each set, and which wins, as docker run sees it.
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
    async with containers(fake_docker, tmp_path / "rootfs", {"lab:1": lab()}):
        runtime, reward = await run_on(DockerRuntime(**options), placed(config))

    argv = docker_run(fake_docker)
    assert reward == 1.0
    assert (argv[2 : argv.index("--security-opt")], argv[-1]) == (run_options, "lab:1")
    assert runtime.params == params


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
IMAGE_FAILURES: dict[str, tuple[RuntimeConfig, list[tuple[str, dict[str, Any]]], Any, bool]] = {
    "a GPU type it cannot select": (
        RuntimeConfig(image="lab:1", resources=RuntimeResources(gpu=RuntimeGPU(type="H100"))),
        [],
        "DockerRuntime cannot select GPUs by type",
        False,
    ),
    "no image at all": (
        RuntimeConfig(resources=RuntimeResources(cpu=1)),
        [],
        "DockerRuntime requires runtime_config.image or runtime_config.compose",
        False,
    ),
    "a Compose document that is not a local file": (
        RuntimeConfig.model_validate({"compose": {"document": {"services": {"main": {}}}}}),
        [],
        "DockerRuntime requires compose as a local file path",
        False,
    ),
    "a container that exits before serving": (
        RuntimeConfig(image="lab:1"),
        [(r"^logs --tail 40 ", {"stderr": "ImportError: boom\n"})],
        "container for image 'lab:1' exited before serving port 8765:\nImportError: boom",
        True,
    ),
    "less free disk than requested": (
        RuntimeConfig(image="lab:1", resources=RuntimeResources(storage_mb=8192)),
        [(r"^exec \S+ df -Pk /$", {"stdout": LOW_DISK})],
        "DockerRuntime requires 8192 MB of free disk; the environment has 4096 MB",
        True,
    ),
    "free disk it cannot read": (
        RuntimeConfig(image="lab:1", resources=RuntimeResources(storage_mb=8192)),
        [(r"^exec \S+ df -Pk /$", {"stdout": "df: not found\n"})],
        "DockerRuntime could not measure the environment's free disk",
        True,
    ),
    "a start slower than its startup limit": (
        RuntimeConfig(image="lab:1", limits=RuntimeLimits(startup_timeout_s=1)),
        [(r"^run ", {"stdout": "cid-1\n", "delay": 3})],
        IsStr(regex=r"docker run --detach .* lab:1 timed out after 1s"),
        False,
    ),
}


@pytest.mark.parametrize(
    ("config", "rules", "error", "started"), IMAGE_FAILURES.values(), ids=IMAGE_FAILURES.keys()
)
async def test_an_image_that_cannot_serve_fails_its_acquisition_and_leaves_nothing_behind(
    config: RuntimeConfig,
    rules: list[tuple[str, dict[str, Any]]],
    error: Any,
    started: bool,
    fake_docker: FakeDocker,
) -> None:
    for match, answer in rules:
        fake_docker.on(match, **answer)

    with pytest.raises((ValueError, RuntimeError)) as raised:
        async with DockerRuntime()(placed(config)):
            pytest.fail("the acquisition should fail")

    assert str(raised.value) == error
    assert fake_docker.commands("rm --force ") == (["rm --force cid-1"] if started else [])
    assert volumes_left(fake_docker) == set()
    if not rules:
        assert fake_docker.calls == []


async def test_a_container_past_its_run_limit_is_removed(fake_docker: FakeDocker) -> None:
    fake_docker.on(r"^port ", stdout="127.0.0.1:43210\n")
    row = placed(RuntimeConfig(image="lab:1", limits=RuntimeLimits(run_timeout_s=1)))

    async with DockerRuntime()(row):
        await asyncio.sleep(1.2)
        removed_while_held = fake_docker.commands("rm --force ")

    assert removed_while_held == ["rm --force cid-1"]


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


def staged(fake_docker: FakeDocker, name: str) -> Any:
    """The staged Compose file ``name`` as ``compose up`` read it."""
    (text,) = {
        text
        for call in fake_docker.calls
        if " up --detach " in call.command
        for path, text in call.files.items()
        if path.endswith(name)
    }
    return json.loads(text)


HEALTHCHECKED = """
services:
  main:
    image: lab:${TAG}
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

SOCKET = "/var/run/docker.sock"
COMPOSE: dict[str, tuple[str, bool, dict[str, Any], RuntimeConfig | None, dict[str, str], Any]] = {
    "only services with a live healthcheck are awaited; resources land on main": (
        HEALTHCHECKED,
        False,
        {"env_vars": {"MODE": "override"}, "bind_mounts": [DockerBindMount("/opt/data", "/data")]},
        RuntimeConfig(resources=RuntimeResources(cpu=2, memory_mb=4096, gpu=RuntimeGPU(count=2))),
        {},
        {
            "awaited": ["main", "db"],
            "published": "main",
            "main": {"environment": {"MODE": "override"}, "cpus": 2.0, "mem_limit": "4096m"},
            "mounts": ["/opt/data"],
        },
    ),
    "the published port belongs to the end of main's network chain": (
        CHAINED,
        False,
        {},
        None,
        {},
        {"awaited": [], "published": "gateway", "main": {}, "mounts": []},
    ),
    "service access mounts the local daemon's socket": (
        CHAINED,
        True,
        {},
        None,
        {"DOCKER_HOST": "unix:///run/user/docker.sock"},
        {
            "awaited": [],
            "published": "gateway",
            "main": {},
            "mounts": ["/run/user/docker.sock"] * 2,
        },
    ),
    "service access asks docker for its socket when DOCKER_HOST is unset": (
        CHAINED,
        True,
        {},
        None,
        {},
        {"awaited": [], "published": "gateway", "main": {}, "mounts": [SOCKET] * 2},
    ),
    "service access through a remote daemon mounts the socket it names": (
        CHAINED,
        True,
        {"compose_service_socket": "/vm/run/docker.sock"},
        None,
        {"DOCKER_HOST": "tcp://docker.remote:2376"},
        {"awaited": [], "published": "gateway", "main": {}, "mounts": ["/vm/run/docker.sock"] * 2},
    ),
}


@pytest.mark.parametrize(
    ("document", "service_access", "options", "config", "environment", "launched"),
    COMPOSE.values(),
    ids=COMPOSE.keys(),
)
async def test_a_compose_row_runs_its_project_with_main_confined_and_takes_it_down(
    document: str,
    service_access: bool,
    options: dict[str, Any],
    config: RuntimeConfig | None,
    environment: dict[str, str],
    launched: dict[str, Any],
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
    row = placed(RuntimeConfig(compose=project).with_overrides(config))
    fake_docker.on(r"^context inspect ", stdout=f"unix://{SOCKET}\n")
    fake_docker.on(r"^compose .* ps --quiet main$", stdout="main-1\n")

    async with containers(fake_docker, tmp_path / "rootfs", {"lab": lab()}) as addresses:
        fake_docker.on(r"^compose .* port \w+ 8765$", stdout=f"{addresses['lab']}\n")
        _, reward = await run_on(DockerRuntime(**options), row)

    main = staged(fake_docker, "override.json")["services"]["main"]
    awaited = [call.argv for call in fake_docker.calls if "--wait" in call.argv]
    (published,) = [call.argv[-2] for call in fake_docker.calls if call.argv[-3:-2] == ["port"]]
    session_volumes = [v for v in main["volumes"] if v["type"] == "volume"]
    assert reward == 1.0
    assert main["security_opt"] == CONFINED
    assert [v["target"] for v in session_volumes] == ["/runtime/sessions", "/media/hud/sessions"]
    assert {
        "awaited": awaited[0][awaited[0].index("--no-build") + 1 :] if awaited else [],
        "published": published,
        "main": {key: main[key] for key in ("environment", "cpus", "mem_limit") if key in main},
        "mounts": [v["source"] for v in main["volumes"] if v["type"] == "bind"],
    } == launched
    assert fake_docker.commands()[-1].endswith("down --volumes --remove-orphans")


async def test_a_compose_storage_request_is_admitted_inside_main(
    fake_docker: FakeDocker, tmp_path: Path
) -> None:
    project = ComposeProject(document=compose_file(tmp_path, CHAINED))
    row = placed(RuntimeConfig(compose=project, resources=RuntimeResources(storage_mb=1024)))
    fake_docker.on(r"^compose .* exec -T main df -Pk /$", stdout=DF_FREE)

    async with containers(fake_docker, tmp_path / "rootfs", {"lab": lab()}) as addresses:
        fake_docker.on(r"^compose .* port \w+ 8765$", stdout=f"{addresses['lab']}\n")
        _, reward = await run_on(DockerRuntime(), row)

    assert reward == 1.0
    assert [command for command in fake_docker.commands() if "df -Pk" in command] != []


INTERPOLATED = """
services:
  main:
    image: ${REGISTRY}/lab:${TAG:-latest}
    environment:
      HOME_DIR: '$HOME'
      PRICE: ${PRICE}
      BASE: ${PREFIX}
      LITERAL: $$PATH
"""


async def test_compose_variables_come_only_from_the_projects_env_file(
    fake_docker: FakeDocker, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TAG", "from-the-host")
    monkeypatch.setenv("HOME", "/home/host")
    env_file = "REGISTRY=registry.test\nPRICE=cost$5\nPREFIX=${REGISTRY}/base\n"
    project = ComposeProject(document=compose_file(tmp_path, INTERPOLATED, env_file))
    fake_docker.on(r"^compose .* port main 8765$", stdout="127.0.0.1:43210\n")

    async with DockerRuntime()(placed(RuntimeConfig(compose=project))):
        pass

    main = staged(fake_docker, "compose.json")["services"]["main"]
    assert main["image"] == "registry.test/lab:latest"
    assert main["environment"] == {
        "HOME_DIR": "$$HOME",
        "PRICE": "cost$$5",
        "BASE": "registry.test/base",
        "LITERAL": "$$PATH",
    }


COMPOSE_FAILURES: dict[str, tuple[str, bool, dict[str, Any], RuntimeConfig | None, str, bool]] = {
    "service access through a remote daemon without a socket": (
        CHAINED,
        True,
        {"DOCKER_HOST": "tcp://docker.remote:2376"},
        None,
        "service access through a remote daemon requires compose_service_socket",
        False,
    ),
    "a GPU type it cannot select": (
        CHAINED,
        False,
        {},
        RuntimeConfig(resources=RuntimeResources(gpu=RuntimeGPU(type="H100"))),
        "cannot select Compose GPUs by type",
        False,
    ),
    "run args, which only apply to images": (
        CHAINED,
        False,
        {"run_args": ["--privileged"]},
        None,
        "run_args apply only to image environments",
        False,
    ),
    "a network chain that loops": (
        "services:\n  main: {image: lab, network_mode: 'service:relay'}\n"
        "  relay: {image: relay, network_mode: 'service:main'}\n",
        False,
        {},
        None,
        "network_mode service cycle includes 'main'",
        False,
    ),
    "a network chain to a missing service": (
        "services:\n  main: {image: lab, network_mode: 'service:gone'}\n",
        False,
        {},
        None,
        "network_mode references unknown service 'gone'",
        False,
    ),
    "an include": (
        "include: [other.yaml]\nservices:\n  main: {image: lab}\n",
        False,
        {},
        None,
        "does not support Compose include or extends",
        False,
    ),
    "an extends": (
        "services:\n  main: {image: lab, extends: {file: base.yaml, service: base}}\n",
        False,
        {},
        None,
        "does not support Compose include or extends",
        False,
    ),
    "a variable only the host would set": (
        "services:\n  main: {image: $HOST_ONLY}\n",
        False,
        {},
        None,
        "variable 'HOST_ONLY' is not set by the project .env",
        False,
    ),
    "a required variable left unset": (
        'services:\n  main: {image: "${IMAGE:?set IMAGE in .env}"}\n',
        False,
        {},
        None,
        "set IMAGE in .env",
        False,
    ),
    "a network owner that exits before serving": (
        CHAINED,
        False,
        {},
        None,
        "gateway service exited before serving port 8765:\nImportError: boom",
        True,
    ),
}


@pytest.mark.parametrize(
    ("document", "service_access", "change", "config", "error", "started"),
    COMPOSE_FAILURES.values(),
    ids=COMPOSE_FAILURES.keys(),
)
async def test_a_compose_project_that_cannot_serve_fails_and_is_taken_down(
    document: str,
    service_access: bool,
    change: dict[str, Any],
    config: RuntimeConfig | None,
    error: str,
    started: bool,
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
    row = placed(RuntimeConfig(compose=project).with_overrides(config))
    fake_docker.on(r"^compose .* logs --tail 40 gateway$", stderr="ImportError: boom\n")

    with pytest.raises((ValueError, RuntimeError), match=re.escape(error)):
        async with DockerRuntime(run_args=change.get("run_args", ()))(row):
            pytest.fail("the acquisition should fail")

    commands = fake_docker.commands()
    if started:
        assert commands[-1].endswith("down --volumes --remove-orphans")
    else:
        assert not any(" up " in command for command in commands)


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
        IsStr(regex=r"(?s)\[snapshotting actor\] .*runtime session contains a symbolic link"),
    )
    assert not any(command.startswith("cp linker:") for command in fake_docker.commands())


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
