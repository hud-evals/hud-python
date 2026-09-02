"""``DockerRuntime()`` provider behavior, driven through a scripted docker CLI.

No daemon needed: a fake ``docker`` executable on PATH records every
invocation and scripts the responses, so these tests pin the provider's
contract — command shape, runtime address, teardown — at the process
boundary.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import sys
import tarfile
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import pytest

import hud.eval.runtime.docker as runtime_module
import hud.utils.process as process_module
from hud.eval.runtime import (
    DaytonaRuntime,
    DockerRuntime,
    ModalRuntime,
    RuntimeConfig,
    RuntimeGPU,
    RuntimeLimits,
    RuntimeResources,
    Shared,
)
from hud.eval.runtime.compose import (
    ComposeConfig,
    ComposeHealthcheck,
    ComposeProject,
    ComposeService,
)
from hud.eval.task import Task

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from daytona import Image as DaytonaImage

FAKE_DOCKER_SH = """\
#!/bin/sh
echo "$@" >> "$DOCKER_LOG"
case "$1" in
  run) echo cid-42 ;;
  port) {port_behavior} ;;
  exec) if [ "$3" = "df" ]; then
    printf 'Filesystem 1024-blocks Used Available Capacity Mounted on\n'
    printf 'overlay 4194304 0 4194304 0%% /\n'
  fi ;;
  logs) echo "ImportError: boom" ;;
esac
"""

FAKE_DOCKER_CMD = """\
@echo off
echo %*>>"%DOCKER_LOG%"
if "%1"=="run" (
  echo cid-42
  exit /b 0
)
if "%1"=="port" (
  {port_behavior}
  exit /b 0
)
if "%1"=="logs" (
  echo ImportError: boom
  exit /b 0
)
exit /b 0
"""


def _port_behavior_for_windows(port_behavior: str) -> str:
    if port_behavior == "echo 127.0.0.1:43210":
        return "echo 127.0.0.1:43210"
    if port_behavior == ":":
        return "rem noop"
    raise ValueError(f"unsupported port_behavior: {port_behavior!r}")


async def _docker_via(fake_exe: Path, *args: str, check: bool = True) -> tuple[str, str]:
    proc = await asyncio.create_subprocess_exec(
        str(fake_exe),
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    out, err = await proc.communicate()
    if check and proc.returncode != 0:
        detail = err.decode("utf-8", "replace").strip() or out.decode("utf-8", "replace").strip()
        raise RuntimeError(f"docker {' '.join(args)} failed ({proc.returncode}): {detail}")
    return out.decode("utf-8", "replace"), err.decode("utf-8", "replace")


@pytest.fixture
def docker_log(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    log = tmp_path / "docker.log"
    log.touch()
    monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("DOCKER_LOG", str(log))
    return log


def _install_fake_docker(
    tmp_path: Path,
    *,
    port_behavior: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if sys.platform == "win32":
        exe = tmp_path / "docker.cmd"
        exe.write_text(
            FAKE_DOCKER_CMD.format(port_behavior=_port_behavior_for_windows(port_behavior))
        )
        import hud.eval.runtime.docker as runtime_module

        async def _docker(*args: str, check: bool = True) -> tuple[str, str]:
            return await _docker_via(exe, *args, check=check)

        monkeypatch.setattr(runtime_module, "_docker", _docker)
        return

    exe = tmp_path / "docker"
    exe.write_text(FAKE_DOCKER_SH.format(port_behavior=port_behavior))
    exe.chmod(0o755)


def _row() -> Task:
    return Task(env="any-env", id="t")


async def _docker_calls(docker_log: Path) -> list[str]:
    return (await asyncio.to_thread(docker_log.read_text)).splitlines()


def _docker_security_args() -> str:
    return " ".join(runtime_module._DOCKER_SECURITY_ARGS)


@dataclass(frozen=True)
class _ModalImageRef:
    kind: str
    name: str
    env_vars: dict[str, str] | None = None

    def env(self, vars: dict[str, str]) -> _ModalImageRef:
        return _ModalImageRef(self.kind, self.name, dict(vars))


@dataclass(frozen=True)
class _ModalSecretRef:
    name: str


class _FakeModalStream:
    def __init__(self, calls: dict[str, Any], key: str) -> None:
        self._calls = calls
        self._key = key

    def __aiter__(self) -> AsyncIterator[str]:
        return self._iterate()

    async def _iterate(self) -> AsyncIterator[str]:
        output = cast("str", self._calls.get(self._key, ""))
        if output:
            yield output
        wait = self._calls.get(f"{self._key}_wait")
        if isinstance(wait, asyncio.Event):
            await wait.wait()


class _FakeModalSandbox:
    object_id = "sb-1"

    def __init__(self, calls: dict[str, Any], port: int) -> None:
        self._calls = calls
        self._port = port
        self.wait_until_ready = SimpleNamespace(aio=self._wait_until_ready)
        self.tunnels = SimpleNamespace(aio=self._tunnels)
        self.terminate = SimpleNamespace(aio=self._terminate)
        self.stdout = _FakeModalStream(calls, "sandbox_stdout")
        self.stderr = _FakeModalStream(calls, "sandbox_stderr")
        self.filesystem = SimpleNamespace(
            copy_from_local=SimpleNamespace(aio=self._copy_from_local),
            copy_to_local=SimpleNamespace(aio=self._copy_to_local),
        )
        self.exec = SimpleNamespace(aio=self._exec)

    async def _wait_until_ready(self, **kwargs: object) -> None:
        self._calls["ready_timeout"] = kwargs["timeout"]

    async def _tunnels(self) -> dict[int, SimpleNamespace]:
        return {self._port: SimpleNamespace(tcp_socket=("modal.host", 4567))}

    async def _terminate(self) -> None:
        self._calls["terminated"] = True

    async def _copy_from_local(self, source: Path, target: str) -> None:
        uploads = self._calls.setdefault("uploads", [])
        assert isinstance(uploads, list)
        uploads.append((source.name, target))
        if source.name == "override.json":
            content = await asyncio.to_thread(source.read_text, "utf-8")
            self._calls["compose_override"] = json.loads(content)

    async def _copy_to_local(self, source: str, target: Path) -> None:
        downloads = self._calls.setdefault("downloads", [])
        assert isinstance(downloads, list)
        downloads.append((source, target.name))

    async def _exec(self, *args: str, **kwargs: object) -> SimpleNamespace:
        commands = self._calls.setdefault("execs", [])
        assert isinstance(commands, list)
        commands.append((args, kwargs))

        async def wait() -> int:
            wait_event = self._calls.pop("exec_wait", None)
            if isinstance(wait_event, asyncio.Event):
                await wait_event.wait()
            return 0

        async def read_stderr() -> str:
            return ""

        async def read_stdout() -> str:
            command = args[-1]
            return (
                "1"
                if "sessions/sess-actor" in command
                and ("test -d" in command or "if [ -d" in command)
                else ""
            )

        if "logs" in args:
            return SimpleNamespace(
                wait=SimpleNamespace(aio=wait),
                stdout=_FakeModalStream(self._calls, "compose_stdout"),
                stderr=_FakeModalStream(self._calls, "compose_stderr"),
            )

        return SimpleNamespace(
            wait=SimpleNamespace(aio=wait),
            stdout=SimpleNamespace(read=SimpleNamespace(aio=read_stdout)),
            stderr=SimpleNamespace(read=SimpleNamespace(aio=read_stderr)),
        )


def _install_fake_modal(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    calls: dict[str, Any] = {}
    modal = ModuleType("modal")

    class Image:
        @staticmethod
        def from_name(name: str) -> _ModalImageRef:
            calls["image_name"] = name
            return _ModalImageRef("name", name)

        @staticmethod
        def from_registry(name: str, secret: object = None) -> _ModalImageRef:
            calls["registry_image"] = name
            calls["registry_secret"] = secret
            return _ModalImageRef("registry", name)

        @staticmethod
        def from_id(image_id: str) -> _ModalImageRef:
            calls["modal_image_id"] = image_id
            return _ModalImageRef("id", image_id)

    async def lookup(app_name: str, *, create_if_missing: bool) -> str:
        calls["app_lookup"] = (app_name, create_if_missing)
        return "app"

    async def create(*args: str, **kwargs: object) -> _FakeModalSandbox:
        calls["sandbox_args"] = args
        calls["sandbox_kwargs"] = kwargs
        ports = kwargs["unencrypted_ports"]
        assert isinstance(ports, list)
        port = ports[0]
        assert isinstance(port, int)
        return _FakeModalSandbox(calls, port)

    setattr(modal, "Image", Image)
    setattr(modal, "App", SimpleNamespace(lookup=SimpleNamespace(aio=lookup)))
    setattr(modal, "Probe", SimpleNamespace(with_tcp=lambda port: ("tcp", port)))
    setattr(modal, "Sandbox", SimpleNamespace(create=SimpleNamespace(aio=create)))
    monkeypatch.setitem(sys.modules, "modal", modal)
    return calls


@dataclass(frozen=True)
class _CreateSandboxFromSnapshotParams:
    snapshot: str
    ephemeral: bool
    auto_stop_interval: int
    name: str | None = None
    env_vars: dict[str, str] | None = None
    linked_sandbox: str | None = None


@dataclass(frozen=True)
class _CreateSnapshotParams:
    name: str
    image: object
    resources: object | None = None


@dataclass(frozen=True)
class _CreateSandboxFromImageParams:
    image: object
    ephemeral: bool
    auto_stop_interval: int
    resources: object | None = None
    name: str | None = None
    env_vars: dict[str, str] | None = None
    linked_sandbox: str | None = None


@dataclass(frozen=True)
class _DaytonaImage:
    name: str


@dataclass(frozen=True)
class _DaytonaResources:
    cpu: float | None = None
    memory: int | None = None
    disk: int | None = None
    gpu: int | None = None
    gpu_type: list[object] | None = None


@dataclass(frozen=True)
class _DaytonaGpuType:
    value: str


@dataclass(frozen=True)
class _SessionExecuteRequest:
    command: str
    run_async: bool


class _FakeDaytonaProcess:
    def __init__(self, calls: dict[str, Any], sandbox_id: str) -> None:
        self._calls = calls
        self._sandbox_id = sandbox_id
        self.logs = SimpleNamespace(
            stderr="ImportError: no module named bugs", output="", stdout=""
        )

    async def create_session(self, session: str) -> None:
        self._calls["session"] = session

    async def execute_session_command(self, session: str, request: object) -> SimpleNamespace:
        self._calls["execute"] = (session, request)
        commands = self._calls.setdefault("session_commands", [])
        assert isinstance(commands, list)
        commands.append((self._sandbox_id, session, request))
        return SimpleNamespace(cmd_id="cmd-1")

    async def get_session_command_logs(self, session: str, cmd_id: str) -> SimpleNamespace:
        self._calls["logs"] = (session, cmd_id)
        return self.logs

    async def get_session_command_logs_async(
        self,
        session: str,
        cmd_id: str,
        on_stdout: Any,
        on_stderr: Any,
    ) -> None:
        self._calls["stream_logs"] = (session, cmd_id)
        on_stdout(cast("str", self._calls.get("daytona_stdout", "")))
        on_stderr(cast("str", self._calls.get("daytona_stderr", "")))
        wait = self._calls.get("daytona_log_wait")
        if isinstance(wait, asyncio.Event):
            await wait.wait()

    async def exec(self, command: str, **kwargs: object) -> SimpleNamespace:
        commands = self._calls.setdefault("process_execs", [])
        assert isinstance(commands, list)
        commands.append((self._sandbox_id, command, kwargs))
        return SimpleNamespace(exit_code=0, result="")


class _FakeDaytonaSandbox:
    def __init__(self, calls: dict[str, Any], sandbox_id: str) -> None:
        self.id = sandbox_id
        self._calls = calls
        self.process = _FakeDaytonaProcess(calls, sandbox_id)
        self.fs = SimpleNamespace(upload_file=self._upload_file)

    async def _upload_file(self, source: str, target: str) -> None:
        uploads = self._calls.setdefault("uploads", [])
        assert isinstance(uploads, list)
        uploads.append((Path(source).name, target))

    async def create_ssh_access(self, *, expires_in_minutes: int) -> SimpleNamespace:
        self._calls["ssh_expires"] = expires_in_minutes
        return SimpleNamespace(token="ssh-token")


def _tree_hash_sync(path_str: str) -> str:
    """Deterministic fingerprint of a context tree: same tree, same hash."""
    digest = hashlib.md5(usedforsecurity=False)
    root = Path(path_str)
    for file in sorted(p for p in root.rglob("*") if p.is_file()):
        digest.update(file.relative_to(root).as_posix().encode())
        digest.update(file.read_bytes())
    return digest.hexdigest()


async def _tree_hash(path_str: str) -> str:
    return await asyncio.to_thread(_tree_hash_sync, path_str)


class _FakeObjectStorage:
    """The SDK hasher the provider borrows to predict a context's upload hash."""

    async def _compute_hash_for_path_md5(
        self, path_str: str, archive_base_path: str | None = None
    ) -> str:
        return await _tree_hash(path_str)


class _FakeSnapshotApi:
    """The registry a Daytona snapshot name resolves against: ``get`` 404s until
    something with that exact name has been built, and each snapshot records what
    it was built from (``image_name`` or ``build_info``), as the server does.
    A deleted name stays taken for a couple more ``get``s — the live API frees
    it asynchronously (~10s), and creating before then conflicts."""

    def __init__(self) -> None:
        self.snapshots: dict[str, SimpleNamespace] = {}
        self.builds: list[str] = []
        self._deleting: dict[str, SimpleNamespace] = {}
        self._deleting_gets: dict[str, int] = {}

    async def get(self, name: str) -> SimpleNamespace:
        if name in self._deleting:
            self._deleting_gets[name] -= 1
            if self._deleting_gets[name] >= 0:
                return self._deleting[name]
            del self._deleting[name], self._deleting_gets[name]
        if name not in self.snapshots:
            raise RuntimeError(f"snapshot {name} not found")
        return self.snapshots[name]

    async def create(self, params: _CreateSnapshotParams) -> None:
        if params.name in self.snapshots or params.name in self._deleting:
            raise ValueError(f"snapshot with name {params.name} already exists")
        image: Any = params.image
        if isinstance(image, str):
            record = SimpleNamespace(name=params.name, image_name=image, build_info=None)
        else:
            record = SimpleNamespace(
                name=params.name,
                image_name=None,
                build_info=SimpleNamespace(
                    dockerfile_content=image.dockerfile(),
                    context_hashes=[
                        await _tree_hash(entry.source_path) for entry in image._context_list
                    ],
                ),
            )
        self.snapshots[params.name] = record
        self.builds.append(params.name)

    async def delete(self, snapshot: SimpleNamespace) -> None:
        self._deleting[snapshot.name] = self.snapshots.pop(snapshot.name)
        self._deleting_gets[snapshot.name] = 1


class _FakeDaytonaClient:
    def __init__(self, calls: dict[str, Any]) -> None:
        self.calls = calls
        self.sandbox = _FakeDaytonaSandbox(calls, "sandbox-1")
        self.snapshot = _FakeSnapshotApi()
        #: sandbox-create params in order; the last entry is the booted sandbox.
        self.created: list[Any] = []
        self.delete_fails = False

    async def create(self, params: object, **kwargs: object) -> _FakeDaytonaSandbox:
        self.calls["create"] = (params, kwargs["timeout"])
        self.created.append(params)
        sandbox = (
            self.sandbox
            if len(self.created) == 1
            else _FakeDaytonaSandbox(self.calls, f"sandbox-{len(self.created)}")
        )
        return sandbox

    async def delete(self, sandbox: _FakeDaytonaSandbox) -> None:
        if self.delete_fails:
            raise RuntimeError("daytona API unreachable")
        self.calls["delete"] = sandbox.id
        deleted = self.calls.setdefault("deleted", [])
        assert isinstance(deleted, list)
        deleted.append(sandbox.id)


class _FakeSSHConnection:
    def __init__(self, calls: dict[str, Any]) -> None:
        self._calls = calls

    async def forward_local_port(
        self,
        listen_host: str,
        listen_port: int,
        dest_host: str,
        dest_port: int,
    ) -> SimpleNamespace:
        self._calls["forward"] = (listen_host, listen_port, dest_host, dest_port)
        return SimpleNamespace(get_port=lambda: 54321)


class _FakeSSHConnect:
    def __init__(
        self,
        calls: dict[str, Any],
        args: tuple[object, ...],
        kwargs: dict[str, object],
    ) -> None:
        self._calls = calls
        self._args = args
        self._kwargs = kwargs

    async def __aenter__(self) -> _FakeSSHConnection:
        self._calls["ssh_connect"] = (self._args, self._kwargs)
        return _FakeSSHConnection(self._calls)

    async def __aexit__(self, *exc_info: object) -> None:
        self._calls["ssh_closed"] = True


def _install_fake_daytona(monkeypatch: pytest.MonkeyPatch) -> _FakeDaytonaClient:
    calls: dict[str, Any] = {}
    client = _FakeDaytonaClient(calls)
    daytona = ModuleType("daytona")
    daytona_async = ModuleType("daytona._async")
    object_storage = ModuleType("daytona._async.object_storage")
    asyncssh = ModuleType("asyncssh")

    class AsyncDaytona:
        async def __aenter__(self) -> _FakeDaytonaClient:
            return client

        async def __aexit__(self, *exc_info: object) -> None:
            calls["client_closed"] = True

    def connect(*args: object, **kwargs: object) -> _FakeSSHConnect:
        return _FakeSSHConnect(calls, args, kwargs)

    setattr(daytona, "AsyncDaytona", AsyncDaytona)
    setattr(daytona, "CreateSnapshotParams", _CreateSnapshotParams)
    setattr(daytona, "CreateSandboxFromSnapshotParams", _CreateSandboxFromSnapshotParams)
    setattr(daytona, "CreateSandboxFromImageParams", _CreateSandboxFromImageParams)
    setattr(daytona, "DaytonaNotFoundError", RuntimeError)
    setattr(daytona, "Image", SimpleNamespace(base=lambda name: _DaytonaImage(name)))
    setattr(daytona, "Resources", _DaytonaResources)
    setattr(daytona, "GpuType", _DaytonaGpuType)
    setattr(daytona, "SessionExecuteRequest", _SessionExecuteRequest)
    setattr(daytona, "_async", daytona_async)
    setattr(daytona_async, "object_storage", object_storage)
    setattr(object_storage, "AsyncObjectStorage", _FakeObjectStorage)
    setattr(asyncssh, "connect", connect)
    monkeypatch.setitem(sys.modules, "daytona", daytona)
    monkeypatch.setitem(sys.modules, "daytona._async", daytona_async)
    monkeypatch.setitem(sys.modules, "daytona._async.object_storage", object_storage)
    monkeypatch.setitem(sys.modules, "asyncssh", asyncssh)
    return client


async def test_acquisition_publishes_ephemeral_port_and_removes_container(
    tmp_path: Path,
    docker_log: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _install_fake_docker(tmp_path, port_behavior="echo 127.0.0.1:43210", monkeypatch=monkeypatch)

    provider = DockerRuntime("img:tag", run_args=("-e", "X=1"))
    async with provider(_row()) as runtime:
        assert runtime.url == "tcp://127.0.0.1:43210"
        await asyncio.sleep(0.05)
        calls = await _docker_calls(docker_log)
        assert calls[0] == (
            f"run --detach -e X=1 {_docker_security_args()} --publish 127.0.0.1::8765 img:tag"
        )
        assert calls[1] == "port cid-42 8765"

    assert (await _docker_calls(docker_log))[-1] == "rm --force cid-42"
    assert capsys.readouterr().out == "ImportError: boom\n"


async def test_docker_session_archives_inside_the_container(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[tuple[str, ...], bool]] = []

    async def fake_docker(*args: str, check: bool = True) -> tuple[str, str]:
        calls.append((args, check))
        if args[:3] == ("exec", "cid-42", "sh"):
            return "1", ""
        return "", ""

    monkeypatch.setattr(runtime_module, "_docker", fake_docker)
    async with runtime_module.DockerEndpoint(
        url="tcp://127.0.0.1:8765",
        container="cid-42",
    ).snapshot_session("sess-actor") as destination:
        assert destination is not None

    probe, export, copy, cleanup = calls
    assert probe[0][:3] == ("exec", "cid-42", "sh")
    assert export[0][:6] == ("exec", "--user", "0", "cid-42", "python3", "-c")
    assert "runtime session contains a symbolic link" in export[0][6]
    assert export[0][7] == "/media/hud/sessions/sess-actor"
    archive = export[0][8]
    assert copy == (("cp", f"cid-42:{archive}", str(destination)), True)
    assert cleanup == (
        ("exec", "--user", "0", "cid-42", "rm", "-f", archive),
        False,
    )


async def test_runtime_config_supplies_image_and_resources(
    tmp_path: Path, docker_log: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_fake_docker(tmp_path, port_behavior="echo 127.0.0.1:43210", monkeypatch=monkeypatch)

    task = Task(
        env="any-env",
        id="t",
        runtime_config=RuntimeConfig(
            image="img:firefox",
            resources=RuntimeResources(cpu=2, memory_mb=4096, gpu=RuntimeGPU()),
        ),
    )

    async with DockerRuntime()(task) as runtime:
        assert runtime.url == "tcp://127.0.0.1:43210"
        assert runtime.config == task.runtime_config

    calls = await _docker_calls(docker_log)
    assert calls[0] == (
        f"run --detach --cpus 2 --memory 4096m --gpus 1 {_docker_security_args()} "
        "--publish 127.0.0.1::8765 img:firefox"
    )


async def test_task_runtime_config_overrides_default_image(
    tmp_path: Path, docker_log: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_fake_docker(tmp_path, port_behavior="echo 127.0.0.1:43210", monkeypatch=monkeypatch)

    task = Task(env="any-env", id="t", runtime_config=RuntimeConfig(image="img:task"))

    async with DockerRuntime(
        "img:default",
        runtime_config=RuntimeConfig(
            resources=RuntimeResources(cpu=2, memory_mb=4096),
        ),
    )(task) as runtime:
        assert runtime.config == RuntimeConfig(
            image="img:task",
            resources=RuntimeResources(cpu=2, memory_mb=4096),
        )

    assert (await _docker_calls(docker_log))[0] == (
        f"run --detach --cpus 2 --memory 4096m {_docker_security_args()} "
        "--publish 127.0.0.1::8765 img:task"
    )


def test_runtime_config_overrides_only_explicit_top_level_fields() -> None:
    default = RuntimeConfig(
        resources=RuntimeResources(
            cpu=2,
            memory_mb=4096,
            gpu=RuntimeGPU(type=["A10G", "L4"], count=2),
        ),
        limits=RuntimeLimits(startup_timeout_s=30, run_timeout_s=120),
    )

    assert default.with_overrides(RuntimeConfig(image="img:task")) == RuntimeConfig(
        image="img:task",
        resources=RuntimeResources(
            cpu=2,
            memory_mb=4096,
            gpu=RuntimeGPU(type=["A10G", "L4"], count=2),
        ),
        limits=RuntimeLimits(startup_timeout_s=30, run_timeout_s=120),
    )
    assert default.with_overrides(
        RuntimeConfig(resources=RuntimeResources(cpu=4))
    ) == RuntimeConfig(
        resources=RuntimeResources(cpu=4),
        limits=RuntimeLimits(startup_timeout_s=30, run_timeout_s=120),
    )
    assert default.with_overrides(RuntimeConfig(resources=None)).resources is None


def test_runtime_config_source_override_is_mutually_exclusive(tmp_path: Path) -> None:
    compose = tmp_path / "compose.yaml"
    replacement = tmp_path / "replacement.yaml"

    assert RuntimeConfig(image="img:default").with_overrides(
        RuntimeConfig(compose=ComposeProject(document=compose))
    ) == RuntimeConfig(compose=ComposeProject(document=compose))
    assert RuntimeConfig(compose=ComposeProject(document=compose)).with_overrides(
        RuntimeConfig(image="img:task")
    ) == RuntimeConfig(image="img:task")
    assert RuntimeConfig(compose=ComposeProject(document=compose, root=tmp_path)).with_overrides(
        RuntimeConfig(compose=ComposeProject(document=replacement))
    ) == RuntimeConfig(compose=ComposeProject(document=replacement))


async def test_runtime_config_rejects_typed_docker_gpu() -> None:
    with pytest.raises(ValueError, match="GPU"):
        async with DockerRuntime()(
            Task(
                env="any-env",
                id="t",
                runtime_config=RuntimeConfig(
                    image="img",
                    resources=RuntimeResources(gpu=RuntimeGPU(type="L40S")),
                ),
            )
        ):
            pass


async def test_docker_runtime_starts_compose_with_a_main_service_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("DOCKER_HOST", raising=False)
    calls: list[tuple[str, ...]] = []
    recipe: dict[str, Any] = {}
    rendered: dict[str, Any] = {}
    port_override = ""
    compose = tmp_path / "compose.yaml"
    compose.write_text(
        "services:\n  main:\n    image: hud-env:one\n  db:\n    image: postgres:17\n",
        encoding="utf-8",
    )
    marker = tmp_path / "prepared"
    (tmp_path / "build.sh").write_text(
        f"#!/bin/sh\nprintf prepared > {marker}\n",
        encoding="utf-8",
    )

    async def fake_docker(*args: str, **_kwargs: Any) -> tuple[str, str]:
        nonlocal port_override
        calls.append(args)
        if args[:2] == ("context", "inspect"):
            return "unix:///Users/test/.docker/run/docker.sock\n", ""
        if args[-4:] == ("up", "--detach", "--no-build", "--remove-orphans"):
            files = [Path(args[index + 1]) for index, value in enumerate(args) if value == "--file"]
            compose_file, override, ports = files
            recipe.update(json.loads(compose_file.read_text("utf-8")))
            rendered.update(json.loads(override.read_text("utf-8")))
            port_override = ports.read_text("utf-8")
        if args[-3:] == ("port", "main", "8765"):
            return "127.0.0.1:43210\n", ""
        return "", ""

    monkeypatch.setattr(runtime_module, "_docker", fake_docker)
    task = Task(
        env="any-env",
        id="t",
        runtime_config=RuntimeConfig(
            compose=ComposeProject(document=compose, service_access=True),
            resources=RuntimeResources(cpu=2, memory_mb=4096),
        ),
    )

    async with DockerRuntime()(task) as runtime:
        assert runtime.url == "tcp://127.0.0.1:43210"
        assert runtime.config == task.runtime_config

    assert marker.read_text("utf-8") == "prepared"
    assert recipe["services"]["main"]["image"] == "hud-env:one"
    assert rendered["services"]["main"] == {
        "security_opt": [
            f"seccomp={runtime_module._DOCKER_SECCOMP_PROFILE}",
            "systempaths=unconfined",
            "apparmor=unconfined",
        ],
        "cpus": 2.0,
        "mem_limit": "4096m",
        "volumes": [
            {
                "type": "bind",
                "source": "/Users/test/.docker/run/docker.sock",
                "target": "/media/hud/docker.sock",
            }
        ],
    }
    assert 'ports: !override ["127.0.0.1::8765"]' in port_override
    up = next(
        call for call in calls if call[-4:] == ("up", "--detach", "--no-build", "--remove-orphans")
    )
    assert up[up.index("--project-directory") + 1] == str(tmp_path)
    assert str(compose) not in up
    assert calls[-1][-3:] == ("down", "--volumes", "--remove-orphans")


async def test_docker_runtime_passes_env_vars_to_docker_run(
    tmp_path: Path, docker_log: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_fake_docker(tmp_path, port_behavior="echo 127.0.0.1:43210", monkeypatch=monkeypatch)

    provider = DockerRuntime("img:tag", env_vars={"OPENAI_API_KEY": "sk-test"})
    async with provider(_row()) as runtime:
        assert runtime.url == "tcp://127.0.0.1:43210"

    calls = await _docker_calls(docker_log)
    assert calls[0] == (
        f"run --detach --env OPENAI_API_KEY=sk-test {_docker_security_args()} "
        "--publish 127.0.0.1::8765 img:tag"
    )


async def test_docker_runtime_stages_env_vars_into_the_compose_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rendered: dict[str, Any] = {}
    compose = tmp_path / "compose.yaml"
    compose.write_text("services:\n  main:\n    image: hud-env:one\n", encoding="utf-8")

    async def fake_docker(*args: str, **_kwargs: Any) -> tuple[str, str]:
        if "up" in args:
            files = [Path(args[index + 1]) for index, value in enumerate(args) if value == "--file"]
            _, override, _ = files
            rendered.update(json.loads(override.read_text("utf-8")))
        if args[-3:] == ("port", "main", "8765"):
            return "127.0.0.1:43210\n", ""
        return "", ""

    monkeypatch.setattr(runtime_module, "_docker", fake_docker)
    task = Task(
        env="any-env",
        id="t",
        runtime_config=RuntimeConfig(compose=ComposeProject(document=compose)),
    )
    provider = DockerRuntime(env_vars={"OPENAI_API_KEY": "sk-test"})

    async with provider(task):
        pass

    assert rendered["services"]["main"]["environment"] == {"OPENAI_API_KEY": "sk-test"}


async def test_docker_runtime_serializes_shared_compose_preparation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    compose = tmp_path / "compose.yaml"
    compose.write_text("services:\n  main:\n    image: hud-env:one\n", encoding="utf-8")
    active = 0
    peak = 0

    async def prepare(_compose: Path, _max_wait: float | None) -> bool:
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.01)
        active -= 1
        return True

    async def fake_docker(*args: str, **_kwargs: Any) -> tuple[str, str]:
        if args[-3:] == ("port", "main", "8765"):
            return "127.0.0.1:43210\n", ""
        return "", ""

    monkeypatch.setattr(runtime_module, "_prepare_compose_project", prepare)
    monkeypatch.setattr(runtime_module, "_docker", fake_docker)
    task = Task(
        env="any-env",
        id="t",
        runtime_config=RuntimeConfig(compose=ComposeProject(document=compose)),
    )
    provider = DockerRuntime()

    async def acquire() -> None:
        async with provider(task):
            pass

    await asyncio.gather(acquire(), acquire())

    assert peak == 1


async def test_docker_times_out_compose_preparation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    compose = tmp_path / "compose.yaml"
    compose.write_text("services:\n  main:\n    image: hud-env:one\n", encoding="utf-8")
    (tmp_path / "build.sh").touch()
    max_waits: list[float | None] = []

    class Process:
        async def complete(self, *, max_wait: float | None = None) -> SimpleNamespace:
            max_waits.append(max_wait)
            return SimpleNamespace(returncode=None, stdout=b"", stderr=b"", timed_out=True)

    async def create_process(*_args: str, **_kwargs: Any) -> Process:
        return Process()

    monkeypatch.setattr(runtime_module, "create_process_group_exec", create_process)
    task = Task(
        env="any-env",
        id="t",
        runtime_config=RuntimeConfig(
            compose=ComposeProject(document=compose),
            limits=RuntimeLimits(startup_timeout_s=45),
        ),
    )

    with pytest.raises(TimeoutError, match="Compose project build timed out after 45 seconds"):
        async with DockerRuntime()(task):
            pass

    assert max_waits == [45]


async def test_docker_runtime_rejects_remote_compose_service_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    compose = tmp_path / "compose.yaml"
    compose.write_text("services:\n  main:\n    image: hud-env:one\n", encoding="utf-8")
    monkeypatch.setenv("DOCKER_HOST", "tcp://docker.example:2376")

    with pytest.raises(ValueError, match="requires compose_service_socket"):
        async with DockerRuntime()(
            Task(
                env="any-env",
                id="t",
                runtime_config=RuntimeConfig(
                    compose=ComposeProject(document=compose, service_access=True),
                ),
            )
        ):
            pass


async def test_docker_runtime_mounts_the_daemon_visible_socket(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    compose = tmp_path / "compose.yaml"
    compose.write_text("services:\n  main:\n    image: hud-env:one\n", encoding="utf-8")
    monkeypatch.setenv("DOCKER_HOST", "tcp://docker.example:2376")
    rendered: dict[str, Any] = {}

    async def fake_docker(*args: str, **_kwargs: Any) -> tuple[str, str]:
        if args[-4:] == ("up", "--detach", "--build", "--remove-orphans"):
            files = [Path(args[index + 1]) for index, value in enumerate(args) if value == "--file"]
            rendered.update(json.loads(files[1].read_text("utf-8")))
        if args[-3:] == ("port", "main", "8765"):
            return "127.0.0.1:43210\n", ""
        return "", ""

    monkeypatch.setattr(runtime_module, "_docker", fake_docker)
    task = Task(
        env="any-env",
        id="t",
        runtime_config=RuntimeConfig(compose=ComposeProject(document=compose, service_access=True)),
    )

    async with DockerRuntime(compose_service_socket="/vm/run/docker.sock")(task):
        pass

    assert rendered["services"]["main"]["volumes"] == [
        {
            "type": "bind",
            "source": "/vm/run/docker.sock",
            "target": "/media/hud/docker.sock",
        }
    ]


def test_docker_runtime_accepts_only_one_environment_definition(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="either image or compose"):
        RuntimeConfig(
            image="img:tag",
            compose=ComposeProject(document=tmp_path / "compose.yaml"),
        )


def test_docker_runtime_accepts_runtime_config_defaults() -> None:
    provider = DockerRuntime("img:tag")
    assert provider.runtime_config == RuntimeConfig(image="img:tag")

    provider_with_resources = DockerRuntime(
        "img:tag",
        runtime_config=RuntimeConfig(resources=RuntimeResources(cpu=2)),
    )
    assert provider_with_resources.runtime_config == RuntimeConfig(
        image="img:tag",
        resources=RuntimeResources(cpu=2),
    )

    provider = DockerRuntime("img:tag", runtime_config=RuntimeConfig(image="other:tag"))
    assert provider.runtime_config == RuntimeConfig(image="other:tag")

    task = Task(env="any-env", id="t", runtime_config=RuntimeConfig(image="other:tag"))
    assert provider_with_resources.runtime_config is not None
    assert provider_with_resources.runtime_config.with_overrides(
        task.runtime_config
    ) == RuntimeConfig(
        image="other:tag",
        resources=RuntimeResources(cpu=2),
    )


async def test_modal_runtime_config_flows_into_modal_sdk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _install_fake_modal(monkeypatch)
    config = RuntimeConfig(
        image="img:tag",
        resources=RuntimeResources(
            cpu=2,
            memory_mb=4096,
            gpu=RuntimeGPU(type="A10G", count=2),
        ),
        limits=RuntimeLimits(startup_timeout_s=30, run_timeout_s=120),
    )
    provider = ModalRuntime(runtime_config=config)

    async with provider(_row()) as runtime:
        assert runtime.url == "tcp://modal.host:4567"
        assert runtime.params == {"provider": "modal", "instance_id": "sb-1"}
        assert runtime.config == config

    assert calls["registry_image"] == "img:tag"
    assert calls["app_lookup"] == ("hud-envs", True)
    assert calls["sandbox_args"] == provider.command
    assert calls["sandbox_kwargs"] == {
        "app": "app",
        "image": _ModalImageRef("registry", "img:tag"),
        "workdir": None,
        "unencrypted_ports": [8765],
        "readiness_probe": ("tcp", 8765),
        "timeout": 120,
        "cpu": 2,
        "memory": 4096,
        "gpu": "A10G:2",
    }
    assert calls["ready_timeout"] == 30
    assert calls["terminated"] is True


async def test_modal_runtime_uses_caller_owned_app(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _install_fake_modal(monkeypatch)
    app = object()

    async def build(*, app: object) -> None:
        calls["image_build_app"] = app

    image = SimpleNamespace(build=SimpleNamespace(aio=build))
    provider = ModalRuntime(app=cast("Any", app), image=cast("Any", image))

    async with provider(_row()):
        pass

    assert "app_lookup" not in calls
    assert calls["image_build_app"] is app
    assert calls["sandbox_kwargs"]["app"] is app


def test_modal_runtime_rejects_app_and_app_name() -> None:
    with pytest.raises(ValueError, match="either app or app_name"):
        ModalRuntime(app=cast("Any", object()), app_name="hud-envs")


async def test_modal_runtime_rejects_gpu_alternatives() -> None:
    provider = ModalRuntime(
        runtime_config=RuntimeConfig(
            image="img:tag",
            resources=RuntimeResources(gpu=RuntimeGPU(type=["H100", "A100"])),
        )
    )

    with pytest.raises(ValueError, match="does not support alternative GPU types"):
        async with provider(_row()):
            pass


async def test_modal_runtime_runs_compose_inside_a_dind_vm(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls = _install_fake_modal(monkeypatch)
    calls["compose_stdout"] = "main  | server ready\n"
    calls["compose_stderr"] = "grader | warning\n"
    project = tmp_path / "artifact"
    compose = project / "compose-project" / "compose.yaml"
    compose.parent.mkdir(parents=True)
    compose.write_text("services:\n  main:\n    image: hud-env:one\n", encoding="utf-8")

    async with ModalRuntime(
        runtime_config=RuntimeConfig(
            compose=ComposeProject(document=compose, root=project, service_access=True),
            limits=RuntimeLimits(startup_timeout_s=600),
        ),
        env_vars={"HUD_API_KEY": "secret"},
    )(_row()) as runtime:
        assert runtime.url == "tcp://modal.host:4567"
        assert runtime.params == {"provider": "modal", "instance_id": "sb-1", "ready_timeout": 600}
        async with runtime.snapshot_session("sess-actor") as archive:
            assert archive is not None
            await runtime.restore_session("sess-verifier", archive)

    assert calls["registry_image"] == "docker:28.3.3-dind"
    kwargs = calls["sandbox_kwargs"]
    assert isinstance(kwargs, dict)
    assert kwargs["experimental_options"] == {"vm_runtime": True}
    assert kwargs["readiness_probe"] is None
    assert kwargs["cpu"] == 4
    assert kwargs["memory"] == 8192
    assert kwargs["image"] == _ModalImageRef("registry", "docker:28.3.3-dind")
    assert "env" not in kwargs
    assert calls["uploads"] == [
        ("project.tar.gz", "/hud/project.tar.gz"),
        ("override.json", "/hud/override.json"),
        ("ports.yaml", "/hud/ports.yaml"),
        ("docker-seccomp.json", "/hud/docker-seccomp.json"),
        ("session.tar.gz", "/media/hud/session.tar.gz"),
    ]
    override = calls["compose_override"]
    assert isinstance(override, dict)
    assert override["services"]["main"]["volumes"] == [
        {
            "type": "bind",
            "source": "/var/run/docker.sock",
            "target": "/media/hud/docker.sock",
        }
    ]
    assert override["services"]["main"]["environment"] == {"HUD_API_KEY": "secret"}
    execs = calls["execs"]
    assert isinstance(execs, list)
    startup = execs[0][0][-1]
    assert "docker compose" in startup
    assert "--project-directory /hud/project/compose-project" in startup
    assert "--file /hud/project/compose-project/compose.yaml" in startup
    assert "sh /hud/project/build.sh" in startup
    assert 'up --detach "$BUILD_FLAG" --remove-orphans' in startup
    compose_logs = execs[1][0]
    assert compose_logs[-3:] == ("logs", "--follow", "--no-color")
    session_commands = [
        call[0][-1]
        for call in execs[1:-1]
        if call[0][:2] == ("sh", "-c") and "docker compose" in call[0][-1]
    ]
    assert session_commands
    assert all(
        "--project-directory /hud/project/compose-project" in command
        and "--file /hud/project/compose-project/compose.yaml" in command
        for command in session_commands
    )
    assert any("runtime session contains an unsupported entry" in call[0][-1] for call in execs)
    teardown = execs[-1][0]
    assert teardown[teardown.index("--project-directory") + 1] == ("/hud/project/compose-project")
    assert teardown[teardown.index("--file") + 1] == ("/hud/project/compose-project/compose.yaml")
    assert "down" in teardown
    assert execs[0][1]["timeout"] == 600
    assert calls["downloads"] == [("/media/hud/session.tar.gz", "session.tar.gz")]
    captured = capsys.readouterr()
    assert captured.out == "main  | server ready\n"
    assert captured.err == "grader | warning\n"


async def test_modal_runtime_bounds_compose_startup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _install_fake_modal(monkeypatch)
    calls["exec_wait"] = asyncio.Event()
    compose = tmp_path / "compose.yaml"
    compose.write_text("services:\n  main:\n    image: hud-env:one\n", encoding="utf-8")

    with pytest.raises(TimeoutError, match="Modal Compose startup timed out after 1 seconds"):
        async with ModalRuntime(
            runtime_config=RuntimeConfig(
                compose=ComposeProject(document=compose),
                limits=RuntimeLimits(startup_timeout_s=1),
            )
        )(_row()):
            pytest.fail("runtime should not become ready")

    assert calls["terminated"] is True


async def test_modal_runtime_rejects_gpu_inside_compose_dind(tmp_path: Path) -> None:
    compose = tmp_path / "compose.yaml"
    compose.write_text("services:\n  main:\n    image: hud-env:one\n", encoding="utf-8")

    provider = ModalRuntime(
        runtime_config=RuntimeConfig(
            compose=ComposeProject(document=compose),
            resources=RuntimeResources(gpu=RuntimeGPU(type="H100")),
        )
    )

    with pytest.raises(ValueError, match=r"cannot attach GPUs.*Docker-in-Docker"):
        async with provider(_row()):
            pass


async def test_modal_runtime_accepts_modal_image_uri(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _install_fake_modal(monkeypatch)
    config = RuntimeConfig(image="modal://im-built")

    async with ModalRuntime(runtime_config=config)(_row()) as runtime:
        assert runtime.config == config

    assert calls["modal_image_id"] == "im-built"
    assert calls["sandbox_kwargs"] == {
        "app": "app",
        "image": _ModalImageRef("id", "im-built"),
        "workdir": None,
        "unencrypted_ports": [8765],
        "readiness_probe": ("tcp", 8765),
        "timeout": 3600,
    }


async def test_modal_task_runtime_config_overlays_provider_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _install_fake_modal(monkeypatch)
    provider = ModalRuntime(
        runtime_config=RuntimeConfig(
            resources=RuntimeResources(cpu=2, memory_mb=4096),
            limits=RuntimeLimits(startup_timeout_s=30, run_timeout_s=120),
        ),
    )
    task = Task(env="any-env", id="t", runtime_config=RuntimeConfig(image="img:task"))

    async with provider(task) as runtime:
        assert runtime.config == RuntimeConfig(
            image="img:task",
            resources=RuntimeResources(cpu=2, memory_mb=4096),
            limits=RuntimeLimits(startup_timeout_s=30, run_timeout_s=120),
        )

    assert calls["registry_image"] == "img:task"
    assert calls["ready_timeout"] == 30
    assert calls["sandbox_kwargs"] == {
        "app": "app",
        "image": _ModalImageRef("registry", "img:task"),
        "workdir": None,
        "unencrypted_ports": [8765],
        "readiness_probe": ("tcp", 8765),
        "timeout": 120,
        "cpu": 2,
        "memory": 4096,
    }


async def test_modal_runtime_config_image_overrides_image_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _install_fake_modal(monkeypatch)
    config = RuntimeConfig(image="img:tag", resources=RuntimeResources(gpu=RuntimeGPU()))
    async with ModalRuntime("ignored-name", runtime_config=config)(_row()) as runtime:
        assert runtime.config == config

    assert calls["registry_image"] == "img:tag"


async def test_modal_runtime_can_override_workdir(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _install_fake_modal(monkeypatch)
    config = RuntimeConfig(image="img:tag")
    provider = ModalRuntime(runtime_config=config, workdir="/app")

    async with provider(_row()) as runtime:
        assert runtime.config == config

    sandbox_kwargs = calls["sandbox_kwargs"]
    assert isinstance(sandbox_kwargs, dict)
    assert sandbox_kwargs["workdir"] == "/app"


async def test_modal_runtime_passes_env_vars_to_sandbox(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _install_fake_modal(monkeypatch)
    config = RuntimeConfig(image="img:tag")
    provider = ModalRuntime(runtime_config=config, env_vars={"TOKEN": "secret"})

    async with provider(_row()) as runtime:
        assert runtime.config == config

    sandbox_kwargs = calls["sandbox_kwargs"]
    assert isinstance(sandbox_kwargs, dict)
    assert sandbox_kwargs["image"] == _ModalImageRef("registry", "img:tag")
    assert sandbox_kwargs["env"] == {"TOKEN": "secret"}


async def test_modal_runtime_streams_sandbox_output_to_terminal(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls = _install_fake_modal(monkeypatch)
    calls["sandbox_stdout"] = "server ready\nrequest complete"
    calls["sandbox_stderr"] = "grader warning\n"
    provider = ModalRuntime(runtime_config=RuntimeConfig(image="img:tag"))

    async with provider(_row()):
        pass

    captured = capsys.readouterr()
    assert captured.out == "server ready\nrequest complete"
    assert captured.err == "grader warning\n"


async def test_modal_runtime_bounds_output_teardown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _install_fake_modal(monkeypatch)
    calls["sandbox_stdout_wait"] = asyncio.Event()
    calls["sandbox_stderr_wait"] = asyncio.Event()
    monkeypatch.setattr(process_module, "OUTPUT_DRAIN_TIMEOUT", 0.01)

    async def run() -> None:
        async with ModalRuntime(runtime_config=RuntimeConfig(image="img:tag"))(_row()):
            pass

    await asyncio.wait_for(run(), timeout=1)
    assert calls["terminated"] is True


async def test_modal_runtime_attaches_sandbox_secrets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _install_fake_modal(monkeypatch)
    secret = _ModalSecretRef("datacuration-cpu50-grader-auth")
    provider = ModalRuntime(
        runtime_config=RuntimeConfig(image="img:tag"),
        sandbox_secrets=[cast("Any", secret)],
    )

    async with provider(_row()):
        pass

    sandbox_kwargs = calls["sandbox_kwargs"]
    assert isinstance(sandbox_kwargs, dict)
    assert sandbox_kwargs["secrets"] == (secret,)
    assert calls["registry_secret"] is None


async def test_modal_runtime_passes_registry_secret_to_from_registry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _install_fake_modal(monkeypatch)
    registry_secret = _ModalSecretRef("registry-creds")
    provider = ModalRuntime(
        runtime_config=RuntimeConfig(image="ghcr.io/org/private:tag"),
        registry_secret=cast("Any", registry_secret),
    )

    async with provider(_row()):
        pass

    assert calls["registry_image"] == "ghcr.io/org/private:tag"
    assert calls["registry_secret"] is registry_secret


async def test_modal_runtime_rejects_sandbox_secrets_for_compose(
    tmp_path: Path,
) -> None:
    compose = tmp_path / "compose.yaml"
    compose.write_text("services:\n  main:\n    image: hud-env:one\n", encoding="utf-8")
    provider = ModalRuntime(
        runtime_config=RuntimeConfig(compose=ComposeProject(document=compose)),
        sandbox_secrets=[cast("Any", _ModalSecretRef("grader-auth"))],
    )

    with pytest.raises(ValueError, match="sandbox secrets require an image runtime"):
        async with provider(_row()):
            pass


async def test_daytona_runtime_config_flows_into_daytona_sdk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _install_fake_daytona(monkeypatch).calls
    config = RuntimeConfig(
        image="img:tag",
        resources=RuntimeResources(
            cpu=2,
            memory_mb=4096,
            gpu=RuntimeGPU(type=["H100", "A100"], count=2),
        ),
        limits=RuntimeLimits(startup_timeout_s=45),
    )
    provider = DaytonaRuntime(runtime_config=config)

    async with provider(_row()) as runtime:
        assert runtime.url == "tcp://127.0.0.1:54321"
        assert runtime.params == {"provider": "daytona", "instance_id": "sandbox-1"}
        assert runtime.config == config

    create_call = calls["create"]
    assert isinstance(create_call, tuple)
    create_params, create_timeout = create_call
    assert create_params == _CreateSandboxFromImageParams(
        image=_DaytonaImage("img:tag"),
        ephemeral=True,
        auto_stop_interval=0,
        resources=_DaytonaResources(
            cpu=2,
            memory=4,
            gpu=2,
            gpu_type=[_DaytonaGpuType("H100"), _DaytonaGpuType("A100")],
        ),
    )
    assert create_timeout == 45
    assert calls["session"] == "hud-serve"
    assert calls["execute"] == (
        "hud-serve",
        _SessionExecuteRequest(
            command=(
                'cd /app && PATH="$PWD/.venv/bin:$PATH" hud serve env.py --host 0.0.0.0 --port 8765'
            ),
            run_async=True,
        ),
    )
    assert calls["ssh_expires"] == 24 * 60
    assert calls["ssh_connect"] == (
        ("ssh.app.daytona.io",),
        {"username": "ssh-token", "known_hosts": None},
    )
    assert calls["forward"] == ("127.0.0.1", 0, "127.0.0.1", 8765)
    assert calls["delete"] == "sandbox-1"


async def test_daytona_runtime_streams_sandbox_output_to_terminal(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls = _install_fake_daytona(monkeypatch).calls
    calls["daytona_stdout"] = "server ready\n"
    calls["daytona_stderr"] = "grader warning\n"

    async with DaytonaRuntime("snapshot")(_row()):
        pass

    captured = capsys.readouterr()
    assert captured.out == "server ready\n"
    assert captured.err == "grader warning\n"
    assert calls["stream_logs"] == ("hud-serve", "cmd-1")


async def test_daytona_runtime_bounds_output_teardown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _install_fake_daytona(monkeypatch).calls
    calls["daytona_log_wait"] = asyncio.Event()
    monkeypatch.setattr(process_module, "OUTPUT_DRAIN_TIMEOUT", 0.01)

    async def run() -> None:
        async with DaytonaRuntime("snapshot")(_row()):
            pass

    await asyncio.wait_for(run(), timeout=1)
    assert calls["delete"] == "sandbox-1"


async def test_daytona_runtime_rejects_compose(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _install_fake_daytona(monkeypatch)
    compose = tmp_path / "compose.yaml"
    compose.write_text("services: {}\n", encoding="utf-8")

    with pytest.raises(ValueError, match=r"does not support runtime_config\.compose"):
        async with DaytonaRuntime(
            runtime_config=RuntimeConfig(compose=ComposeProject(document=compose))
        )(_row()):
            pass

    assert client.created == []


def test_compose_config_normalizes_supported_short_syntax(tmp_path: Path) -> None:
    compose = tmp_path / "compose.yaml"
    compose.write_text(
        """
services:
  main:
    image: example:latest
    environment:
      - FIRST=one
      - EMPTY=
    entrypoint: /bin/start --flag
    command: serve --port 8000
    expose: [8000]
    ports:
      - 8000
      - 127.0.0.1:8080:80/tcp
""",
        encoding="utf-8",
    )

    service = ComposeConfig.from_file(compose).services["main"]

    assert service.environment == {"FIRST": "one", "EMPTY": ""}
    assert service.entrypoint == ["/bin/start", "--flag"]
    assert service.command == ["serve", "--port", "8000"]
    assert service.expose == ["8000"]
    assert [port.model_dump(exclude_none=True) for port in service.ports] == [
        {"target": 8000, "protocol": "tcp"},
        {
            "target": 80,
            "protocol": "tcp",
            "published": 8080,
            "host_ip": "127.0.0.1",
        },
    ]

    mapped = ComposeService.model_validate({"environment": {"COUNT": 2, "ENABLED": True}})
    assert mapped.environment == {"COUNT": "2", "ENABLED": "true"}


def test_compose_service_tcp_ports_excludes_udp_expose() -> None:
    service = ComposeService.model_validate({"expose": [80, "443/tcp", "53/udp"]})

    assert service.tcp_ports == {80, 443}


@pytest.mark.parametrize(
    "service",
    [
        "extends: {file: base.yaml, service: base}",
    ],
)
def test_compose_config_rejects_external_resolution(tmp_path: Path, service: str) -> None:
    compose = tmp_path / "compose.yaml"
    compose.write_text(f"services:\n  main:\n    {service}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="does not support Compose"):
        ComposeConfig.from_file(compose)


def test_compose_config_interpolates_only_artifact_supplied_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOST_ONLY", "secret")
    (tmp_path / ".env").write_text(
        "REGISTRY=example\nIMAGE=${REGISTRY}:1\nEMPTY=\nTOKEN=cost$5\n",
        encoding="utf-8",
    )
    compose = tmp_path / "compose.yaml"
    compose.write_text(
        """
services:
  main:
    image: ${IMAGE}
    command: "${EMPTY:-serve} $$HOME ${MISSING-default} $? $"
    environment:
      LITERAL: '$HOST_ONLY'
      TOKEN: ${TOKEN}
""",
        encoding="utf-8",
    )

    config = ComposeConfig.from_file(compose)
    service = config.services["main"]

    assert service.image == "example:1"
    assert service.command == ["serve", "$$HOME", "default", "$$?", "$$"]
    assert service.environment == {"LITERAL": "$$HOST_ONLY", "TOKEN": "cost$$5"}

    normalized = tmp_path / "normalized.json"
    normalized.write_text(json.dumps(config.model_dump(mode="json")), encoding="utf-8")
    assert ComposeConfig.from_file(normalized) == config


@pytest.mark.parametrize(
    ("image", "message"),
    [
        ("$HOST_ONLY", r"HOST_ONLY.*not set by the project \.env"),
        ("${IMAGE:?set IMAGE in .env}", r"set IMAGE in \.env"),
    ],
)
def test_compose_config_rejects_unbound_variables(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    image: str,
    message: str,
) -> None:
    monkeypatch.setenv("HOST_ONLY", "secret")
    compose = tmp_path / "compose.yaml"
    compose.write_text(
        f"services:\n  main:\n    image: {image}\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=message):
        ComposeConfig.from_file(compose)


def test_compose_config_relocates_project_paths() -> None:
    compose = ComposeConfig.model_validate(
        {
            "services": {
                "main": {
                    "build": {
                        "context": ".",
                        "additional_contexts": {
                            "fixture": "./fixture",
                            "base": "service:base",
                        },
                    },
                    "env_file": [".env", {"path": "./optional.env", "required": False}],
                    "label_file": "./labels",
                    "volumes": [
                        "./data:/data:ro",
                        "cache:/cache",
                        {"type": "bind", "source": "./config", "target": "/config"},
                    ],
                },
                "base": {"image": "alpine"},
            },
            "configs": {"settings": {"file": "./settings.json"}},
            "secrets": {"token": {"file": "./token"}},
        }
    )

    relocated = compose.with_project_directory("./environment").model_dump(
        mode="json", exclude_none=True
    )
    main = relocated["services"]["main"]

    assert main["build"] == {
        "context": "./environment",
        "additional_contexts": {
            "fixture": "./environment/fixture",
            "base": "service:base",
        },
    }
    assert main["env_file"] == [
        "./environment/.env",
        {"path": "./environment/optional.env", "required": False},
    ]
    assert main["label_file"] == "./environment/labels"
    assert main["volumes"] == [
        "./environment/data:/data:ro",
        "cache:/cache",
        {"type": "bind", "source": "./environment/config", "target": "/config"},
    ]
    assert relocated["configs"] == {"settings": {"file": "./environment/settings.json"}}
    assert relocated["secrets"] == {"token": {"file": "./environment/token"}}


@pytest.mark.parametrize("path", ["../data", "./nested/../../data"])
def test_compose_config_rejects_paths_outside_project(path: str) -> None:
    compose = ComposeConfig.model_validate({"services": {"main": {"volumes": [f"{path}:/data"]}}})

    with pytest.raises(ValueError, match="escapes its project directory"):
        compose.with_project_directory("./environment")


def test_compose_healthcheck_does_not_invent_duration_values() -> None:
    service = ComposeService(
        healthcheck=ComposeHealthcheck(test=["CMD", "healthcheck"]),
    )

    assert service.model_dump(exclude_none=True)["healthcheck"] == {"test": ["CMD", "healthcheck"]}


async def test_daytona_task_runtime_config_overlays_provider_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _install_fake_daytona(monkeypatch).calls
    provider = DaytonaRuntime(
        runtime_config=RuntimeConfig(
            resources=RuntimeResources(cpu=2, memory_mb=4096),
            limits=RuntimeLimits(startup_timeout_s=45),
        ),
    )
    task = Task(env="any-env", id="t", runtime_config=RuntimeConfig(image="img:task"))

    async with provider(task) as runtime:
        assert runtime.config == RuntimeConfig(
            image="img:task",
            resources=RuntimeResources(cpu=2, memory_mb=4096),
            limits=RuntimeLimits(startup_timeout_s=45),
        )

    create_call = calls["create"]
    assert isinstance(create_call, tuple)
    create_params, create_timeout = create_call
    assert create_params == _CreateSandboxFromImageParams(
        image=_DaytonaImage("img:task"),
        ephemeral=True,
        auto_stop_interval=0,
        resources=_DaytonaResources(cpu=2, memory=4),
    )
    assert create_timeout == 45


async def test_daytona_snapshot_sandboxes_disable_auto_stop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _install_fake_daytona(monkeypatch).calls

    async with DaytonaRuntime("snapshot")(_row()):
        pass

    create_call = calls["create"]
    assert isinstance(create_call, tuple)
    create_params, create_timeout = create_call
    assert create_params == _CreateSandboxFromSnapshotParams(
        snapshot="snapshot",
        ephemeral=True,
        auto_stop_interval=0,
    )
    assert create_timeout == 120


@dataclass(frozen=True)
class _BuildContextEntry:
    source_path: str
    archive_path: str


class _BuildImage:
    def __init__(self, context: Path) -> None:
        self._context_list = [_BuildContextEntry(source_path=str(context), archive_path=".")]

    def dockerfile(self) -> str:
        return "FROM python:3.11-slim\nCOPY . .\n"


def _build_image(context: Path) -> _BuildImage:
    """A stand-in for ``daytona.Image.from_dockerfile``: the Dockerfile text plus
    the context entries the SDK would archive and upload."""
    return _BuildImage(context)


async def _boot_snapshot(context: Path, daytona: _FakeDaytonaClient) -> str:
    """Acquire once through a fresh provider; return the snapshot it booted."""
    async with DaytonaRuntime("env", image=cast("DaytonaImage", _build_image(context)))(_row()):
        pass
    return daytona.created[-1].snapshot


async def test_daytona_rebuilds_the_snapshot_in_place_when_the_env_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The name is the durable handle; content drift rebuilds it, never renames it —
    # otherwise every rollout after an edit silently measures the old code.
    daytona = _install_fake_daytona(monkeypatch)
    (tmp_path / "env.py").write_text("REWARD = 1.0\n")

    first = await _boot_snapshot(tmp_path, daytona)
    unchanged = await _boot_snapshot(tmp_path, daytona)
    (tmp_path / "env.py").write_text("REWARD = 0.0\n")
    edited = await _boot_snapshot(tmp_path, daytona)

    assert first == unchanged == edited == "env"
    # Built once per distinct content; the unchanged re-acquisition reused it,
    # and the stale build was replaced, not left behind.
    assert daytona.snapshot.builds == ["env", "env"]
    assert list(daytona.snapshot.snapshots) == ["env"]


async def test_daytona_registry_ref_image_builds_and_reuses_by_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # image= is documented as "Dockerfile/registry ref"; a plain ref string must
    # build once, reuse while unchanged, and rebuild when repointed.
    daytona = _install_fake_daytona(monkeypatch)

    for ref in ("registry/env:1", "registry/env:1", "registry/env:2"):
        async with DaytonaRuntime("env", image=ref)(_row()):
            pass

    assert daytona.snapshot.builds == ["env", "env"]
    assert daytona.snapshot.snapshots["env"].image_name == "registry/env:2"


async def test_daytona_sends_cpu_as_a_whole_number(monkeypatch: pytest.MonkeyPatch) -> None:
    # Daytona's API rejects 2.0, and no equality assertion can catch it: 2.0 == 2.
    daytona = _install_fake_daytona(monkeypatch)
    config = RuntimeConfig(image="img:tag", resources=RuntimeResources(cpu=2))

    async with DaytonaRuntime(runtime_config=config)(_row()):
        pass

    assert isinstance(daytona.created[-1].resources.cpu, int)


async def test_daytona_rejects_a_fractional_cpu_request(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_daytona(monkeypatch)
    config = RuntimeConfig(image="img:tag", resources=RuntimeResources(cpu=1.5))

    with pytest.raises(ValueError, match="whole number of CPUs"):
        async with DaytonaRuntime(runtime_config=config)(_row()):
            pass


async def test_daytona_rounds_minimum_storage_up_to_gibibytes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daytona = _install_fake_daytona(monkeypatch)
    config = RuntimeConfig(image="img:tag", resources=RuntimeResources(storage_mb=1025))

    async with DaytonaRuntime(runtime_config=config)(_row()):
        pass

    assert daytona.created[-1].resources.disk == 2


async def test_modal_accepts_best_effort_storage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = RuntimeConfig(image="img:tag", resources=RuntimeResources(storage_mb=1024))
    calls = _install_fake_modal(monkeypatch)

    async with ModalRuntime(runtime_config=config)(_row()):
        pass

    assert calls["sandbox_kwargs"] == {
        "app": "app",
        "image": _ModalImageRef("registry", "img:tag"),
        "workdir": None,
        "unencrypted_ports": [8765],
        "readiness_probe": ("tcp", 8765),
        "timeout": 3600,
    }


async def test_docker_admits_minimum_free_disk(
    tmp_path: Path, docker_log: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_fake_docker(
        tmp_path,
        port_behavior="echo 127.0.0.1:43210",
        monkeypatch=monkeypatch,
    )
    config = RuntimeConfig(image="img:tag", resources=RuntimeResources(storage_mb=1024))

    async with DockerRuntime(runtime_config=config)(_row()):
        pass

    calls = await _docker_calls(docker_log)
    assert "exec cid-42 df -Pk /" in calls


async def test_docker_honors_startup_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[tuple[str, ...], dict[str, Any]]] = []

    async def fake_docker(*args: str, **kwargs: Any) -> tuple[str, str]:
        calls.append((args, kwargs))
        if args[0] == "run":
            return "cid-42\n", ""
        if args[0] == "port":
            return "127.0.0.1:43210\n", ""
        return "", ""

    monkeypatch.setattr(runtime_module, "_docker", fake_docker)
    config = RuntimeConfig(image="img:tag", limits=RuntimeLimits(startup_timeout_s=300))

    async with DockerRuntime(runtime_config=config)(_row()) as runtime:
        assert runtime.params == {"ready_timeout": 300}

    assert calls[0][1]["deadline"] == 300


@pytest.mark.parametrize("compose", [False, True])
async def test_docker_run_timeout_owns_shared_substrate_lifetime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    compose: bool,
) -> None:
    torn_down = asyncio.Event()
    teardown_calls: list[tuple[str, ...]] = []
    provisions = 0

    async def fake_docker(*args: str, **kwargs: Any) -> tuple[str, str]:
        nonlocal provisions
        if args[0] == "run":
            provisions += 1
            return "cid-42\n", ""
        if "up" in args:
            provisions += 1
        if args[0] == "port" or args[-3:] == ("port", "main", "8765"):
            return "127.0.0.1:43210\n", ""
        if args[-3:] == ("ps", "--quiet", "main"):
            return "cid-42\n", ""
        if args[:2] == ("rm", "--force") or "down" in args:
            teardown_calls.append(args)
            torn_down.set()
        return "", ""

    monkeypatch.setattr(runtime_module, "_docker", fake_docker)

    limits = RuntimeLimits(run_timeout_s=1)
    if compose:
        compose_file = tmp_path / "compose.yaml"
        compose_file.write_text("services:\n  main:\n    image: img:tag\n")
        source = RuntimeConfig(compose=ComposeProject(document=compose_file), limits=limits)
    else:
        source = RuntimeConfig(image="img:tag", limits=limits)

    async with Shared(DockerRuntime(runtime_config=source), width=2) as runtime:
        async with runtime(_row()):
            pass
        async with runtime(_row()):
            await asyncio.wait_for(torn_down.wait(), 2)
        async with runtime(_row()):
            pass

    assert teardown_calls
    assert provisions == 2


async def test_daytona_names_a_sandbox_it_could_not_delete(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # Teardown must report the leak without shadowing the run's own error.
    daytona = _install_fake_daytona(monkeypatch)
    daytona.delete_fails = True

    with caplog.at_level(logging.WARNING, logger="hud.eval.runtime"):
        async with DaytonaRuntime("snapshot")(_row()):
            pass

    assert "sandbox-1" in caplog.text


async def test_daytona_sizes_each_row_from_its_own_runtime_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Resources are baked into a snapshot at build time, so rows that request
    # different sizes must boot distinct snapshots, not the first row's.
    daytona = _install_fake_daytona(monkeypatch)
    (tmp_path / "env.py").write_text("REWARD = 1.0\n")
    provider = DaytonaRuntime("env", image=cast("DaytonaImage", _build_image(tmp_path)))

    for cpu in (2, 4):
        task = Task(
            env="any-env",
            id="t",
            runtime_config=RuntimeConfig(resources=RuntimeResources(cpu=cpu)),
        )
        async with provider(task):
            pass

    assert daytona.snapshot.builds == ["env-2cpu", "env-4cpu"]
    assert [params.snapshot for params in daytona.created] == ["env-2cpu", "env-4cpu"]


async def test_daytona_attaches_the_env_output_to_a_failed_handshake(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The caller gets "closed connection during 'hello'" and nothing else.
    _install_fake_daytona(monkeypatch)

    with pytest.raises(EOFError) as excinfo:
        async with DaytonaRuntime("snapshot")(_row()):
            raise EOFError("env closed connection during 'hello'")

    assert any("no module named bugs" in note for note in excinfo.value.__notes__)


async def test_daytona_runtime_config_rejects_unsupported_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_daytona(monkeypatch)
    with pytest.raises(ValueError, match="resources"):
        async with DaytonaRuntime(
            "snapshot",
            runtime_config=RuntimeConfig(resources=RuntimeResources(cpu=1)),
        )(_row()):
            pass

    with pytest.raises(ValueError, match="run_timeout_s"):
        async with DaytonaRuntime(
            "snapshot",
            runtime_config=RuntimeConfig(limits=RuntimeLimits(run_timeout_s=60)),
        )(_row()):
            pass

    with pytest.raises(ValueError, match="run_timeout_s"):
        async with DaytonaRuntime(
            runtime_config=RuntimeConfig(
                image="img:tag",
                limits=RuntimeLimits(run_timeout_s=60),
            ),
        )(_row()):
            pass


async def test_container_that_dies_before_serving_fails_with_its_logs(
    tmp_path: Path, docker_log: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # ``docker port`` on an exited container prints nothing.
    _install_fake_docker(tmp_path, port_behavior=":", monkeypatch=monkeypatch)

    provider = DockerRuntime("img:tag")
    with pytest.raises(RuntimeError, match="exited before serving") as err:
        async with provider(_row()):
            pass

    assert "ImportError: boom" in str(err.value)
    calls = await _docker_calls(docker_log)
    assert "logs --tail 40 cid-42" in calls
    assert calls[-1] == "rm --force cid-42"  # cleanup still runs on failure


def test_docker_profile_allows_workspace_namespace_syscalls() -> None:
    profile = json.loads(runtime_module._DOCKER_SECCOMP_PROFILE.read_text())
    denied = {name for rule in profile["syscalls"] for name in rule["names"]}
    ptrace = next(rule for rule in profile["syscalls"] if "ptrace" in rule["names"])

    assert profile["defaultAction"] == "SCMP_ACT_ALLOW"
    assert {
        "mount",
        "pivot_root",
        "setns",
        "umount",
        "umount2",
        "unshare",
    }.isdisjoint(denied)
    assert {
        "bpf",
        "keyctl",
        "perf_event_open",
        "ptrace",
        "userfaultfd",
    } <= denied
    assert ptrace["excludes"] == {"caps": ["CAP_SYS_PTRACE"]}


async def test_docker_runtime_always_prepares_for_workspace_isolation(
    tmp_path: Path, docker_log: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_fake_docker(tmp_path, port_behavior="echo 127.0.0.1:43210", monkeypatch=monkeypatch)

    async with DockerRuntime("img:tag")(_row()):
        pass

    calls = await _docker_calls(docker_log)
    assert f"seccomp={runtime_module._DOCKER_SECCOMP_PROFILE}" in calls[0]
    assert "seccomp=unconfined" not in calls[0]
    assert "systempaths=unconfined" in calls[0]
    assert "apparmor=unconfined" in calls[0]


def test_compose_network_owner_follows_service_chains_and_stages_its_port(
    tmp_path: Path,
) -> None:
    compose = tmp_path / "compose.yaml"
    compose.write_text(
        "services:\n"
        "  main:\n    image: hud:latest\n    network_mode: service:relay\n"
        "  relay:\n    image: relay:latest\n    network_mode: service:gateway\n"
        "  gateway:\n    image: gateway:latest\n",
        encoding="utf-8",
    )
    config = ComposeConfig.from_file(compose)

    assert config.network_owner("main") == "gateway"
    with ComposeProject(document=compose).stage(
        "127.0.0.1::8765",
        port_service=config.network_owner("main"),
        seccomp="profile.json",
    ) as files:
        assert "  gateway:" in files.ports.read_text("utf-8")


async def test_compose_startup_failure_uses_network_owner_logs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    compose = tmp_path / "compose.yaml"
    compose.write_text(
        "services:\n"
        "  main:\n    image: hud:latest\n    network_mode: service:gateway\n"
        "  gateway:\n    image: gateway:latest\n",
        encoding="utf-8",
    )
    calls: list[tuple[str, ...]] = []

    async def fake_docker(*args: str, **_kwargs: Any) -> tuple[str, str]:
        calls.append(args)
        if args[-4:] == ("logs", "--tail", "40", "gateway"):
            return "gateway failed\n", ""
        return "", ""

    monkeypatch.setattr(runtime_module, "_docker", fake_docker)
    task = Task(
        env="any-env",
        id="t",
        runtime_config=RuntimeConfig(compose=ComposeProject(document=compose)),
    )

    with pytest.raises(RuntimeError, match="Compose gateway service exited") as error:
        async with DockerRuntime()(task):
            pass

    assert "gateway failed" in str(error.value)
    assert any(call[-4:] == ("logs", "--tail", "40", "gateway") for call in calls)


def test_compose_stage_archives_the_normalized_document(tmp_path: Path) -> None:
    project = tmp_path / "project"
    recipe = project / "recipe" / "compose.yaml"
    recipe.parent.mkdir(parents=True)
    (recipe.parent / ".env").write_text("IMAGE=example:1\n", encoding="utf-8")
    recipe.write_text(
        "services:\n  main:\n    image: ${IMAGE}\n    environment:\n      HOME: '$HOME'\n",
        encoding="utf-8",
    )

    with ComposeProject(document=recipe, root=project).stage(
        "127.0.0.1::8765",
        seccomp="profile.json",
        archive=True,
    ) as files:
        assert files.archive is not None
        assert files.project_directory == recipe.parent.resolve()
        assert json.loads(files.compose.read_text("utf-8"))["services"]["main"] == {
            "image": "example:1",
            "environment": {"HOME": "$$HOME"},
            "expose": [],
            "ports": [],
            "volumes": [],
        }
        with tarfile.open(files.archive, "r:gz") as archive:
            archived = archive.extractfile("recipe/compose.yaml")
            assert archived is not None
            assert json.load(archived)["services"]["main"]["image"] == "example:1"

    assert "${IMAGE}" in recipe.read_text("utf-8")


@pytest.mark.parametrize(
    ("services", "message"),
    [
        (
            {
                "main": {"network_mode": "service:relay"},
                "relay": {"network_mode": "service:main"},
            },
            "cycle",
        ),
        ({"main": {"network_mode": "service:missing"}}, "unknown service"),
    ],
)
def test_compose_network_owner_rejects_invalid_graphs(
    services: dict[str, dict[str, str]],
    message: str,
) -> None:
    config = ComposeConfig.model_validate({"services": services})

    with pytest.raises(ValueError, match=message):
        config.network_owner("main")


async def test_docker_rejects_insufficient_free_disk(
    tmp_path: Path,
    docker_log: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_docker(
        tmp_path,
        port_behavior="echo 127.0.0.1:43210",
        monkeypatch=monkeypatch,
    )
    config = RuntimeConfig(image="img:tag", resources=RuntimeResources(storage_mb=8192))

    with pytest.raises(RuntimeError, match=r"requires 8192 MB.*has 4096 MB"):
        async with DockerRuntime(runtime_config=config)(_row()):
            pass

    assert (await _docker_calls(docker_log))[-1] == "rm --force cid-42"
