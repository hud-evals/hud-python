"""``DockerRuntime()`` provider behavior, driven through a scripted docker CLI.

No daemon needed: a fake ``docker`` executable on PATH records every
invocation and scripts the responses, so these tests pin the provider's
contract — command shape, runtime address, teardown — at the process
boundary.
"""

from __future__ import annotations

import asyncio
import os
import sys
from dataclasses import dataclass
from pathlib import Path  # noqa: TC003  # runtime use in _install_fake_docker
from types import ModuleType, SimpleNamespace

import pytest

from hud.eval.runtime import (
    DaytonaRuntime,
    DockerRuntime,
    ModalRuntime,
    RuntimeConfig,
    RuntimeGPU,
    RuntimeLimits,
    RuntimeResources,
)
from hud.eval.task import Task

FAKE_DOCKER_SH = """\
#!/bin/sh
echo "$@" >> "$DOCKER_LOG"
case "$1" in
  run) echo cid-42 ;;
  port) {port_behavior} ;;
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
        import hud.eval.runtime as runtime_module

        async def _docker(*args: str, check: bool = True) -> tuple[str, str]:
            return await _docker_via(exe, *args, check=check)

        monkeypatch.setattr(runtime_module, "_docker", _docker)
        return

    exe = tmp_path / "docker"
    exe.write_text(FAKE_DOCKER_SH.format(port_behavior=port_behavior))
    exe.chmod(0o755)


def _row() -> Task:
    return Task(env="any-env", id="t")


@dataclass(frozen=True)
class _ModalImageRef:
    kind: str
    name: str


class _FakeModalSandbox:
    object_id = "sb-1"

    def __init__(self, calls: dict[str, object], port: int) -> None:
        self._calls = calls
        self._port = port
        self.wait_until_ready = SimpleNamespace(aio=self._wait_until_ready)
        self.tunnels = SimpleNamespace(aio=self._tunnels)
        self.terminate = SimpleNamespace(aio=self._terminate)

    async def _wait_until_ready(self, **kwargs: object) -> None:
        self._calls["ready_timeout"] = kwargs["timeout"]

    async def _tunnels(self) -> dict[int, SimpleNamespace]:
        return {self._port: SimpleNamespace(tcp_socket=("modal.host", 4567))}

    async def _terminate(self) -> None:
        self._calls["terminated"] = True


def _install_fake_modal(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    calls: dict[str, object] = {}
    modal = ModuleType("modal")

    class Image:
        @staticmethod
        def from_name(name: str) -> _ModalImageRef:
            calls["image_name"] = name
            return _ModalImageRef("name", name)

        @staticmethod
        def from_registry(name: str) -> _ModalImageRef:
            calls["registry_image"] = name
            return _ModalImageRef("registry", name)

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


@dataclass(frozen=True)
class _CreateSnapshotParams:
    name: str
    image: object


@dataclass(frozen=True)
class _CreateSandboxFromImageParams:
    image: object
    ephemeral: bool
    resources: object | None = None


@dataclass(frozen=True)
class _DaytonaImage:
    name: str


@dataclass(frozen=True)
class _DaytonaResources:
    cpu: float | None = None
    memory: int | None = None
    gpu: int | None = None
    gpu_type: list[object] | None = None


@dataclass(frozen=True)
class _DaytonaGpuType:
    name: str


@dataclass(frozen=True)
class _SessionExecuteRequest:
    command: str
    run_async: bool


class _FakeDaytonaProcess:
    def __init__(self, calls: dict[str, object]) -> None:
        self._calls = calls

    async def create_session(self, session: str) -> None:
        self._calls["session"] = session

    async def execute_session_command(self, session: str, request: object) -> None:
        self._calls["execute"] = (session, request)


class _FakeDaytonaSandbox:
    id = "sandbox-1"

    def __init__(self, calls: dict[str, object]) -> None:
        self._calls = calls
        self.process = _FakeDaytonaProcess(calls)

    async def create_ssh_access(self, *, expires_in_minutes: int) -> SimpleNamespace:
        self._calls["ssh_expires"] = expires_in_minutes
        return SimpleNamespace(token="ssh-token")


class _FakeDaytonaClient:
    def __init__(self, calls: dict[str, object]) -> None:
        self.calls = calls
        self.sandbox = _FakeDaytonaSandbox(calls)

    async def create(self, params: object, **kwargs: object) -> _FakeDaytonaSandbox:
        self.calls["create"] = (params, kwargs["timeout"])
        return self.sandbox

    async def delete(self, sandbox: _FakeDaytonaSandbox) -> None:
        self.calls["delete"] = sandbox.id


class _FakeSSHConnection:
    def __init__(self, calls: dict[str, object]) -> None:
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
        calls: dict[str, object],
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


def _install_fake_daytona(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    calls: dict[str, object] = {}
    client = _FakeDaytonaClient(calls)
    daytona = ModuleType("daytona")
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
    setattr(asyncssh, "connect", connect)
    monkeypatch.setitem(sys.modules, "daytona", daytona)
    monkeypatch.setitem(sys.modules, "asyncssh", asyncssh)
    return calls


async def test_acquisition_publishes_ephemeral_port_and_removes_container(
    tmp_path: Path, docker_log: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_fake_docker(tmp_path, port_behavior="echo 127.0.0.1:43210", monkeypatch=monkeypatch)

    provider = DockerRuntime("img:tag", run_args=("-e", "X=1"))
    async with provider(_row()) as runtime:
        assert runtime.url == "tcp://127.0.0.1:43210"
        calls = docker_log.read_text().splitlines()
        assert calls[0] == "run --detach -e X=1 --publish 127.0.0.1::8765 img:tag"
        assert calls[1] == "port cid-42 8765"

    assert docker_log.read_text().splitlines()[-1] == "rm --force cid-42"


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

    calls = docker_log.read_text().splitlines()
    assert calls[0] == (
        "run --detach --cpus 2 --memory 4096m --gpus 1 --publish 127.0.0.1::8765 img:firefox"
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

    assert docker_log.read_text().splitlines()[0] == (
        "run --detach --cpus 2 --memory 4096m --publish 127.0.0.1::8765 img:task"
    )


def test_runtime_config_overrides_only_explicit_top_level_fields() -> None:
    default = RuntimeConfig(
        resources=RuntimeResources(
            cpu=2,
            memory_mb=4096,
            gpu=RuntimeGPU(type="A10G", count=2),
        ),
        limits=RuntimeLimits(startup_timeout_s=30, run_timeout_s=120),
    )

    assert default.with_overrides(RuntimeConfig(image="img:task")) == RuntimeConfig(
        image="img:task",
        resources=RuntimeResources(
            cpu=2,
            memory_mb=4096,
            gpu=RuntimeGPU(type="A10G", count=2),
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


async def test_runtime_config_rejects_unsupported_docker_fields() -> None:
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

    with pytest.raises(ValueError, match="limits"):
        async with DockerRuntime()(
            Task(
                env="any-env",
                id="t",
                runtime_config=RuntimeConfig(
                    image="img",
                    limits=RuntimeLimits(run_timeout_s=60),
                ),
            )
        ):
            pass


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
        "unencrypted_ports": [8765],
        "readiness_probe": ("tcp", 8765),
        "timeout": 120,
        "cpu": 2,
        "memory": 4096,
        "gpu": "A10G:2",
    }
    assert calls["ready_timeout"] == 30
    assert calls["terminated"] is True


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


async def test_daytona_runtime_config_flows_into_daytona_sdk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _install_fake_daytona(monkeypatch)
    config = RuntimeConfig(
        image="img:tag",
        resources=RuntimeResources(
            cpu=2,
            memory_mb=4096,
            gpu=RuntimeGPU(type="H100", count=2),
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
        resources=_DaytonaResources(
            cpu=2,
            memory=4,
            gpu=2,
            gpu_type=[_DaytonaGpuType("H100")],
        ),
    )
    assert create_timeout == 45
    assert calls["session"] == "hud-serve"
    assert calls["execute"] == (
        "hud-serve",
        _SessionExecuteRequest(
            command="cd /app && hud serve env.py --host 0.0.0.0 --port 8765",
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
    assert calls["client_closed"] is True


async def test_daytona_task_runtime_config_overlays_provider_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _install_fake_daytona(monkeypatch)
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
        resources=_DaytonaResources(cpu=2, memory=4),
    )
    assert create_timeout == 45


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
    calls = docker_log.read_text().splitlines()
    assert "logs --tail 40 cid-42" in calls
    assert calls[-1] == "rm --force cid-42"  # cleanup still runs on failure
