"""TensorlakeRuntime: a row boots in a Tensorlake sandbox, reached over its tunnel.

CI has no Tensorlake account, so a fake ``tensorlake.sandbox`` module records each SDK
call. The rows pin what the provider asks Tensorlake for and what it tears down.
"""

from __future__ import annotations

import asyncio
import logging
import sys
from dataclasses import dataclass
from types import ModuleType, SimpleNamespace
from typing import TYPE_CHECKING, Any

import pytest

from hud.eval.runtime import (
    RuntimeConfig,
    RuntimeGPU,
    RuntimeLimits,
    RuntimeResources,
    TensorlakeRuntime,
)
from hud.eval.runtime.compose import ComposeProject
from hud.eval.task import Task

if TYPE_CHECKING:
    from pathlib import Path


def _row() -> Task:
    return Task(env="any-env", id="t")


@dataclass(frozen=True)
class _TensorlakeGpuRequest:
    count: int
    model: str


class _FakeTensorlakeTunnel:
    local_host = "127.0.0.1"
    local_port = 54321

    def __init__(self, calls: dict[str, Any]) -> None:
        self._calls = calls

    async def close(self) -> None:
        self._calls["tunnel_closed"] = True


class _FakeTensorlakeSandbox:
    def __init__(self, calls: dict[str, Any], sandbox_id: str, *, terminate_fails: bool) -> None:
        self.sandbox_id = sandbox_id
        self._calls = calls
        self._terminate_fails = terminate_fails
        self.output_lines = ["ImportError: no module named bugs"]
        calls["sandbox"] = self

    async def start_process(
        self, command: str, args: list[str], *, working_dir: str | None
    ) -> SimpleNamespace:
        self._calls["start_process"] = (command, args, working_dir)
        return SimpleNamespace(pid=7)

    async def create_tunnel(self, remote_port: int, *, local_port: int) -> _FakeTensorlakeTunnel:
        self._calls["tunnel"] = (remote_port, local_port)
        return _FakeTensorlakeTunnel(self._calls)

    async def get_output(self, pid: int) -> SimpleNamespace:
        self._calls["get_output"] = pid
        lines = list(self.output_lines)
        return SimpleNamespace(pid=pid, lines=lines, line_count=len(lines))

    async def terminate(self) -> None:
        if self._terminate_fails:
            raise RuntimeError("tensorlake API unreachable")
        self._calls["terminated"] = self.sandbox_id


class _FakeTensorlakeSandboxPending(Exception):
    def __init__(self, sandbox_id: str, *, pending_reason: str, timeout: float | None) -> None:
        super().__init__(f"Sandbox {sandbox_id} is still pending; it keeps its place in the queue")
        self.sandbox_id = sandbox_id
        self.pending_reason = pending_reason
        self.timeout = timeout


class _FakeTensorlakePendingSandbox:
    def __init__(self, client: _FakeTensorlakeClient, sandbox_id: str) -> None:
        self.sandbox_id = sandbox_id
        self._client = client

    async def ready(self, budget: float | None) -> _FakeTensorlakeSandbox:
        self._client.calls["ready_timeout"] = budget
        if self._client.startup == "queued":
            raise _FakeTensorlakeSandboxPending(
                self.sandbox_id, pending_reason="no_resources_available", timeout=budget
            )
        if self._client.startup == "never":
            await asyncio.Event().wait()
        return _FakeTensorlakeSandbox(
            self._client.calls, self.sandbox_id, terminate_fails=self._client.terminate_fails
        )


class _FakeTensorlakeClient:
    def __init__(self) -> None:
        self.calls: dict[str, Any] = {}
        self.created: list[dict[str, Any]] = []
        self.terminate_fails = False
        # "running" starts at once; "queued" is still pending when the wait
        # budget runs out; "never" waits for capacity until cancelled.
        self.startup = "running"

    async def __aenter__(self) -> _FakeTensorlakeClient:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        self.calls["client_closed"] = True

    async def create(self, **kwargs: Any) -> _FakeTensorlakePendingSandbox:
        self.created.append(kwargs)
        return _FakeTensorlakePendingSandbox(self, f"sandbox-{len(self.created)}")

    async def delete(self, sandbox_id: str) -> None:
        self.calls["deleted"] = sandbox_id


def _install_fake_tensorlake(monkeypatch: pytest.MonkeyPatch) -> _FakeTensorlakeClient:
    client = _FakeTensorlakeClient()
    tensorlake = ModuleType("tensorlake")
    sandbox_module = ModuleType("tensorlake.sandbox")

    class GpuModel(str):
        def __new__(cls, value: str) -> GpuModel:
            if value not in {"H100", "A100-80GB"}:
                raise ValueError(f"{value!r} is not a valid GpuModel")
            return super().__new__(cls, value)

    setattr(sandbox_module, "AsyncSandboxClient", SimpleNamespace(for_cloud=lambda: client))
    setattr(sandbox_module, "GpuModel", GpuModel)
    setattr(sandbox_module, "GpuRequest", _TensorlakeGpuRequest)
    setattr(sandbox_module, "SandboxPending", _FakeTensorlakeSandboxPending)
    setattr(tensorlake, "sandbox", sandbox_module)
    monkeypatch.setitem(sys.modules, "tensorlake", tensorlake)
    monkeypatch.setitem(sys.modules, "tensorlake.sandbox", sandbox_module)
    return client


async def test_tensorlake_runtime_config_flows_into_tensorlake_sdk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _install_fake_tensorlake(monkeypatch)
    config = RuntimeConfig(
        image="hud-env",
        resources=RuntimeResources(
            cpu=1.5,
            memory_mb=4096,
            storage_mb=10240,
            gpu=RuntimeGPU(type="H100", count=2),
        ),
        limits=RuntimeLimits(startup_timeout_s=45, run_timeout_s=600),
    )
    provider = TensorlakeRuntime(runtime_config=config)

    async with provider(_row()) as runtime:
        assert runtime.url == "tcp://127.0.0.1:54321"
        assert runtime.params == {"provider": "tensorlake", "instance_id": "sandbox-1"}
        assert runtime.config == config
        assert "tunnel_closed" not in client.calls

    assert client.created == [
        {
            "image": "hud-env",
            "cpus": 1.5,
            "memory_mb": 4096,
            "disk_mb": 10240,
            "gpu": _TensorlakeGpuRequest(count=2, model="H100"),
            "timeout_secs": 600,
            "wait": False,
        }
    ]
    assert client.calls["ready_timeout"] == 45
    assert client.calls["start_process"] == (
        "sh",
        ["-c", 'PATH="$PWD/.venv/bin:$PATH" hud serve env.py --host 0.0.0.0 --port 8765'],
        "/app",
    )
    assert client.calls["tunnel"] == (8765, 0)
    assert client.calls["tunnel_closed"] is True
    assert client.calls["terminated"] == "sandbox-1"
    assert client.calls["client_closed"] is True


async def test_tensorlake_task_runtime_config_overlays_provider_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _install_fake_tensorlake(monkeypatch)
    provider = TensorlakeRuntime(
        "provider-image",
        runtime_config=RuntimeConfig(resources=RuntimeResources(cpu=2)),
    )
    task = Task(env="any-env", id="t", runtime_config=RuntimeConfig(image="task-image"))

    async with provider(task) as runtime:
        assert runtime.config == RuntimeConfig(
            image="task-image", resources=RuntimeResources(cpu=2)
        )

    assert client.created == [{"image": "task-image", "cpus": 2, "wait": False}]


async def test_tensorlake_boots_a_bare_image_name_with_sdk_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _install_fake_tensorlake(monkeypatch)

    async with TensorlakeRuntime("hud-env", workdir=None, port=9000)(_row()) as runtime:
        assert runtime.config is None

    assert client.created == [{"image": "hud-env", "wait": False}]
    assert client.calls["ready_timeout"] is None
    assert client.calls["start_process"] == (
        "sh",
        ["-c", 'PATH="$PWD/.venv/bin:$PATH" hud serve env.py --host 0.0.0.0 --port 9000'],
        None,
    )
    assert client.calls["tunnel"] == (9000, 0)


async def test_tensorlake_requires_an_image(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _install_fake_tensorlake(monkeypatch)

    with pytest.raises(ValueError, match=r"requires image_name or runtime_config\.image"):
        async with TensorlakeRuntime()(_row()):
            pass

    assert client.created == []


async def test_tensorlake_runtime_rejects_compose(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _install_fake_tensorlake(monkeypatch)
    compose = tmp_path / "compose.yaml"
    compose.write_text("services: {}\n", encoding="utf-8")

    with pytest.raises(ValueError, match=r"does not support runtime_config\.compose"):
        async with TensorlakeRuntime(
            runtime_config=RuntimeConfig(compose=ComposeProject(document=compose))
        )(_row()):
            pass

    assert client.created == []


async def test_tensorlake_runtime_config_rejects_unsupported_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _install_fake_tensorlake(monkeypatch)

    with pytest.raises(ValueError, match=r"resources\.os"):
        async with TensorlakeRuntime(
            "hud-env", runtime_config=RuntimeConfig(resources=RuntimeResources(os="linux"))
        )(_row()):
            pass

    with pytest.raises(ValueError, match="exactly one GPU type"):
        async with TensorlakeRuntime(
            "hud-env",
            runtime_config=RuntimeConfig(resources=RuntimeResources(gpu=RuntimeGPU(count=1))),
        )(_row()):
            pass

    with pytest.raises(ValueError, match="exactly one GPU type"):
        async with TensorlakeRuntime(
            "hud-env",
            runtime_config=RuntimeConfig(
                resources=RuntimeResources(gpu=RuntimeGPU(type=["H100", "A100-80GB"]))
            ),
        )(_row()):
            pass

    with pytest.raises(ValueError, match="not a valid GpuModel"):
        async with TensorlakeRuntime(
            "hud-env",
            runtime_config=RuntimeConfig(resources=RuntimeResources(gpu=RuntimeGPU(type="B300"))),
        )(_row()):
            pass

    assert client.created == []


async def test_tensorlake_streams_env_output_to_terminal(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    client = _install_fake_tensorlake(monkeypatch)

    async with TensorlakeRuntime("hud-env")(_row()):
        await asyncio.sleep(0)
        client.calls["sandbox"].output_lines.append("grader warning")

    # Each line once: the first poll at start, the last poll on exit.
    assert capsys.readouterr().out == "ImportError: no module named bugs\ngrader warning\n"
    assert client.calls["terminated"] == "sandbox-1"


async def test_tensorlake_attaches_the_env_output_to_a_failed_handshake(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The caller gets "closed connection during 'hello'" and nothing else.
    client = _install_fake_tensorlake(monkeypatch)

    with pytest.raises(EOFError) as excinfo:
        async with TensorlakeRuntime("hud-env")(_row()):
            raise EOFError("env closed connection during 'hello'")

    assert client.calls["get_output"] == 7
    assert any("no module named bugs" in note for note in excinfo.value.__notes__)
    assert client.calls["tunnel_closed"] is True
    assert client.calls["terminated"] == "sandbox-1"


async def test_tensorlake_deletes_a_sandbox_still_queued_when_the_rollout_is_cancelled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The rollout deadline fires while the sandbox waits for capacity. It was
    # created, so it bills; the provider must not leave it in the queue.
    client = _install_fake_tensorlake(monkeypatch)
    client.startup = "never"

    async def acquire() -> None:
        async with TensorlakeRuntime("hud-env")(_row()):
            pass

    with pytest.raises(TimeoutError):
        async with asyncio.timeout(0.05):
            await acquire()

    assert client.calls["deleted"] == "sandbox-1"
    assert "terminated" not in client.calls
    assert client.calls["client_closed"] is True


async def test_tensorlake_deletes_a_sandbox_that_missed_its_startup_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _install_fake_tensorlake(monkeypatch)
    client.startup = "queued"
    provider = TensorlakeRuntime(
        "hud-env", runtime_config=RuntimeConfig(limits=RuntimeLimits(startup_timeout_s=30))
    )

    with pytest.raises(TimeoutError, match=r"sandbox-1 .* 30s \(no_resources_available\)"):
        async with provider(_row()):
            pass

    assert client.calls["ready_timeout"] == 30
    assert client.calls["deleted"] == "sandbox-1"
    assert "terminated" not in client.calls


async def test_tensorlake_names_a_sandbox_it_could_not_terminate(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # Teardown must report the leak without shadowing the run's own error.
    client = _install_fake_tensorlake(monkeypatch)
    client.terminate_fails = True

    with caplog.at_level(logging.WARNING, logger="hud.eval.runtime"):
        async with TensorlakeRuntime("hud-env")(_row()):
            pass

    assert "sandbox-1" in caplog.text
