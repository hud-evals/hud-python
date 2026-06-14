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
from pathlib import Path  # noqa: TC003  # runtime use in _install_fake_docker

import pytest

from hud.eval.runtime import DockerRuntime
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
