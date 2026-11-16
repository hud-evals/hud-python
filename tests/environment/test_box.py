"""A workspace whose sessions run in a Docker container; needs Docker (``-m e2e``)."""

from __future__ import annotations

import asyncio
import shutil
import subprocess
import sys
from typing import TYPE_CHECKING, cast

import pytest

from hud.capabilities import open_client
from hud.environment import Workspace
from hud.environment.box import BoxSandbox
from hud.runtime.docker import DockerEndpoint

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterator

    from hud.capabilities import SSHClient

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.docker,
    pytest.mark.skipif(sys.platform == "win32", reason="the box is a Linux container"),
]

IMAGE = "python:3.12-slim"
# Every command line in the box, one per line.
LIST_COMMANDS = 'for d in /proc/[0-9]*; do tr "\\0" " " < "$d/cmdline"; echo; done'


@pytest.fixture(scope="module")
def container() -> Iterator[str]:
    if shutil.which("docker") is None:
        pytest.skip("needs a running Docker daemon")
    started = subprocess.run(
        ["docker", "run", "--detach", IMAGE, "sleep", "infinity"],
        capture_output=True,
        text=True,
        check=False,
        timeout=300,
    )
    if started.returncode != 0:
        pytest.skip(f"cannot start a container: {started.stderr.strip()}")
    container_id = started.stdout.strip()
    try:
        yield container_id
    finally:
        subprocess.run(["docker", "rm", "--force", container_id], capture_output=True, check=False)


@pytest.fixture
async def shell(container: str) -> AsyncIterator[SSHClient]:
    box = DockerEndpoint(f"docker://{container}", container=container)
    workspace = Workspace(
        BoxSandbox(box, guest_path="/tmp", user="1000", env={"GREETING": "hello"})
    )
    await workspace.start()
    try:
        async with open_client(workspace.capability()) as client:
            yield cast("SSHClient", client)
    finally:
        await workspace.stop()


def _commands_in(container: str) -> list[str]:
    listed = subprocess.run(
        ["docker", "exec", container, "sh", "-c", LIST_COMMANDS],
        capture_output=True,
        text=True,
        check=True,
    )
    return [line.strip() for line in listed.stdout.splitlines() if line.strip()]


async def test_sessions_run_in_the_box_as_its_user_in_its_directory_with_its_env(
    shell: SSHClient, container: str
) -> None:
    result = await shell.run("id -u; pwd; echo $GREETING; hostname", timeout=60)

    assert result.returncode == 0, result.stderr
    assert str(result.stdout).split() == ["1000", "/tmp", "hello", container[:12]]


async def test_a_session_asking_for_a_terminal_gets_one(shell: SSHClient) -> None:
    process = await shell.conn.create_process("tty", term_type="xterm", encoding="utf-8")
    output, _ = await asyncio.wait_for(process.communicate(), 60)

    assert str(output).strip().startswith("/dev/pts/")


async def test_a_finished_command_keeps_its_work_and_an_abandoned_one_is_stopped_in_the_box(
    shell: SSHClient, container: str
) -> None:
    started = await shell.run("sleep 301 >/dev/null 2>&1 & echo started", timeout=60)
    assert str(started.stdout).strip() == "started"
    with pytest.raises(TimeoutError):
        await shell.run("sleep 401", timeout=2)

    for _ in range(50):
        commands = _commands_in(container)
        if "sleep 401" not in commands:
            break
        await asyncio.sleep(0.2)
    assert "sleep 301" in commands
    assert "sleep 401" not in commands
