"""Env-run daemons publish capabilities at serve time, never at declaration.

``env.workspace(root)`` (and, generally, ``env.add_capability(...)`` from an
``@env.initialize`` hook) defers everything — keys, sockets, the directory —
until the env actually serves. The manifest carries the published address,
and ``env.stop()`` runs the matching shutdown hooks.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import signal
import subprocess
import sys
from typing import TYPE_CHECKING, cast

import pytest

from hud.capabilities import Capability
from hud.environment import Environment

from .conftest import served

if TYPE_CHECKING:
    from pathlib import Path

    from hud.capabilities.ssh import SSHClient


def test_attaching_a_workspace_writes_nothing(tmp_path: Path) -> None:
    env = Environment("pure")
    env.workspace(tmp_path / "root")

    assert env.capabilities == []  # published at serve time, not declaration
    assert not (tmp_path / "root").exists()


async def test_serving_publishes_the_workspace_capability(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from hud.settings import settings

    monkeypatch.setattr(settings, "file_tracking_enabled", True)
    env = Environment("ws-env")
    env.workspace(tmp_path / "root")

    async with served(env) as client:
        cap = client.binding("shell")
        assert cap.protocol == "ssh/2"
        assert cap.url.startswith("ssh://")
        assert cap.params["host_pubkey"].startswith("ssh-ed25519")
        filetracking = client.binding("filetracking")
        assert filetracking.protocol == "filetracking/1"
        assert filetracking.params == {
            "root": (tmp_path / "root").resolve().as_posix(),
            "setup_diff": True,
        }
        # Key material lives outside the served root (the agent's surface).
        assert not (tmp_path / "root" / ".hud").exists()


async def test_workspace_file_tracking_can_be_opted_out(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from hud.settings import settings

    monkeypatch.setattr(settings, "file_tracking_enabled", True)
    env = Environment("ws-env")
    env.workspace(tmp_path / "root", track_files=False)

    async with served(env) as client:
        assert client.binding("shell").protocol == "ssh/2"
        with pytest.raises(KeyError):
            client.binding("filetracking")


async def test_reconnecting_reuses_the_same_workspace(tmp_path: Path) -> None:
    from hud.clients import connect
    from hud.eval.runtime import _local

    env = Environment("ws-env")
    env.workspace(tmp_path / "root")

    # Client-side urls are per-connection (forwarded); the daemon's identity
    # is its host key, which only stays stable if the workspace is reused.
    async with _local(env) as runtime:
        async with connect(runtime) as client:
            first = client.binding("shell").params["host_pubkey"]
        async with connect(runtime) as client:
            assert client.binding("shell").params["host_pubkey"] == first


def _pid_status(pid: int) -> str | None:
    result = subprocess.run(
        ["ps", "-o", "stat=", "-p", str(pid)],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    return result.stdout.strip() or None


def _pid_is_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    status = _pid_status(pid)
    if status is None:
        return False
    return not status.startswith("Z")


async def _wait_for_pid_inactive(pid: int, max_wait: float = 2.0) -> bool:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + max_wait
    while loop.time() < deadline:
        if not _pid_is_running(pid):
            return True
        await asyncio.sleep(0.05)
    return not _pid_is_running(pid)


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX process-group regression")
async def test_workspace_command_teardown_kills_background_children(tmp_path: Path) -> None:
    env = Environment("ws-env")
    env.workspace(tmp_path / "root", track_files=False)
    pid_file = tmp_path / "root" / "child.pid"
    pid: int | None = None

    try:
        async with served(env) as client:
            ssh = cast("SSHClient", await client.open("shell"))
            await ssh.conn.run(
                "sleep 120 >/dev/null 2>&1 < /dev/null & echo $! > child.pid",
                check=True,
            )
            pid = int(pid_file.read_text())
            assert await _wait_for_pid_inactive(pid)
    finally:
        if pid is not None and _pid_is_running(pid):
            with contextlib.suppress(ProcessLookupError):
                os.kill(pid, signal.SIGKILL)


async def test_stop_tears_down_the_workspace(tmp_path: Path) -> None:
    import asyncio
    from urllib.parse import urlsplit

    env = Environment("ws-env")
    env.workspace(tmp_path / "root")

    async with served(env):
        # The substrate-local address (the manifest carries a forwarded one).
        backing_port = urlsplit(env.capability("shell").url).port
        assert backing_port is not None

    with pytest.raises(OSError):
        _, writer = await asyncio.open_connection("127.0.0.1", backing_port)
        writer.close()


async def test_restarting_replaces_the_published_address_without_duplicates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from hud.settings import settings

    monkeypatch.setattr(settings, "file_tracking_enabled", True)
    env = Environment("ws-env")
    env.workspace(tmp_path / "root")

    async with served(env):
        pass
    async with served(env):
        assert [c.name for c in env.capabilities] == ["shell", "filetracking"]


async def test_any_initialize_hook_can_publish_a_capability() -> None:
    """Publication is protocol-agnostic: no SDK type per daemon kind."""
    import asyncio

    server = await asyncio.start_server(lambda r, w: w.close(), "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    torn_down = False

    env = Environment("browser-env")

    @env.initialize
    async def _up() -> None:
        env.add_capability(Capability.cdp(name="browser", url=f"ws://127.0.0.1:{port}"))

    @env.shutdown
    async def _down() -> None:
        nonlocal torn_down
        torn_down = True
        server.close()

    assert env.capabilities == []
    async with served(env) as client:
        cap = client.binding("browser")
        assert cap.protocol == "cdp/1.3"
        assert cap.url.startswith("ws://127.0.0.1:")
    assert torn_down  # env.stop() ran the shutdown hook


async def test_loopback_declarations_are_forwarded_and_remote_ones_pass_through() -> None:
    local = Capability.cdp(name="browser", url="ws://127.0.0.1:9222")
    remote = Capability.ssh(name="box", url="ssh://box.example.com:22", host_pubkey="ssh-ed25519 x")
    env = Environment("mixed-env", capabilities=[local, remote])

    async with served(env) as client:
        forwarded = client.binding("browser")
        # Loopback means substrate-local: the client substitutes a local
        # forwarder address, everything else about the capability intact.
        assert forwarded.url != local.url
        assert forwarded.url.startswith("ws://127.0.0.1:")
        assert (forwarded.name, forwarded.protocol) == (local.name, local.protocol)
        # Globally-reachable addresses are the client's to dial directly.
        assert client.binding("box") == remote


def test_a_capability_without_a_url_is_rejected() -> None:
    with pytest.raises(ValueError, match="initialize hook"):
        Environment("bad", capabilities=[Capability(name="b", protocol="cdp/1.3", url="")])


def test_non_capability_entries_are_rejected() -> None:
    with pytest.raises(TypeError, match="expected Capability"):
        Environment("bad", capabilities=cast("list[Capability]", [object()]))
