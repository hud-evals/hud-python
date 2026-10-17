"""A workspace publishes its ``ssh/2`` binding when the environment serves, never before.

``env.workspace(root)`` is pure declaration: keys, sockets and the directory
materialize at serve time, the manifest carries the published address, and
stopping the environment takes the daemon down.
"""

from __future__ import annotations

import asyncio
import stat
import tempfile
from pathlib import Path
from typing import Any, cast
from urllib.parse import urlsplit

import asyncssh
import pytest

from hud.capabilities import Capability
from hud.clients import connect
from hud.environment import Environment, Workspace
from hud.environment.workspace import usable_bwrap
from hud.eval import LocalRuntime, Task
from tests.support import HudEnv, served

from .conftest import SANDBOX, shell

TEMP = Path(tempfile.gettempdir()).resolve()


def test_attaching_a_workspace_writes_nothing(tmp_path: Path) -> None:
    env = Environment("pure")
    env.workspace(tmp_path / "root")

    assert env.capabilities == []
    assert not (tmp_path / "root").exists()


@pytest.mark.parametrize(
    "isolation",
    [
        pytest.param("none", id="unsandboxed-host"),
        pytest.param("bwrap", id="sandboxed-host", marks=SANDBOX),
    ],
)
async def test_serving_publishes_the_shell_and_file_tracking_bindings(
    tmp_path: Path, hud_env: HudEnv, isolation: str
) -> None:
    if isolation == "none" and usable_bwrap() is not None:
        pytest.skip("this host sandboxes every workspace")
    hud_env.set(HUD_FILE_TRACKING_ENABLED="1")
    env = Environment("ws-env")
    env.workspace(tmp_path / "root")

    async with served(env) as client:
        assert client.manifest is not None
        (declared,) = [cap for cap in client.manifest.bindings if cap.name == "shell"]
        shell_binding = client.binding("shell")
        tracking = client.binding("filetracking")

    assert declared.params["isolation"] == isolation
    assert (shell_binding.protocol, urlsplit(shell_binding.url).scheme) == ("ssh/2", "ssh")
    assert shell_binding.params["host_pubkey"].startswith("ssh-ed25519")
    assert tracking.protocol == "filetracking/1"
    assert tracking.params == {"root": (tmp_path / "root").resolve().as_posix(), "setup_diff": True}


async def test_workspace_file_tracking_can_be_opted_out(tmp_path: Path, hud_env: HudEnv) -> None:
    hud_env.set(HUD_FILE_TRACKING_ENABLED="1")
    env = Environment("ws-env")
    env.workspace(tmp_path / "root", track_files=False)

    async with served(env) as client:
        assert client.binding("shell").protocol == "ssh/2"
        with pytest.raises(KeyError):
            client.binding("filetracking")


async def test_reconnecting_reuses_the_same_workspace(tmp_path: Path) -> None:
    env = Environment("ws-env")
    env.workspace(tmp_path / "root")

    # Client-side urls are per-connection (forwarded); the daemon's identity is
    # its host key, which only stays stable if the workspace is reused.
    async with LocalRuntime(env)(Task(env=env.name, id="test")) as runtime:
        async with connect(runtime) as client:
            first = client.binding("shell").params["host_pubkey"]
        async with connect(runtime) as client:
            assert client.binding("shell").params["host_pubkey"] == first


async def test_restarting_replaces_the_published_address_without_duplicates(
    tmp_path: Path, hud_env: HudEnv
) -> None:
    hud_env.set(HUD_FILE_TRACKING_ENABLED="1")
    env = Environment("ws-env")
    env.workspace(tmp_path / "root")

    async with served(env):
        pass
    async with served(env):
        assert [c.name for c in env.capabilities] == ["shell", "filetracking"]


async def test_stop_tears_down_the_workspace(tmp_path: Path) -> None:
    env = Environment("ws-env")
    env.workspace(tmp_path / "root")

    async with served(env):
        # The substrate-local address; the manifest carries a forwarded one.
        backing_port = urlsplit(env.capability("shell").url).port
        assert backing_port is not None

    with pytest.raises(OSError):
        _, writer = await asyncio.open_connection("127.0.0.1", backing_port)
        writer.close()


async def test_start_returns_once_the_ssh_acceptor_is_ready(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    listen_started = asyncio.Event()
    allow_listen = asyncio.Event()
    listen = asyncssh.listen

    async def delayed_listen(*args: Any, **kwargs: Any) -> asyncssh.SSHAcceptor:
        listen_started.set()
        await allow_listen.wait()
        return await listen(*args, **kwargs)

    monkeypatch.setattr(asyncssh, "listen", delayed_listen)
    ws = Workspace(tmp_path / "root", track_files=False)
    start = asyncio.create_task(ws.start())
    await asyncio.wait_for(listen_started.wait(), 1.0)

    assert not start.done()

    allow_listen.set()
    await start
    try:
        key_path = ws.ssh_client_key_path
        assert key_path is not None
        host, port = ws.ssh_url.removeprefix("ssh://").split(":")
        async with asyncssh.connect(
            host, int(port), username=ws.ssh_user, client_keys=[str(key_path)], known_hosts=None
        ) as conn:
            assert (await conn.run("echo ready")).stdout == "ready\n"
    finally:
        await ws.stop()


async def test_session_credentials_live_privately_outside_the_served_root(tmp_path: Path) -> None:
    """The agent's shell root must not hold its own SSH key material."""
    async with shell(tmp_path / "root") as (workspace, client):
        credentials = Path(client.capability.params["client_key_path"]).parent
        mode = stat.S_IMODE(credentials.stat().st_mode)
        listing = await client.run("ls -A", check=True)
        other = Workspace(tmp_path / "other")
        other_credentials = Path(other.capability().params["client_key_path"]).parent
        await other.stop()

    assert credentials.resolve().is_relative_to(TEMP)
    assert not credentials.is_relative_to(workspace.root)
    assert mode == 0o700
    assert other_credentials != credentials
    assert listing.stdout == ""
    assert not credentials.exists()


async def test_the_ssh_daemon_serves_no_sftp_subsystem(tmp_path: Path) -> None:
    async with shell(tmp_path / "root") as (_, client):
        with pytest.raises(asyncssh.ChannelOpenError):
            await client.conn.start_sftp_client()


async def test_manifest_keeps_environment_addresses_and_client_bindings_are_forwarded() -> None:
    local = Capability.cdp(name="browser", url="ws://127.0.0.1:9222")
    remote = Capability.ssh(name="box", url="ssh://box.example.com:22", host_pubkey="ssh-ed25519 x")
    env = Environment("mixed-env", capabilities=[local, remote])

    async with served(env) as client:
        assert client.manifest is not None
        assert client.manifest.bindings == [local, remote]
        for declared in (local, remote):
            forwarded = client.binding(declared.name)
            assert forwarded.url != declared.url
            assert urlsplit(forwarded.url).hostname == "127.0.0.1"
            assert (forwarded.name, forwarded.protocol) == (declared.name, declared.protocol)


@pytest.mark.parametrize(
    ("capabilities", "error", "message"),
    [
        pytest.param(
            [Capability(name="b", protocol="cdp/1.3", url="")],
            ValueError,
            "initialize hook",
            id="a-binding-without-an-address",
        ),
        pytest.param([object()], TypeError, "expected Capability", id="not-a-capability"),
    ],
)
def test_a_declared_capability_the_env_cannot_publish_is_refused(
    capabilities: list[Any], error: type[Exception], message: str
) -> None:
    with pytest.raises(error, match=message):
        Environment("bad", capabilities=cast("list[Capability]", capabilities))
