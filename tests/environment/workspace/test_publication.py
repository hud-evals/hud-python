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
from urllib.parse import urlsplit

import asyncssh
import pytest

from hud.clients import connect
from hud.environment import Environment, Workspace
from hud.eval import LocalRuntime, Task

from .conftest import shell

TEMP = Path(tempfile.gettempdir()).resolve()


async def test_a_workspace_lives_exactly_while_its_environment_serves(tmp_path: Path) -> None:
    env = Environment("ws-env")
    env.workspace(tmp_path / "root")
    created_before_serving = (tmp_path / "root").exists()

    async with LocalRuntime(env)(Task(env=env.name, id="test")) as runtime:
        async with connect(runtime) as client:
            first = client.binding("shell").params["host_pubkey"]
        async with connect(runtime) as client:
            again = client.binding("shell").params["host_pubkey"]
        backing_port = urlsplit(env.capability("shell").url).port

    assert not created_before_serving
    assert again == first
    with pytest.raises(OSError):
        _, writer = await asyncio.open_connection("127.0.0.1", backing_port)
        writer.close()


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
