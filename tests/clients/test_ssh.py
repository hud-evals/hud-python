"""``SSHClient`` against a real ``env.workspace`` served on loopback.

Most tests connect to the workspace's ``ssh`` binding through
:func:`tests.clients.conftest.relayed`, which can sever the connection
mid-command, refuse reconnects, or stall a handshake.
"""

from __future__ import annotations

import asyncio
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import urlsplit

import pytest

from hud import Environment
from hud.capabilities import SSHClient
from hud.clients import connect
from hud.eval import LocalRuntime, Task
from hud.eval.runtime import Runtime
from tests.clients.conftest import Action, control_peer, relayed
from tests.conftest import eventually
from tests.fixtures.relay import FlakyRelay, relay

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="the workspace shell is POSIX")


@asynccontextmanager
async def shell(tmp_path: Path) -> AsyncIterator[tuple[SSHClient, FlakyRelay, Path]]:
    """A client connected to a served workspace's ``ssh`` binding through a relay."""
    root = tmp_path / "workspace"
    root.mkdir()
    environment = Environment("ssh-faults")
    environment.workspace(root, track_files=False)
    task = Task(env=environment.name, id="scenario")
    async with (
        LocalRuntime(environment)(task) as runtime,
        connect(runtime) as control,
        relayed(control.binding("shell")) as (capability, flaky),
    ):
        client = await SSHClient.connect(capability)
        try:
            yield client, flaky, root
        finally:
            await client.close()


def written(path: Path) -> bool:
    return path.is_file() and bool(path.read_text().strip())


async def drop(client: SSHClient, flaky: FlakyRelay) -> None:
    """Sever the relayed connection and wait until the client has seen it close."""
    flaky.sever()
    await eventually(client.conn.is_closed)


async def test_files_round_trip_through_the_exec_channel(tmp_path: Path) -> None:
    async with shell(tmp_path) as (client, _, root):
        await client.write_text("empty.txt", "")
        await client.write_text("it's here.txt", "héllo\nwörld\n")
        (root / "raw.bin").write_bytes(b"\x00\xff\x10")

        assert (root / "empty.txt").read_text() == ""
        assert await client.read_text("empty.txt") == ""
        assert await client.read_text(str(root / "it's here.txt")) == "héllo\nwörld\n"
        assert await client.read_bytes("raw.bin") == b"\x00\xff\x10"
        assert await client.listdir(".") == ["empty.txt", "it's here.txt", "raw.bin"]


async def test_a_command_lost_in_flight_is_not_replayed_and_the_next_one_reconnects(
    tmp_path: Path,
) -> None:
    async with shell(tmp_path) as (client, flaky, root):
        command = asyncio.create_task(client.run("echo ran >> marker; exec sleep 30"))
        await eventually(lambda: written(root / "marker"))

        flaky.sever()
        with pytest.raises(ConnectionError, match="SSH command ended without an exit status"):
            await command
        result = await client.run("cat marker", timeout=10)
        process = await client.create_process("echo opened")
        opened = await process.wait()

    assert result.stdout == "ran\n"
    assert opened.stdout == b"opened\n"
    assert flaky.accepted == 2


async def test_a_command_timeout_covers_a_stalled_reconnect(tmp_path: Path) -> None:
    async with shell(tmp_path) as (client, flaky, _):
        flaky.stall()
        await drop(client, flaky)

        with pytest.raises(TimeoutError):
            await client.run("true", timeout=0.5)
        flaky.release()
        result = await client.run("echo back", timeout=10)

    assert result.stdout == "back\n"


async def test_an_opened_client_retries_an_upstream_that_refuses_and_closes_with_the_connection(
    tmp_path: Path,
) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    env = Environment("tunneled")
    env.workspace(root, track_files=False)
    task = Task(env=env.name, id="scenario")
    async with LocalRuntime(env)(task) as runtime, connect(runtime) as workspace:
        shell_binding = workspace.binding("shell")
        address = urlsplit(shell_binding.url)
        assert address.hostname is not None and address.port is not None
        async with relay(address.hostname, address.port) as flaky:
            flaky.refuse(2)
            binding = {**shell_binding.to_manifest(), "url": "ssh://agent@environment:22"}
            hello = {"session_id": "s-1", "env": {"name": "stub", "version": "1.0"}}
            async with control_peer(
                [Action("answer", result={**hello, "bindings": [binding]})],
                upstreams={"shell": ("127.0.0.1", flaky.port)},
            ) as peer:
                async with connect(Runtime(peer.url)) as client:
                    opened = await client.open("shell")
                    assert isinstance(opened, SSHClient)
                    result = await opened.run("echo through the tunnel", timeout=10)
                    again = await client.open("ssh")

                assert again is opened
                assert opened.conn.is_closed()
                await eventually(lambda: flaky.open == 0)

    assert result.stdout == "through the tunnel\n"
    assert (flaky.accepted, peer.tunnels) == (3, 3)


def running(pid_file: Path) -> bool:
    """Whether the process whose pid ``pid_file`` holds is live (a zombie counts as gone)."""
    stat = Path(f"/proc/{pid_file.read_text().strip()}/stat")
    try:
        return stat.read_text().split(") ", 1)[1][0] != "Z"
    except FileNotFoundError:
        return False


async def test_a_cancelled_and_a_timed_out_command_end_and_the_next_one_runs(
    tmp_path: Path,
) -> None:
    async with shell(tmp_path) as (client, _, root):
        command = asyncio.create_task(client.run("echo $$ > cancelled; exec sleep 30"))
        await eventually(lambda: written(root / "cancelled"))
        command.cancel()
        with pytest.raises(asyncio.CancelledError):
            await command
        await eventually(lambda: not running(root / "cancelled"))

        with pytest.raises(TimeoutError):
            await client.run("echo $$ > timed_out; exec sleep 30", timeout=0.5)
        await eventually(lambda: written(root / "timed_out"))
        await eventually(lambda: not running(root / "timed_out"))

        result = await client.run("echo ok", timeout=10)

    assert result.stdout == "ok\n"
