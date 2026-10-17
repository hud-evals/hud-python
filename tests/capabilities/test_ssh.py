"""What only ``SSHClient`` does: file transfer, Windows shells, its reconnect policy.

Every test serves a real ``env.workspace`` on loopback and connects to its
``ssh`` binding through :func:`tests.support.relay`, which can sever the
connection mid-command, refuse reconnects, or stall a handshake. The contract
every client keeps (connecting, a drop, a cancelled or timed-out command) is in
``test_clients.py``.
"""

from __future__ import annotations

import asyncio
import os
import sys
from contextlib import asynccontextmanager
from dataclasses import replace
from typing import TYPE_CHECKING
from urllib.parse import urlsplit

import pytest

from hud import Environment
from hud.capabilities import Connection, SSHClient
from hud.clients import connect
from hud.eval.runtime import Runtime
from tests.support import FlakyRelay, answer, control_peer, eventually, relay, served

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="the workspace shell is POSIX")

POWERSHELL = """\
#!{python}
import base64, pathlib, re, sys

script = base64.b64decode(sys.argv[-1]).decode("utf-16-le")
with open({log!r}, "a") as log:
    log.write(script + "\\n")
literals = [text.replace("''", "'") for text in re.findall(r"'((?:[^']|'')*)'", script)]
path = literals[-1]
if "ReadAllBytes" in script:
    print(base64.b64encode(pathlib.Path(path).read_bytes()).decode())
elif "Get-ChildItem" in script:
    print("\\n".join(sorted(entry.name for entry in pathlib.Path(path).iterdir())))
elif "WriteAllBytes" in script:
    pathlib.Path(path).write_bytes(b"")
else:
    with open(path, "ab") as file:
        file.write(base64.b64decode(literals[0]))
"""


@asynccontextmanager
async def shell(
    tmp_path: Path, *, env: dict[str, str] | None = None
) -> AsyncIterator[tuple[SSHClient, FlakyRelay, Path]]:
    """A client connected to a served workspace's ``ssh`` binding through a relay."""
    root = tmp_path / "workspace"
    root.mkdir()
    environment = Environment("ssh-faults")
    environment.workspace(root, track_files=False, env=env)
    async with served(environment) as control:
        binding = control.binding("shell")
        parts = urlsplit(binding.url)
        assert parts.hostname is not None and parts.port is not None
        async with relay(parts.hostname, parts.port) as flaky:
            capability = replace(binding, url=f"ssh://{parts.username}@127.0.0.1:{flaky.port}")
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


async def test_windows_shells_move_files_as_base64_through_encoded_powershell(
    tmp_path: Path,
) -> None:
    tools = tmp_path / "tools"
    tools.mkdir()
    log = tmp_path / "powershell.log"
    stub = tools / "powershell"
    stub.write_text(POWERSHELL.format(python=sys.executable, log=str(log)), encoding="utf-8")
    stub.chmod(0o755)
    async with shell(tmp_path, env={"PATH": f"{tools}{os.pathsep}{os.environ['PATH']}"}) as (
        client,
        _,
        root,
    ):
        windows_client = SSHClient(
            replace(client.capability, params={**client.capability.params, "shell": "powershell"}),
            client.conn,
        )
        content = "x" * 7000 + "é"
        target = str(root / "it's.txt")

        await windows_client.write_text(target, content)
        read = await windows_client.read_text(target)
        listing = await windows_client.listdir(str(root))

    assert (read, listing) == (content, ["it's.txt"])


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


async def test_reconnecting_gives_up_after_three_refused_attempts(tmp_path: Path) -> None:
    async with shell(tmp_path) as (client, flaky, _):
        flaky.refuse(3)
        await drop(client, flaky)

        with pytest.raises(ConnectionError, match="SSH reconnect failed after 3 attempts"):
            await client.run("true")

    assert flaky.accepted == 4


async def test_a_command_timeout_covers_a_stalled_reconnect(tmp_path: Path) -> None:
    async with shell(tmp_path) as (client, flaky, _):
        flaky.stall()
        await drop(client, flaky)

        with pytest.raises(TimeoutError):
            await client.run("true", timeout=0.5)
        flaky.release()
        result = await client.run("echo back", timeout=10)

    assert result.stdout == "back\n"


async def test_closing_during_a_reconnect_discards_the_new_connection(tmp_path: Path) -> None:
    async with shell(tmp_path) as (client, flaky, _):
        flaky.stall()
        await drop(client, flaky)
        command = asyncio.create_task(client.run("true"))
        await flaky.wait_until(stalled=1)

        closing = asyncio.create_task(client.close())
        await asyncio.sleep(0)
        flaky.release()
        with pytest.raises(ConnectionError, match="SSH client is closed"):
            await command
        await closing

        await flaky.wait_until(accepted=2)
        await eventually(lambda: flaky.open == 0)


@pytest.mark.parametrize(
    ("advertised", "connection_capability", "message"),
    [
        (False, "shell", "SSH capability does not support process-bound connections"),
        (True, "browser", "connections do not belong to SSH capability 'shell': inference"),
    ],
)
async def test_process_bound_connections_must_be_advertised_and_belong_to_the_shell(
    advertised: bool, connection_capability: str, message: str, tmp_path: Path
) -> None:
    connection = Connection(
        name="inference",
        capability=connection_capability,
        url="https://inference.example",
        headers={"Authorization": "Bearer secret"},
    )
    async with shell(tmp_path) as (client, _, _):
        bound = SSHClient(
            replace(
                client.capability,
                params={**client.capability.params, "process_connections": advertised},
            ),
            client.conn,
        )

        with pytest.raises(ValueError, match=message):
            await bound.create_process("true", connections=(connection,))


async def test_an_opened_client_retries_an_upstream_that_refuses_and_closes_with_the_connection(
    tmp_path: Path,
) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    env = Environment("tunneled")
    env.workspace(root, track_files=False)
    async with served(env) as workspace:
        shell_binding = workspace.binding("shell")
        address = urlsplit(shell_binding.url)
        assert address.hostname is not None and address.port is not None
        async with relay(address.hostname, address.port) as flaky:
            flaky.refuse(2)
            binding = {**shell_binding.to_manifest(), "url": "ssh://agent@environment:22"}
            hello = {"session_id": "s-1", "env": {"name": "stub", "version": "1.0"}}
            async with control_peer(
                [answer({**hello, "bindings": [binding]})],
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
