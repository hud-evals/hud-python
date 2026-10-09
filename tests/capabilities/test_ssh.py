"""``SSHClient`` against a served workspace, through a relay that fails on demand.

Every test serves a real ``env.workspace`` on loopback and connects to its
``ssh`` binding through :func:`tests.harness.relay`, which can sever the
connection mid-command, refuse reconnects, or stall a handshake.
"""

from __future__ import annotations

import asyncio
import os
import sys
from contextlib import asynccontextmanager
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import urlsplit

import pytest

from hud import Environment
from hud.capabilities import Connection, SSHClient
from tests.harness import FlakyRelay, eventually, relay, served

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

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


def running(pid: int) -> bool:
    """Whether ``pid`` is a live process (an unreaped zombie counts as gone)."""
    stat = Path(f"/proc/{pid}/stat")
    try:
        return stat.read_text().split(") ", 1)[1][0] != "Z"
    except FileNotFoundError:
        return False


async def remote_pid(root: Path) -> int:
    """The pid a command wrote to ``pid`` in the workspace."""
    await eventually(lambda: written(root / "pid"))
    return int((root / "pid").read_text())


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
    scripts = log.read_text("utf-8").splitlines()
    assert [script.split("(", 1)[0] for script in scripts] == [
        "[IO.File]::WriteAllBytes",
        "$b=[Convert]::FromBase64String",
        "$b=[Convert]::FromBase64String",
        "[Convert]::ToBase64String",
        "Get-ChildItem -Force -Name -LiteralPath '" + str(root) + "'",
    ]
    assert f"'{root}/it''s.txt'" in scripts[0]


async def test_a_timed_out_command_is_killed_and_the_next_one_runs(tmp_path: Path) -> None:
    async with shell(tmp_path) as (client, _, root):
        with pytest.raises(TimeoutError):
            await client.run("echo $$ > pid; exec sleep 30", timeout=0.5)
        pid = await remote_pid(root)
        await eventually(lambda: not running(pid))

        result = await client.run("echo next", timeout=10)

    assert result.stdout == "next\n"


async def test_a_cancelled_command_kills_its_remote_process(tmp_path: Path) -> None:
    async with shell(tmp_path) as (client, _, root):
        command = asyncio.create_task(client.run("echo $$ > pid; exec sleep 30"))
        pid = await remote_pid(root)

        command.cancel()
        with pytest.raises(asyncio.CancelledError):
            await command

        await eventually(lambda: not running(pid))


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


async def test_a_handshake_the_server_drops_is_a_connection_error(tmp_path: Path) -> None:
    async with shell(tmp_path) as (client, flaky, _):
        flaky.refuse(1)

        with pytest.raises(ConnectionError, match="SSH connection failed during handshake"):
            await SSHClient.connect(client.capability)


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
