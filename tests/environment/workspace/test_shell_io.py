"""What an agent's shell session carries over the SSH wire, and what survives it.

Every scenario connects to a served workspace's ``ssh/2`` binding. The
``backing`` axis runs the same scenario on whatever the host gives (sessions run
directly where bubblewrap is unavailable) and in the sandbox lane, where each
session is spawned by the sandbox's namespace host.
"""

from __future__ import annotations

import asyncio
import shlex
import shutil
import sys
import time
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit

import asyncssh
import pytest

from tests.conftest import SANDBOX, eventually

from .conftest import shell

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable
    from pathlib import Path

    from hud.capabilities import SSHClient
    from hud.environment import Workspace

BACKINGS = [
    pytest.param({}, id="host"),
    pytest.param({"require_isolation": True}, id="sandbox", marks=SANDBOX),
]


async def alive(client: SSHClient, pid: int) -> bool:
    """Whether ``pid``, as the session numbers it, is a live process (an unreaped zombie
    counts as gone). A sandbox numbers its processes in a pid namespace of its own."""
    status = (await client.run(f"ps -o stat= -p {pid}")).stdout
    assert isinstance(status, str)
    return bool(status.strip()) and not status.strip().startswith("Z")


async def test_output_arrives_while_the_command_is_still_running(tmp_path: Path) -> None:
    """Held until exit, a long build tells the agent nothing while it runs and
    a session that never exits says nothing at all."""
    async with shell(tmp_path / "root") as (_, client):
        started = time.monotonic()
        process = await client.conn.create_process("echo first; sleep 5; echo second")
        first = await asyncio.wait_for(process.stdout.readline(), 10)
        elapsed = time.monotonic() - started
        process.channel.close()

    assert first.strip() == "first"
    assert elapsed < 2.0, f"first line took {elapsed:.1f}s — output is not streaming"


PAYLOAD = b"0123456789abcdef" * 131072 + b"tail"
STREAMS = [
    pytest.param(
        "cat; printf stderr-tail >&2",
        PAYLOAD,
        (PAYLOAD, b"stderr-tail", 0),
        id="large-streams-and-the-status-survive",
    ),
    pytest.param(
        "printf done",
        b"x" * 500_000,
        (b"done", b"", 0),
        id="large-input-to-an-early-exit-does-not-hang",
    ),
    pytest.param(
        "head -c 8388608 /dev/zero; head -c 8388608 /dev/zero >&2; exit 7",
        b"",
        (bytes(8388608), bytes(8388608), 7),
        id="output-beyond-every-buffer-drains-before-the-status",
    ),
]


@pytest.mark.parametrize("backing", BACKINGS)
@pytest.mark.parametrize(("command", "stdin", "expected"), STREAMS)
async def test_a_session_carries_its_streams_and_exit_status(
    tmp_path: Path,
    backing: dict[str, Any],
    command: str,
    stdin: bytes,
    expected: tuple[bytes, bytes, int],
) -> None:
    async with shell(tmp_path / "root", **backing) as (_, client):
        async with asyncio.timeout(30):
            result = await client.conn.run(command, input=stdin, encoding=None)

    assert (result.stdout, result.stderr, result.exit_status) == expected


@pytest.mark.parametrize("backing", BACKINGS)
async def test_a_session_that_asks_for_a_terminal_gets_one(
    tmp_path: Path, backing: dict[str, Any]
) -> None:
    """Programs branch on isatty: without a pty they take their batch path, so
    a terminal task is graded on behaviour a terminal would never produce."""
    async with shell(tmp_path / "root", **backing) as (_, client):
        with_pty = await client.conn.run(
            "test -t 0 && test -t 1 && echo TTY || echo NOT_TTY; stty size",
            term_type="xterm-256color",
            term_size=(120, 40),
            check=False,
        )
        without = await client.conn.run("test -t 1 && echo TTY || echo NOT_TTY", check=False)

    assert "TTY" in str(with_pty.stdout) and "NOT_TTY" not in str(with_pty.stdout)
    assert "40 120" in str(with_pty.stdout)
    assert "NOT_TTY" in str(without.stdout)


@pytest.mark.parametrize("backing", BACKINGS)
async def test_a_resize_does_not_cost_the_session_its_keyboard(
    tmp_path: Path, backing: dict[str, Any]
) -> None:
    """Resizing the terminal leaves the session reading its input."""
    async with shell(tmp_path / "root", **backing) as (_, client):
        process = await client.conn.create_process(
            "cat", term_type="xterm-256color", term_size=(80, 24)
        )
        process.channel.change_terminal_size(132, 43)
        await asyncio.sleep(0.2)
        process.stdin.write("still-listening\n")
        echoed = await asyncio.wait_for(process.stdout.readline(), 5)
        process.channel.close()

    assert "still-listening" in echoed


@pytest.mark.parametrize("backing", BACKINGS)
async def test_terminal_output_survives_a_client_that_reads_slowly(
    tmp_path: Path, backing: dict[str, Any]
) -> None:
    async with shell(tmp_path / "root", **backing) as (_, client):
        binding = client.capability
        address = urlsplit(binding.url)
        assert address.hostname is not None and address.port is not None
        async with asyncssh.connect(
            address.hostname,
            address.port,
            username=binding.params["user"],
            client_keys=[asyncssh.import_private_key(binding.params["client_key"])],
            known_hosts=None,
            window=32768,
        ) as conn:
            result = await asyncio.wait_for(
                conn.run(
                    "head -c 500000 /dev/zero | tr '\\000' x",
                    term_type="xterm",
                    encoding=None,
                ),
                30,
            )

    assert (result.exit_status, result.stdout) == (0, b"x" * 500000)


@pytest.mark.e2e
@pytest.mark.sandbox
@pytest.mark.parametrize(
    ("command", "marker"),
    [
        pytest.param(
            "sh -c 'sleep .1; exec >/dev/null 2>&1; : > silent-done' & printf started",
            "silent-done",
            id="inherited-eof",
        ),
        pytest.param(
            "sh -c \"sleep .1; trap '' PIPE; printf late-out || :; "
            "printf late-err >&2 || :; exec >/dev/null 2>&1; "
            ': > late-done" & printf started',
            "late-done",
            id="late-output",
        ),
        pytest.param(
            "cd . && nohup sh -c 'sleep .1' </dev/null >stdin.log 2>&1 && "
            "exec >/dev/null 2>&1 && : > stdin-done & printf started",
            "stdin-done",
            id="stdin-detached",
        ),
        pytest.param(
            "(cd . && exec sh -c 'sleep .1; : > redirected-done') "
            "</dev/null >redirected.log 2>&1 & printf started",
            "redirected-done",
            id="fully-redirected",
        ),
    ],
)
async def test_a_backgrounded_job_finishing_leaves_the_next_sandboxed_command_working(
    tmp_path: Path, command: str, marker: str
) -> None:
    root = tmp_path / "root"
    async with shell(root, require_isolation=True) as (_, client):
        launched = await asyncio.wait_for(client.conn.run(f"bash -c {shlex.quote(command)}"), 10)
        await eventually(lambda: (root / marker).exists())
        probe = await asyncio.wait_for(client.conn.run("printf healthy"), 10)

    assert (launched.exit_status, launched.stdout) == (0, "started")
    assert (probe.exit_status, probe.stdout) == (0, "healthy")


async def output_split_mid_character_is_replaced(
    workspace: Workspace, client: SSHClient, root: Path
) -> None:
    del workspace, root
    completed = await client.run("printf '\\226\\214 split-utf8'", check=True)
    assert completed.stdout == "�� split-utf8"


async def a_channel_closed_while_a_trapped_child_runs(
    workspace: Workspace, client: SSHClient, root: Path
) -> None:
    # The pid file doubles as the child's readiness signal: it is written after
    # the TERM trap is installed, and the foreground command exits only after it.
    # Unsandboxed, teardown's TERM fires once the foreground command exits, so
    # the close lands mid-teardown; a sandboxed session tears down when its
    # channel closes.
    remote = await client.conn.create_process(
        "sh -c 'trap \"echo started > teardown-started; "
        'while :; do sleep 1; done" TERM; '
        "echo $$ > close-race-child.pid; while :; do sleep 1; done' & "
        "while [ ! -s close-race-child.pid ]; do sleep 0.01; done; "
        "printf output"
    )
    pid = 0
    try:
        child = root / "close-race-child.pid"
        await eventually(lambda: child.exists() and bool(child.read_text().strip()))
        pid = int(child.read_text())
        if not workspace.bwrap_available:
            await eventually(lambda: (root / "teardown-started").exists())
        remote.close()
        await asyncio.wait_for(remote.wait_closed(), 5.0)
        await eventually(lambda: gone(client, pid), within=5.0)
    finally:
        remote.close()
        if pid:
            await client.run(f"kill -KILL {pid}")


async def gone(client: SSHClient, pid: int) -> bool:
    return not await alive(client, pid)


@pytest.mark.parametrize(
    "fault",
    [
        pytest.param(output_split_mid_character_is_replaced, id="invalid-utf8-output"),
        pytest.param(
            a_channel_closed_while_a_trapped_child_runs,
            id="channel-closed-while-a-trapped-child-runs",
        ),
    ],
)
@pytest.mark.parametrize("backing", BACKINGS)
async def test_the_connection_survives_a_command_that_goes_wrong(
    tmp_path: Path,
    fault: Callable[[Workspace, SSHClient, Path], Awaitable[None]],
    backing: dict[str, Any],
) -> None:
    root = tmp_path / "root"
    async with shell(root, **backing) as (workspace, client):
        connection = client.conn
        await fault(workspace, client, root)
        after = await client.run("printf healthy", check=True, timeout=5.0)

    assert after.stdout == "healthy"
    assert client.conn is connection


@pytest.mark.parametrize("backing", BACKINGS)
async def test_a_command_returns_although_a_detached_descendant_holds_its_output(
    tmp_path: Path, backing: dict[str, Any]
) -> None:
    root = tmp_path / "root"
    setsid = shutil.which("setsid")
    if setsid is not None:
        detached_command = (
            f"{shlex.quote(setsid)} sh -c "
            "'echo $$ > detached.pid; printf ready; kill -KILL $PPID; sleep 120'"
        )
    else:
        code = (
            "import os,signal,time;"
            "from pathlib import Path;"
            "parent=os.getppid();"
            "os.setsid();"
            "Path('detached.pid').write_text(str(os.getpid()));"
            "os.write(1,b'ready');"
            "os.kill(parent,signal.SIGKILL);"
            "time.sleep(120)"
        )
        detached_command = f"{shlex.quote(sys.executable)} -c {shlex.quote(code)}"

    async with shell(root, **backing) as (_, client):
        remote = await client.conn.create_process(f"{detached_command} & wait")
        try:
            await eventually(
                lambda: bool(
                    (root / "detached.pid").exists() and (root / "detached.pid").read_text().strip()
                )
            )
            pid = int((root / "detached.pid").read_text())
            completed = await remote.wait(timeout=5.0)
        finally:
            remote.close()
            await asyncio.wait_for(remote.wait_closed(), 5.0)
        try:
            survived = await alive(client, pid)
        finally:
            await client.run(f"kill -KILL -- -{pid}")

    assert completed.stdout == "ready"
    assert survived
