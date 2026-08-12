from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import AsyncMock

import asyncssh
import pytest

from hud.capabilities.base import Capability
from hud.capabilities.ssh import SSHClient, SSHConnectionError

if TYPE_CHECKING:
    from collections.abc import Callable


class _Completed:
    stdout = "ok"
    stderr = ""
    exit_status = 0
    returncode: int | None = 0


class _Process:
    def __init__(
        self,
        *,
        completed: _Completed | None = None,
        wait_error: BaseException | None = None,
        block: bool = False,
    ) -> None:
        self.completed = completed or _Completed()
        self.wait_error = wait_error
        self.block = block
        self.on_wait: Callable[[], None] | None = None
        self.started = asyncio.Event()
        self.closed = False
        self.terminated = False
        self.waited_closed = False

    async def wait(self, *, check: bool, **kwargs: Any) -> _Completed:
        assert kwargs == {"timeout": None}
        del check
        self.started.set()
        if self.on_wait is not None:
            self.on_wait()
        if self.block:
            await asyncio.Event().wait()
        if self.wait_error is not None:
            raise self.wait_error
        return self.completed

    def close(self) -> None:
        self.closed = True

    def terminate(self) -> None:
        self.terminated = True

    async def wait_closed(self) -> None:
        self.waited_closed = True


class _Connection:
    def __init__(
        self,
        *,
        closed: bool = False,
        run_error: Exception | None = None,
        process: _Process | None = None,
        stall_open: bool = False,
        open_error: Exception | None = None,
    ) -> None:
        self.closed = closed
        self.run_error = run_error
        self.process = process or _Process()
        self.stall_open = stall_open
        self.open_error = open_error
        self.open_cancelled = False
        self.commands: list[str] = []

    def is_closed(self) -> bool:
        return self.closed

    def close(self) -> None:
        self.closed = True

    async def wait_closed(self) -> None:
        pass

    async def run(self, command: str, **kwargs: Any) -> _Completed:
        del kwargs
        self.commands.append(command)
        if self.run_error is not None:
            if isinstance(self.run_error, asyncssh.ConnectionLost):
                self.closed = True
            raise self.run_error
        return _Completed()

    async def create_process(self, *args: object, **kwargs: Any) -> _Process:
        del kwargs
        self.commands.append(str(args[0]))
        if self.stall_open:
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                self.open_cancelled = True
                raise
        if self.open_error is not None:
            raise self.open_error
        if self.run_error is not None:
            if isinstance(self.run_error, asyncssh.ConnectionLost):
                self.closed = True
            raise self.run_error
        return self.process


def _capability() -> Capability:
    return Capability(
        name="shell",
        protocol="ssh/2",
        url="ssh://workspace.example:2222",
        params={"user": "agent", "client_key_path": "/tmp/key"},
    )


def _client(connection: object) -> SSHClient:
    return SSHClient(_capability(), cast("asyncssh.SSHClientConnection", connection))


async def test_connect_keeps_tunneled_connection_active(monkeypatch: pytest.MonkeyPatch) -> None:
    connection = cast("asyncssh.SSHClientConnection", object())
    connect = AsyncMock(return_value=connection)
    monkeypatch.setattr(asyncssh, "connect", connect)

    client = await SSHClient.connect(
        Capability(name="shell", protocol="ssh/2", url="ssh://sandbox.example:8765")
    )

    assert client.conn is connection
    connect.assert_awaited_once_with(
        host="sandbox.example",
        port=8765,
        username="agent",
        client_keys=None,
        known_hosts=None,
        errors="replace",
        keepalive_interval=15,
        keepalive_count_max=4,
    )


async def test_run_does_not_replay_a_command_lost_in_flight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dropped = _Connection(run_error=asyncssh.ConnectionLost("dropped"))
    replacement = _Connection()
    client = _client(dropped)
    reconnect = AsyncMock(return_value=replacement)
    monkeypatch.setattr(client, "_connect", reconnect)

    with pytest.raises(SSHConnectionError, match="lost during operation"):
        await client.run("apply-side-effect")

    reconnect.assert_not_awaited()
    assert dropped.commands == ["apply-side-effect"]

    await client.run("next-command")
    reconnect.assert_awaited_once_with(client.capability)
    assert replacement.commands == ["next-command"]


async def test_create_process_reconnects_before_opening_channel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dropped = _Connection(closed=True)
    replacement = _Connection()
    client = _client(dropped)
    reconnect = AsyncMock(return_value=replacement)
    monkeypatch.setattr(client, "_connect", reconnect)

    process = await client.create_process("bridge")

    assert process is replacement.process
    reconnect.assert_awaited_once_with(client.capability)
    assert dropped.commands == []
    assert replacement.commands == ["bridge"]


async def test_create_process_preserves_reconnect_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _client(_Connection(closed=True))
    reconnect = AsyncMock(side_effect=OSError("unreachable"))
    monkeypatch.setattr(client, "_connect", reconnect)
    monkeypatch.setattr("hud.capabilities.ssh.asyncio.sleep", AsyncMock())

    with pytest.raises(SSHConnectionError, match="reconnect failed after 3 attempts"):
        await client.create_process("bridge")


async def test_run_classifies_rejected_session_as_connection_error() -> None:
    connection = _Connection(
        open_error=asyncssh.ChannelOpenError(asyncssh.OPEN_RESOURCE_SHORTAGE, "busy")
    )
    client = _client(connection)

    with pytest.raises(SSHConnectionError, match="rejected the session"):
        await client.run("echo never", timeout=1)

    assert connection.closed is False


async def test_run_timeout_includes_opening_the_process() -> None:
    connection = _Connection(stall_open=True)
    client = _client(connection)

    with pytest.raises(TimeoutError):
        await client.run("echo never", timeout=0.01)

    assert connection.open_cancelled is True


async def test_run_timeout_includes_reconnecting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connection = _Connection(closed=True)
    client = _client(connection)
    cancelled = False

    async def reconnect(capability: Capability) -> asyncssh.SSHClientConnection:
        nonlocal cancelled
        assert capability is client.capability
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled = True
            raise
        raise AssertionError("reconnect unexpectedly completed")

    monkeypatch.setattr(client, "_connect", reconnect)

    with pytest.raises(TimeoutError):
        await client.run("echo never", timeout=0.01)

    assert cancelled is True


async def test_run_preserves_timeout_when_the_connection_closes() -> None:
    process = _Process(wait_error=TimeoutError())
    connection = _Connection(process=process)
    process.on_wait = lambda: setattr(connection, "closed", True)
    client = _client(connection)

    with pytest.raises(TimeoutError):
        await client.run("echo never", timeout=1)


@pytest.mark.parametrize("command_timeout", [None, 300])
async def test_run_cancellation_terminates_the_remote_process(
    command_timeout: float | None,
) -> None:
    connection = _Connection(process=_Process(block=True))
    client = _client(connection)
    run = asyncio.create_task(client.run("echo never", timeout=command_timeout))
    await connection.process.started.wait()

    run.cancel()

    with pytest.raises(asyncio.CancelledError):
        await run
    assert connection.process.terminated is True
    assert connection.process.closed is False
    assert connection.process.waited_closed is True


async def test_run_rejects_a_completed_process_without_a_returncode() -> None:
    completed = _Completed()
    completed.returncode = None
    connection = _Connection(process=_Process(completed=completed))
    client = _client(connection)

    with pytest.raises(SSHConnectionError, match="without an exit status"):
        await client.run("echo incomplete", timeout=1)

    assert connection.closed is True


async def test_windows_write_uses_one_timeout_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = SSHClient(
        Capability(
            name="shell",
            protocol="ssh/2",
            url="ssh://workspace.example:2222",
            params={"shell": "powershell"},
        ),
        cast("asyncssh.SSHClientConnection", _Connection()),
    )
    timeouts: list[float] = []

    async def run(*args: object, **kwargs: Any) -> _Completed:
        del args
        timeout = kwargs["timeout"]
        assert isinstance(timeout, float)
        timeouts.append(timeout)
        await asyncio.sleep(0.01)
        return _Completed()

    monkeypatch.setattr(client, "run", run)

    await client.write_text("C:\\file.txt", "x" * 7000, timeout_s=1)

    assert len(timeouts) == 3
    assert timeouts[0] > timeouts[1] > timeouts[2]


async def test_close_during_reconnect_discards_the_replacement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dropped = _Connection(closed=True)
    replacement = _Connection()
    reconnect_started = asyncio.Event()
    allow_reconnect = asyncio.Event()

    async def reconnect(capability: Capability) -> asyncssh.SSHClientConnection:
        assert capability is client.capability
        reconnect_started.set()
        await allow_reconnect.wait()
        return cast("asyncssh.SSHClientConnection", replacement)

    client = _client(dropped)
    monkeypatch.setattr(client, "_connect", reconnect)

    run = asyncio.create_task(client.run("echo never"))
    await reconnect_started.wait()
    close = asyncio.create_task(client.close())
    await asyncio.sleep(0)
    allow_reconnect.set()

    with pytest.raises(SSHConnectionError, match="client is closed"):
        await run
    await close

    assert replacement.closed is True
    assert replacement.commands == []


async def test_reconnect_exhaustion_raises_connection_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _client(_Connection(closed=True))
    reconnect = AsyncMock(side_effect=OSError("unreachable"))
    sleep = AsyncMock()
    monkeypatch.setattr(client, "_connect", reconnect)
    monkeypatch.setattr("hud.capabilities.ssh.asyncio.sleep", sleep)

    with pytest.raises(SSHConnectionError, match="failed after 3 attempts"):
        await client.run("echo never")

    assert reconnect.await_count == 3
    assert [call.args for call in sleep.await_args_list] == [(0.25,), (0.5,)]
