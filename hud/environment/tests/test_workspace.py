"""Workspace contract tests: credential placement and the shell_uid wall."""

from __future__ import annotations

import asyncio
import base64
import contextlib
import itertools
import json
import os
import shutil
import signal
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import time
import urllib.parse
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast
from unittest import mock
from unittest.mock import AsyncMock, Mock

import asyncssh
import pytest

from hud.capabilities import Connection, SSHClient
from hud.environment import namespace as namespace_mod
from hud.environment import process_guard as process_guard_mod
from hud.environment import workspace as workspace_mod
from hud.environment.egress import (
    ConnectionRelay,
    Peer,
    WorkspaceRoute,
    _field,
    _UnixServer,
    _Unrelayable,
)
from hud.environment.workspace import Bubblewrap, Mount, Workspace
from hud.utils.process import ProcessGroup, ProcessResult

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Mapping

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="POSIX workspace semantics")


async def _connect(ws: Workspace) -> asyncssh.SSHClientConnection:
    host, port = ws.ssh_url.removeprefix("ssh://").split(":")
    key_path = ws.ssh_client_key_path
    assert key_path is not None
    return await asyncssh.connect(
        host,
        int(port),
        username=ws.ssh_user,
        client_keys=[str(key_path)],
        known_hosts=None,
    )


@contextlib.asynccontextmanager
async def _connected_namespace_host(
    monkeypatch: pytest.MonkeyPatch,
) -> AsyncIterator[namespace_mod.NamespaceHost]:
    socket_dir = tempfile.TemporaryDirectory(prefix="hud-namespace-", dir="/tmp")
    socket_path = Path(socket_dir.name) / "host.sock"
    server_host = namespace_mod._NamespaceHost(
        socket_path,
        setup_loopback=False,
        holder_argv=[],
        bwrap="",
        launcher_depth=0,
        map_identities=False,
        ports=frozenset(),
    )
    holders = [SimpleNamespace(terminate=AsyncMock()) for _ in range(2)]
    monkeypatch.setattr(
        server_host,
        "_start_holder",
        AsyncMock(side_effect=[(holder, os.getpid()) for holder in holders]),
    )

    create_process_group_exec = namespace_mod.create_process_group_exec

    async def spawn_without_nsenter(*argv: str, **kwargs: Any) -> ProcessGroup:
        command = argv[argv.index("--") + 1 :]
        return await create_process_group_exec(*command, **kwargs)

    monkeypatch.setattr(namespace_mod, "create_process_group_exec", spawn_without_nsenter)
    listen = asyncssh.listen
    listening = asyncio.Event()

    async def listen_and_signal(*args: Any, **kwargs: Any) -> asyncssh.SSHAcceptor:
        server = await listen(*args, **kwargs)
        listening.set()
        return server

    monkeypatch.setattr(asyncssh, "listen", listen_and_signal)
    serve_task = asyncio.create_task(server_host.serve())
    client: namespace_mod.NamespaceHost | None = None
    try:
        await asyncio.wait_for(listening.wait(), 1.0)
        client = namespace_mod.NamespaceHost(socket_path)
        await client.connect()
        yield client
    finally:
        if client is not None:
            await client.close()
        serve_task.cancel()
        await asyncio.gather(serve_task, return_exceptions=True)
        socket_dir.cleanup()


def _wait_for_path(path: Path, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while not path.exists():
        if time.monotonic() >= deadline:
            raise TimeoutError(f"timed out waiting for {path}")
        time.sleep(0.01)


@pytest.mark.asyncio
async def test_start_waits_until_the_ssh_acceptor_is_ready(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
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
        async with await _connect(ws) as conn:
            assert (await conn.run("echo ready")).stdout == "ready\n"
    finally:
        await ws.stop()


@pytest.mark.asyncio
async def test_credentials_live_outside_the_served_root(tmp_path: Path) -> None:
    """The agent's shell root must not contain its SSH key material."""
    ws = Workspace(tmp_path / "root")
    await ws.start()
    try:
        key_path = ws.ssh_client_key_path
        assert key_path is not None
        assert not key_path.is_relative_to(ws.root)
        assert not (ws.root / ".hud").exists()
        # The daemon still works from the external credentials.
        async with await _connect(ws) as conn:
            result = await conn.run("echo ok")
            stdout = result.stdout
            assert isinstance(stdout, str) and "ok" in stdout
    finally:
        await ws.stop()
    assert not key_path.exists()


@pytest.mark.asyncio
async def test_sftp_subsystem_is_not_served(tmp_path: Path) -> None:
    ws = Workspace(tmp_path / "root")
    await ws.start()
    try:
        async with await _connect(ws) as conn:
            with pytest.raises(asyncssh.ChannelOpenError):
                await conn.start_sftp_client()
    finally:
        await ws.stop()


@pytest.mark.asyncio
async def test_file_operations_use_the_exec_channel(tmp_path: Path) -> None:
    ws = Workspace(tmp_path / "root")
    await ws.start()
    try:
        async with await _connect(ws) as conn:
            client = SSHClient(ws.capability(), conn)
            await client.write_text("hello world.txt", "héllo\n")
            assert await client.read_text("hello world.txt") == "héllo\n"
            await client.write_text("empty.txt", "")
            assert await client.read_text("empty.txt") == ""
            assert await client.listdir(".") == ["empty.txt", "hello world.txt"]
            # Absolute paths mean what they say in the session's namespace —
            # never re-anchored under the workspace.
            outside = tmp_path / "outside.txt"
            await client.write_text(str(outside), "done")
            assert outside.read_text() == "done"
            assert await client.read_text(str(outside)) == "done"
            assert not (tmp_path / "root" / str(outside).lstrip("/")).exists()
    finally:
        await ws.stop()


@pytest.mark.asyncio
async def test_output_arrives_while_the_command_is_still_running(tmp_path: Path) -> None:
    """Held until exit, a long build tells the agent nothing while it runs and
    a session that never exits says nothing at all."""
    ws = Workspace(tmp_path / "root")
    await ws.start()
    try:
        async with await _connect(ws) as conn:
            started = time.monotonic()
            process = await conn.create_process("echo first; sleep 5; echo second")
            first = await asyncio.wait_for(process.stdout.readline(), 10)
            elapsed = time.monotonic() - started
            process.channel.close()
    finally:
        await ws.stop()

    assert first.strip() == "first"
    # Held until exit it would take the full sleep; the point is that it does not.
    assert elapsed < 2.0, f"first line took {elapsed:.1f}s — output is not streaming"


@pytest.mark.parametrize(
    ("command", "marker"),
    [
        (
            "sh -c 'sleep .1; exec >/dev/null 2>&1; : > silent-done' & printf started",
            "silent-done",
        ),
        (
            "sh -c \"sleep .1; trap '' PIPE; printf late-out || :; "
            "printf late-err >&2 || :; exec >/dev/null 2>&1; "
            ': > late-done" & printf started',
            "late-done",
        ),
        (
            "cd . && nohup sh -c 'sleep .1' "
            "</dev/null >stdin.log 2>&1 && exec >/dev/null 2>&1 && "
            ": > stdin-done & printf started",
            "stdin-done",
        ),
        (
            "(cd . && exec sh -c 'sleep .1; : > redirected-done') "
            "</dev/null >redirected.log 2>&1 & printf started",
            "redirected-done",
        ),
    ],
    ids=("inherited-eof", "late-output", "stdin-detached", "fully-redirected"),
)
@pytest.mark.asyncio
async def test_background_completion_preserves_subsequent_namespace_commands(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    command: str,
    marker: str,
) -> None:
    shell = shutil.which("bash") or "sh"
    async with _connected_namespace_host(monkeypatch) as namespace:
        launched = await namespace.spawn(
            [shell, "-c", command],
            cwd=tmp_path,
            env=dict(os.environ),
            persistent=True,
        )
        launch_result = await asyncio.wait_for(launched.complete(), 5.0)

        assert launch_result.returncode == 0
        assert launch_result.stdout == b"started"

        await asyncio.to_thread(_wait_for_path, tmp_path / marker)
        await asyncio.sleep(0)

        probe = await namespace.spawn(
            [shell, "-c", "printf healthy"],
            cwd=tmp_path,
            env=dict(os.environ),
            persistent=True,
        )
        probe_result = await asyncio.wait_for(probe.complete(), 5.0)

    assert probe_result.returncode == 0
    assert probe_result.stdout == b"healthy"


@pytest.mark.asyncio
async def test_namespace_process_forwards_standard_streams(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shell = shutil.which("bash") or "sh"
    async with _connected_namespace_host(monkeypatch) as namespace:
        process = await namespace.spawn(
            [
                shell,
                "-c",
                "IFS= read -r line; printf 'out:%s' \"$line\"; printf err >&2",
            ],
            cwd=tmp_path,
            env=dict(os.environ),
            persistent=True,
        )
        process.stdin.write(b"input\n")
        process.stdin.write_eof()
        result = await asyncio.wait_for(process.complete(), 5.0)

    assert result.returncode == 0
    assert result.stdout == b"out:input"
    assert result.stderr == b"err"


@pytest.mark.asyncio
async def test_a_session_that_asks_for_a_terminal_gets_one(tmp_path: Path) -> None:
    """Programs branch on isatty: without a pty they take their batch path, so
    a terminal task is graded on behaviour a terminal would never produce."""
    ws = Workspace(tmp_path / "root")
    await ws.start()
    try:
        async with await _connect(ws) as conn:
            with_pty = await conn.run(
                "test -t 0 && test -t 1 && echo TTY || echo NOT_TTY; tput cols 2>/dev/null",
                term_type="xterm-256color",
                term_size=(120, 40),
                check=False,
            )
            without = await conn.run("test -t 1 && echo TTY || echo NOT_TTY", check=False)
    finally:
        await ws.stop()

    assert "TTY" in str(with_pty.stdout) and "NOT_TTY" not in str(with_pty.stdout)
    # The size the client asked for reaches the terminal, not a default.
    assert "120" in str(with_pty.stdout)
    assert "NOT_TTY" in str(without.stdout)


@pytest.mark.asyncio
async def test_a_resize_does_not_cost_the_session_its_keyboard(tmp_path: Path) -> None:
    """asyncssh delivers a resize as an exception on the stdin read, and it is
    not an asyncssh.Error — unhandled it escapes the relay and input stops."""
    ws = Workspace(tmp_path / "root")
    await ws.start()
    try:
        async with await _connect(ws) as conn:
            process = await conn.create_process(
                "cat", term_type="xterm-256color", term_size=(80, 24)
            )
            process.channel.change_terminal_size(132, 43)
            await asyncio.sleep(0.2)
            # Input still reaches the shell after the resize.
            process.stdin.write("still-listening\n")
            echoed = await asyncio.wait_for(process.stdout.readline(), 5)
            process.channel.close()
    finally:
        await ws.stop()

    assert "still-listening" in echoed


@pytest.mark.asyncio
async def test_session_setup_failure_is_reported_to_the_client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ws = Workspace(tmp_path / "root")
    monkeypatch.setattr(ws, "sandbox_pid", AsyncMock(side_effect=RuntimeError("map failed")))
    await ws.start()
    try:
        async with await _connect(ws) as conn:
            result = await conn.run("echo unreachable", check=False)
    finally:
        await ws.stop()

    assert result.exit_status == 1
    assert "workspace: cannot prepare shell: map failed" in str(result.stderr)


def _wall(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(Workspace, "_drops_privileges", lambda self: True)
    monkeypatch.setattr(Workspace, "_setpriv", lambda self: "/usr/bin/setpriv")


@pytest.mark.asyncio
async def test_dropped_session_env_excludes_server_secrets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A dropped shell must not inherit the server's environment (secrets)."""
    _wall(monkeypatch)
    monkeypatch.setenv("HUD_API_KEY", "super-secret")

    ws = Workspace(tmp_path / "root", shell_uid=1000, env={"CUSTOM": "1"})
    session_env = ws._session_env()
    assert session_env is not None
    assert "HUD_API_KEY" not in session_env
    assert session_env["CUSTOM"] == "1"
    assert "PATH" in session_env
    # The server's HOME (/root) is unreadable by the dropped uid.
    assert session_env["HOME"] == ws._guest_path


def _sandbox_env(argv: list[str]) -> dict[str, str]:
    """The environment the sandboxed payload starts from (its ``env -i`` set)."""
    assignments = argv[argv.index("-i") + 1 :]
    return dict(item.split("=", 1) for item in itertools.takewhile(lambda a: "=" in a, assignments))


def test_bwrap_drops_host_env_when_walled(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The bwrap path must not re-inject host secrets, while per-call env
    overrides still reach the sandbox."""
    monkeypatch.setenv("HUD_API_KEY", "super-secret")
    _wall(monkeypatch)

    ws = Workspace(tmp_path / "root", shell_uid=1000, env={"CUSTOM": "1"})
    monkeypatch.setattr(ws, "_bwrap", Bubblewrap("/usr/bin/bwrap"))
    argv = ws.shell_argv("echo hi", env={"PER_CALL": "1"})

    sandbox_env = _sandbox_env(argv)
    assert "HUD_API_KEY" not in sandbox_env
    assert sandbox_env["CUSTOM"] == "1"
    assert "PATH" in sandbox_env
    assert sandbox_env["PER_CALL"] == "1"


def test_bwrap_inherits_host_env_when_not_walled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SENTINEL", "visible")
    ws = Workspace(tmp_path / "root")
    monkeypatch.setattr(ws, "_bwrap", Bubblewrap("/usr/bin/bwrap"))
    argv = ws.bwrap_argv(["bash", "-lc", "true"])
    assert _sandbox_env(argv)["SENTINEL"] == "visible"


def test_the_harness_own_configuration_never_reaches_a_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """HUD's variables in the serving process are a credential leak where they
    hold a key, and a tell everywhere else — an agent that finds HUD_ anything
    knows what is running it."""
    monkeypatch.setenv("HUD_API_KEY", "super-secret")
    monkeypatch.setenv("HUD_SKIP_VERSION_CHECK", "1")
    monkeypatch.setenv("ORDINARY", "kept")
    # What the task itself declares is the task's, HUD-shaped name or not.
    ws = Workspace(tmp_path / "root", env={"HUD_TASK_DECLARED": "mine"})
    monkeypatch.setattr(ws, "_bwrap", Bubblewrap("/usr/bin/bwrap"))

    for argv in (
        ws.shell_argv("echo hi"),
        ws.bwrap_argv(["true"]),
        ws.session_argv("echo hi"),
    ):
        session_env = _sandbox_env(argv)
        assert "HUD_API_KEY" not in session_env
        assert "HUD_SKIP_VERSION_CHECK" not in session_env
        assert session_env["ORDINARY"] == "kept"
        assert session_env["HUD_TASK_DECLARED"] == "mine"


#: Options bubblewrap gained after 0.4, the newest release on distros still in
#: use (debian bullseye ships 0.4.1). One of these in a session's argv aborts
#: every command on such a host with "Unknown option", which grades as a
#: legitimate zero rather than a broken environment.
_POST_0_4_BWRAP_OPTIONS = frozenset(
    {
        "--clearenv",  # 0.5.0
        "--assert-userns-disabled",  # 0.5.0
        "--overlay",  # 0.8.0
        "--tmp-overlay",  # 0.8.0
        "--ro-overlay",  # 0.8.0
        "--overlay-src",  # 0.8.0
        "--size",  # 0.9.0
        "--chmod",  # 0.9.0
    }
)


def test_session_argv_runs_on_bubblewrap_0_4(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Sessions must not pass an option an old-but-usable bwrap will reject."""
    ws = Workspace(
        tmp_path / "root",
        shell_uid=1000,
        env={"CUSTOM": "1"},
        mounts=(Mount("tmpfs", dst="/tests"),),
    )
    monkeypatch.setattr(ws, "_bwrap", Bubblewrap("/usr/bin/bwrap"))
    _wall(monkeypatch)

    for argv in (
        ws.shell_argv("echo hi"),
        ws.shell_argv(),
        ws.bwrap_argv(["true"]),
        ws.session_argv("echo hi"),
    ):
        assert not _POST_0_4_BWRAP_OPTIONS.intersection(argv)


def test_hosted_sessions_do_not_create_new_namespaces(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ws = Workspace(tmp_path / "root", guest_path="/app", network=True)
    monkeypatch.setattr(ws, "_bwrap", Bubblewrap("/usr/bin/bwrap"))

    first = ws.session_argv("start-a-server &")
    second = ws.session_argv("curl localhost")

    assert first[0] == "/usr/bin/bwrap" and second[0] == "/usr/bin/bwrap"
    for argv in (first, second):
        assert "--unshare-user" not in argv
        assert "--unshare-pid" not in argv
        assert "--unshare-net" not in argv
        assert "--die-with-parent" not in argv
        dev = argv.index("--dev-bind")
        assert argv[dev : dev + 3] == ["--dev-bind", "/dev", "/dev"]
        assert argv.count("--cap-drop") == 3
        assert argv[argv.index("--chdir") + 1] == "/app"


@pytest.mark.asyncio
async def test_concurrent_sessions_share_one_sandbox(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An agent issues parallel tool calls; if each started its own sandbox,
    what one backgrounds would be invisible to the next."""
    ws = Workspace(tmp_path / "root")
    monkeypatch.setattr(ws, "_bwrap", Bubblewrap("/usr/bin/bwrap"))
    spawned = 0

    async def fake_spawn() -> int:
        nonlocal spawned
        spawned += 1
        await asyncio.sleep(0.01)  # the real spawn awaits bwrap's readiness
        ws._sandbox = cast("Any", SimpleNamespace(returncode=None))
        ws._sandbox_init = 4000 + spawned
        return ws._sandbox_init

    monkeypatch.setattr(ws, "_start_sandbox", fake_spawn)

    pids = await asyncio.gather(*(ws.sandbox_pid() for _ in range(4)))

    assert spawned == 1
    assert set(pids) == {4001}


def test_the_sandbox_reports_readiness_before_sessions_join_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """bwrap names the child pid before that child has built its mount
    namespace, so the pid alone is not proof the sandbox can run anything."""
    ws = Workspace(tmp_path / "root")
    monkeypatch.setattr(ws, "_bwrap", Bubblewrap("/usr/bin/bwrap"))
    argv = ws.bwrap_argv(["sh", "-c", "echo ready"], info_fd=7)

    assert argv[argv.index("--info-fd") + 1] == "7"
    # The signal comes from the payload, which bwrap runs only after setup.
    assert argv[-1] == "echo ready"


def test_network_ownership_matches_bubblewrap_isolation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for allowed, network, owns in (
        ({"example.com"}, True, True),  # a policy needs a network to apply to
        (set(), True, True),  # declared unreachable
        (None, False, True),  # no-network, however it was spelled
        (None, True, False),  # the substrate's network, as before
    ):
        ws = Workspace(tmp_path / "root", network=network, allowed_hosts=allowed)
        monkeypatch.setattr(ws, "_bwrap", Bubblewrap("/usr/bin/bwrap"))

        assert ws.owns_netns is owns
        assert ("--unshare-net" in ws.bwrap_argv(["true"])) is owns


@pytest.mark.asyncio
async def test_run_uses_fresh_mounts_and_shared_network_with_visitor_egress(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PATH", "/task/bin:/usr/bin")
    monkeypatch.setenv("HUD_API_KEY", "sk-secret")
    ws = Workspace(tmp_path / "root", env={"AGENT_ONLY": "secret"})
    monkeypatch.setattr(ws, "_bwrap", Bubblewrap("/usr/bin/bwrap"))

    @contextlib.asynccontextmanager
    async def visiting(allowed):
        assert allowed == {"pypi.org"}
        yield {"HTTPS_PROXY": "http://visitor"}

    complete = AsyncMock(return_value=ProcessResult(0, b"passed", b""))
    spawn = AsyncMock(return_value=SimpleNamespace(complete=complete))

    monkeypatch.setattr(ws, "sandbox_pid", AsyncMock(return_value=7))
    monkeypatch.setattr(ws, "visiting", visiting)
    ws._namespace = cast("Any", SimpleNamespace(spawn=spawn))

    result = await ws.run(
        ["test.sh"],
        env={"VERIFIER_ONLY": "yes"},
        identity=None,
        inherit_workspace_env=False,
        allowed_hosts={"pypi.org"},
        max_wait=12,
    )

    assert result.stdout == b"passed"
    complete.assert_awaited_once_with(max_wait=12)
    assert spawn.await_args is not None
    (argv,), kwargs = spawn.await_args
    assert argv[0] == "/usr/bin/bwrap"
    assert "--unshare-user" not in argv
    assert "--unshare-pid" not in argv
    assert "HTTPS_PROXY=http://visitor" in argv
    assert "PATH=/task/bin:/usr/bin" in argv
    assert "VERIFIER_ONLY=yes" in argv
    assert kwargs["mount_view"] == "host"
    assert "AGENT_ONLY=secret" not in argv
    assert not any(arg.startswith("HUD_API_KEY=") for arg in argv)
    assert kwargs["env"]["HTTPS_PROXY"] == "http://visitor"


@pytest.mark.asyncio
async def test_run_uses_a_disposable_writable_hosts_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "hosts"
    source.write_text("127.0.0.1 localhost\n", encoding="utf-8")
    ws = Workspace(tmp_path / "root")
    ws._hosts_path = source
    monkeypatch.setattr(ws, "_bwrap", Bubblewrap("/usr/bin/bwrap"))
    monkeypatch.setattr(ws, "sandbox_pid", AsyncMock(return_value=7))

    @contextlib.asynccontextmanager
    async def visiting(allowed):
        assert allowed == ()
        yield {}

    mounted: Path | None = None

    async def spawn(argv: list[str], **kwargs: Any) -> Any:
        nonlocal mounted
        del kwargs
        hosts_index = argv.index("/etc/hosts")
        assert argv[hosts_index - 2] == "--bind"
        mounted = Path(argv[hosts_index - 1])
        assert mounted.read_text("utf-8") == source.read_text("utf-8")
        assert mounted.parent.stat().st_mode & 0o777 == 0o711
        mounted.write_text("127.0.0.1 verifier-added\n", encoding="utf-8")
        return SimpleNamespace(complete=AsyncMock(return_value=ProcessResult(0, b"passed", b"")))

    monkeypatch.setattr(ws, "visiting", visiting)
    ws._namespace = cast("Any", SimpleNamespace(spawn=spawn))

    result = await ws.run(["test.sh"], identity=None, writable_hosts=True)

    assert result.stdout == b"passed"
    assert mounted is not None and not await asyncio.to_thread(mounted.exists)
    assert not mounted.parent.exists()
    assert await asyncio.to_thread(source.read_text, "utf-8") == "127.0.0.1 localhost\n"


def test_peer_forwarders_use_the_substrate_listen_backlog() -> None:
    assert _UnixServer.request_queue_size == socket.SOMAXCONN


@pytest.mark.asyncio
async def test_visiting_none_uses_the_workspace_egress(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ws = Workspace(tmp_path / "root")
    monkeypatch.setattr(ws, "sandbox_pid", AsyncMock(return_value=7))
    ws._egress = cast(
        "Any",
        SimpleNamespace(environment=Mock(return_value={"HTTPS_PROXY": "http://workspace"})),
    )

    async with ws.visiting(None) as environment:
        assert environment == {"HTTPS_PROXY": "http://workspace"}

    ws.allowed_hosts = frozenset({"pypi.org"})
    async with ws.visiting({"pypi.org"}) as environment:
        assert environment == {"HTTPS_PROXY": "http://workspace"}


@pytest.mark.asyncio
async def test_visiting_uses_the_reserved_workspace_bridge(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    credentials = Path(tempfile.mkdtemp(prefix="hud-visitor-", dir="/tmp"))
    try:
        ws = Workspace(
            tmp_path / "root",
            peers=[Peer("db", 5432)],
            ports=[5432],
            credentials_dir=credentials,
        )
        monkeypatch.setattr(ws, "sandbox_pid", AsyncMock(return_value=7))

        async with ws.visiting({"pypi.org"}) as environment:
            assert environment["HTTPS_PROXY"].endswith("127.0.0.1:3129")
            assert "db" in environment["NO_PROXY"]
            assert "127.0.0.2" in environment["NO_PROXY"]
            assert (credentials / "visitor" / "egress.sock").is_socket()

        assert not (credentials / "visitor" / "egress.sock").exists()
    finally:
        shutil.rmtree(credentials)


@pytest.mark.asyncio
async def test_overlapping_runs_keep_their_visitor_policies_and_credentials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from hud.environment import egress as egress_mod

    credentials = Path(tempfile.mkdtemp(prefix="hud-visitors-", dir="/tmp"))
    ws = Workspace(tmp_path / "root", credentials_dir=credentials)
    monkeypatch.setattr(ws, "_bwrap", Bubblewrap("/usr/bin/bwrap"))
    monkeypatch.setattr(ws, "sandbox_pid", AsyncMock(return_value=7))

    def unavailable(*_args: object, **_kwargs: object) -> socket.socket:
        raise ConnectionRefusedError

    monkeypatch.setattr(egress_mod, "_connect_public", unavailable)
    started = {host: asyncio.Event() for host in ("first.example", "second.example")}
    release = {host: asyncio.Event() for host in started}
    environments: dict[str, Mapping[str, str]] = {}

    class VisitorProcess:
        def __init__(self, host: str, environment: Mapping[str, str]) -> None:
            self.host = host
            self.environment = environment

        async def complete(self, *, max_wait: float | None = None) -> ProcessResult:
            environments[self.host] = self.environment
            started[self.host].set()
            await release[self.host].wait()
            return ProcessResult(0, b"", b"")

    async def launch(
        command: list[str],
        *,
        env: Mapping[str, str] | None = None,
        **_kwargs: object,
    ) -> VisitorProcess:
        assert env is not None
        return VisitorProcess(command[0], env)

    def proxy_response(
        environment: Mapping[str, str],
        host: str,
        *,
        credentials_from: Mapping[str, str] | None = None,
    ) -> bytes:
        proxy = urllib.parse.urlsplit((credentials_from or environment)["http_proxy"])
        assert proxy.username == "hud" and proxy.password is not None
        token = urllib.parse.unquote(proxy.password)
        authorization = base64.b64encode(f"hud:{token}".encode()).decode()
        client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        client.settimeout(5)
        client.connect(str(credentials / "visitor" / "egress.sock"))
        try:
            client.sendall(
                (
                    f"GET http://{host}/ HTTP/1.1\r\n"
                    f"Host: {host}\r\n"
                    f"Proxy-Authorization: Basic {authorization}\r\n\r\n"
                ).encode()
            )
            return client.recv(4096)
        finally:
            client.close()

    monkeypatch.setattr(ws, "launch", launch)
    tasks = [
        asyncio.create_task(
            ws.run(["first.example"], allowed_hosts={"first.example"}, identity=None)
        )
    ]
    try:
        await asyncio.wait_for(started["first.example"].wait(), 1)
        tasks.append(
            asyncio.create_task(
                ws.run(["second.example"], allowed_hosts={"second.example"}, identity=None)
            )
        )
        await asyncio.sleep(0)
        assert not started["second.example"].is_set()

        first = environments["first.example"]
        assert b"502" in proxy_response(first, "first.example")
        assert b"403" in proxy_response(first, "second.example")
        release["first.example"].set()
        assert (await tasks[0]).returncode == 0

        await asyncio.wait_for(started["second.example"].wait(), 1)
        second = environments["second.example"]
        assert first["http_proxy"] != second["http_proxy"]
        assert b"502" in proxy_response(second, "second.example")
        assert b"403" in proxy_response(second, "first.example")
        assert b"407" in proxy_response(second, "second.example", credentials_from=first)
    finally:
        for event in release.values():
            event.set()
        await asyncio.gather(*tasks, return_exceptions=True)
        shutil.rmtree(credentials, ignore_errors=True)


@pytest.mark.asyncio
async def test_staged_verifier_is_launched_by_the_namespace_host(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ws = Workspace(tmp_path / "root")
    monkeypatch.setattr(
        ws,
        "_bwrap",
        Bubblewrap("/usr/bin/bwrap", pid_unshare="/usr/bin/unshare"),
    )
    monkeypatch.setattr(ws, "sandbox_pid", AsyncMock(return_value=121))
    complete = AsyncMock(return_value=ProcessResult(0, b"passed", b""))
    spawn = AsyncMock(return_value=SimpleNamespace(complete=complete))
    ws._namespace = cast("Any", SimpleNamespace(spawn=spawn))

    await ws.run(["test.sh"], identity=None)

    assert spawn.await_args is not None
    (argv,), kwargs = spawn.await_args
    assert argv[0] == "/usr/bin/bwrap"
    assert "CAP_SYS_ADMIN" in argv
    assert "/usr/bin/unshare" not in argv
    assert kwargs["mount_view"] == "host"


@pytest.mark.asyncio
async def test_launch_keeps_an_environment_entrypoint_outside_agent_sessions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ws = Workspace(tmp_path / "root")
    _wall(monkeypatch)
    monkeypatch.setattr(workspace_mod, "_is_root", lambda: True)
    monkeypatch.setattr(workspace_mod.sys, "platform", "linux")
    monkeypatch.setattr(ws, "_bwrap", Bubblewrap("/usr/bin/bwrap"))
    monkeypatch.setattr(ws, "sandbox_pid", AsyncMock(return_value=121))
    launched = SimpleNamespace(returncode=None)
    spawn = AsyncMock(return_value=launched)
    ws._namespace = cast("Any", SimpleNamespace(spawn=spawn))

    process = await ws.launch(
        ["start-environment", "sleep", "infinity"],
        cwd="/app",
        identity=(1000, 2000),
        no_new_privs=False,
        persistent=True,
        scope="environment",
    )

    assert process is launched
    assert spawn.await_args is not None
    (argv,), kwargs = spawn.await_args
    assert argv[0] == "/usr/bin/bwrap"
    assert argv[argv.index("--uid") + 1] == "1000"
    assert argv[argv.index("--gid") + 1] == "2000"
    assert "--unshare-user-try" in argv
    assert "--dev-bind" in argv
    assert "--reuid" not in argv
    assert argv[argv.index("--chdir") + 1] == "/app"
    assert kwargs["mount_view"] == "host"
    assert kwargs["identity"] == (1000, 2000)
    assert kwargs["persistent"] is True
    assert kwargs["scope"] == "environment"


@pytest.mark.asyncio
async def test_terminate_sessions_preserves_the_namespace_host(tmp_path: Path) -> None:
    ws = Workspace(tmp_path / "root")
    terminate_sessions = AsyncMock()
    ws._namespace = cast("Any", SimpleNamespace(terminate_sessions=terminate_sessions))

    await ws.terminate_sessions()

    terminate_sessions.assert_awaited_once_with()
    assert ws._namespace is not None


@pytest.mark.asyncio
async def test_namespace_management_does_not_share_process_connections(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    host = namespace_mod.NamespaceHost(tmp_path / "namespace.sock")
    process_connection = SimpleNamespace(run=AsyncMock())
    host._connection = cast("Any", process_connection)
    management = SimpleNamespace(
        run=AsyncMock(return_value=SimpleNamespace(returncode=0)),
        close=Mock(),
        wait_closed=AsyncMock(),
    )
    open_connection = AsyncMock(return_value=management)
    monkeypatch.setattr(host, "_open_connection", open_connection)

    await host.terminate_sessions()

    open_connection.assert_awaited_once_with()
    process_connection.run.assert_not_awaited()
    management.close.assert_called_once_with()
    management.wait_closed.assert_awaited_once_with()


def test_process_guard_executes_projected_file_without_module_reentry() -> None:
    guard = cast(
        "Any",
        SimpleNamespace(
            backend="ptrace",
            sandbox_helper="/tmp/.hud-process-connection/process_guard.py",
            sandbox_socket="/tmp/.hud-process-connection/control.sock",
        ),
    )
    argv = workspace_mod._guarded_process_argv(
        "/usr/bin/python3",
        guard,
        ["bash", "-lc", "true"],
    )

    assert argv == [
        "/usr/bin/python3",
        "/tmp/.hud-process-connection/process_guard.py",
        "--backend",
        "ptrace",
        "/tmp/.hud-process-connection/control.sock",
        "--",
        "bash",
        "-lc",
        "true",
    ]
    result = subprocess.run(
        [sys.executable, str(Path(workspace_mod.__file__).with_name("process_guard.py")), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert result.stderr == ""


def test_process_guard_selects_probed_ptrace_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(process_guard_mod, "_backend", None)
    monkeypatch.setattr(process_guard_mod, "_backend_probed", False)
    monkeypatch.setattr(process_guard_mod.sys, "platform", "linux")
    run = Mock(return_value=SimpleNamespace(returncode=0, stdout="ptrace\n"))
    monkeypatch.setattr(process_guard_mod.subprocess, "run", run)

    assert process_guard_mod._detected_backend() == "ptrace"
    assert process_guard_mod._detected_backend() == "ptrace"
    run.assert_called_once()


@pytest.mark.asyncio
async def test_namespace_host_only_terminates_a_used_session_holder(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    host = namespace_mod._NamespaceHost(
        tmp_path / "namespace.sock",
        setup_loopback=False,
        holder_argv=[],
        bwrap="bwrap",
        launcher_depth=0,
        map_identities=False,
        ports=frozenset(),
    )
    holder = AsyncMock()
    host.holders["session"] = (holder, 7)
    kill = Mock()
    monkeypatch.setattr(os, "kill", kill)

    await host._terminate_sessions()
    holder.terminate.assert_not_awaited()
    kill.assert_not_called()

    host.session_used = True
    await host._terminate_sessions()
    kill.assert_called_once_with(7, signal.SIGKILL)
    holder.terminate.assert_awaited_once_with()
    holder.wait.assert_not_awaited()


@pytest.mark.asyncio
async def test_run_can_use_a_fresh_no_network_sandbox(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ws = Workspace(tmp_path / "root")
    monkeypatch.setattr(
        ws,
        "_bwrap",
        Bubblewrap("/usr/bin/bwrap", pid_unshare="/usr/bin/unshare"),
    )
    install_identity_map = AsyncMock(return_value=9)
    complete = AsyncMock(return_value=ProcessResult(0, b"isolated", b""))
    spawn = AsyncMock(return_value=SimpleNamespace(complete=complete))

    monkeypatch.setattr(workspace_mod, "install_identity_map", install_identity_map)
    monkeypatch.setattr(workspace_mod, "create_process_group_exec", spawn)

    result = await ws.run(
        ["test.sh"],
        isolated=True,
        identity=None,
        max_wait=5,
    )

    assert result.stdout == b"isolated"
    complete.assert_awaited_once_with(max_wait=5)
    install_identity_map.assert_awaited_once()
    assert spawn.await_args is not None
    argv, kwargs = spawn.await_args
    assert argv[0] == "/usr/bin/bwrap"
    assert "/usr/bin/unshare" not in argv
    assert "--unshare-net" in argv
    assert len(kwargs["pass_fds"]) == 2


@pytest.mark.asyncio
async def test_an_isolated_command_keeps_the_image_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A command must not lose the image's PATH — the interpreters and tools
    the task installed — merely because it asked for an isolated sandbox.
    Both branches of run() give it the serving environment, less HUD's own."""
    monkeypatch.setenv("PATH", "/task/bin:/usr/bin")
    monkeypatch.setenv("HUD_API_KEY", "sk-secret")
    ws = Workspace(tmp_path / "root")
    monkeypatch.setattr(ws, "_bwrap", Bubblewrap("/usr/bin/bwrap"))
    monkeypatch.setattr(workspace_mod, "install_identity_map", AsyncMock(return_value=9))
    complete = AsyncMock(return_value=ProcessResult(0, b"", b""))
    spawn = AsyncMock(return_value=SimpleNamespace(complete=complete))
    monkeypatch.setattr(workspace_mod, "create_process_group_exec", spawn)

    await ws.run(["test.sh"], isolated=True, identity=None)

    assert spawn.await_args is not None
    argv, _ = spawn.await_args
    assert "PATH=/task/bin:/usr/bin" in argv
    assert not any(arg.startswith("HUD_API_KEY=") for arg in argv)


@pytest.mark.asyncio
async def test_failed_isolated_run_terminates_its_sandbox(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ws = Workspace(tmp_path / "root")
    monkeypatch.setattr(ws, "_bwrap", Bubblewrap("/usr/bin/bwrap"))
    process = SimpleNamespace(terminate=AsyncMock(), complete=AsyncMock())
    monkeypatch.setattr(
        workspace_mod,
        "create_process_group_exec",
        AsyncMock(return_value=process),
    )
    monkeypatch.setattr(
        workspace_mod,
        "install_identity_map",
        AsyncMock(side_effect=RuntimeError("map failed")),
    )

    with pytest.raises(RuntimeError, match="map failed"):
        await ws.run(["test.sh"], isolated=True, identity=None)

    process.terminate.assert_awaited_once()
    process.complete.assert_not_awaited()


@pytest.mark.asyncio
async def test_failed_sandbox_start_discards_the_holder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ws = Workspace(tmp_path / "root", network=True)
    monkeypatch.setattr(ws, "_bwrap", Bubblewrap("/usr/bin/bwrap"))
    failed_holder = SimpleNamespace(
        returncode=None,
        stdout=SimpleNamespace(readline=AsyncMock(return_value=b"ready\n")),
        stderr=SimpleNamespace(read=AsyncMock(return_value=b"")),
        kill=Mock(),
        wait=AsyncMock(return_value=0),
    )
    monkeypatch.setattr(
        asyncio,
        "create_subprocess_exec",
        AsyncMock(return_value=failed_holder),
    )
    monkeypatch.setattr(
        workspace_mod,
        "install_identity_map",
        AsyncMock(side_effect=RuntimeError("map failed")),
    )

    with pytest.raises(RuntimeError, match="map failed"):
        await ws.sandbox_pid()

    failed_holder.kill.assert_called_once()


@pytest.mark.asyncio
async def test_staged_sandbox_hosts_the_namespace_server_outside_bwrap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ws = Workspace(tmp_path / "root", network=False)
    monkeypatch.setattr(
        ws,
        "_bwrap",
        Bubblewrap("/usr/bin/bwrap", pid_unshare="/usr/bin/unshare"),
    )
    stdin = SimpleNamespace(write=Mock(), drain=AsyncMock(), close=Mock())
    sandbox = SimpleNamespace(
        returncode=None,
        stdin=stdin,
        stdout=SimpleNamespace(readline=AsyncMock(return_value=b'{"holder_pid": 121}\n')),
        stderr=SimpleNamespace(read=AsyncMock(return_value=b"")),
    )
    spawn = AsyncMock(return_value=sandbox)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
    bridge = SimpleNamespace(
        stdin=SimpleNamespace(write=Mock(), drain=AsyncMock()),
        stdout=SimpleNamespace(readline=AsyncMock(return_value=b"ready\n")),
        stderr=SimpleNamespace(read=AsyncMock(return_value=b"")),
    )
    namespace = SimpleNamespace(
        connect=AsyncMock(),
        forward=AsyncMock(),
        spawn=AsyncMock(return_value=bridge),
    )
    monkeypatch.setattr(workspace_mod, "NamespaceHost", Mock(return_value=namespace))

    assert await ws.sandbox_pid() == 121

    assert spawn.await_args is not None
    argv, _ = spawn.await_args
    assert argv[:2] == ("/usr/bin/unshare", "--net")
    assert "/usr/bin/bwrap" not in argv
    config = json.loads(stdin.write.call_args.args[0])
    assert config["holder_argv"][:5] == [
        "/usr/bin/unshare",
        "--kill-child=KILL",
        "--pid",
        "--mount-proc",
        "/usr/bin/bwrap",
    ]
    holder_dev = config["holder_argv"].index("--dev-bind")
    assert config["holder_argv"][holder_dev : holder_dev + 3] == [
        "--dev-bind",
        "/dev",
        "/dev",
    ]
    assert config["map_identities"] is True
    assert config["launcher_depth"] == 2


def test_a_peer_answers_at_the_address_the_task_expects() -> None:
    """A task that names a service says where it expects to find it. Placed
    anywhere else, the task's own client configuration points at nothing."""
    from hud.environment.egress import Peer, bind_addresses

    # One peer per port is the ordinary case: it is at localhost, which is
    # what a task saying "localhost:5432" or "http://localhost:8080" means.
    single = bind_addresses([Peer("db", 5432), Peer("api", 8080)])
    assert single == {"db": "127.0.0.1", "api": "127.0.0.1"}

    # Two services cannot both hold one port there, so the second moves — and
    # is still reached by its name, which is how a task addresses two anyway.
    both = bind_addresses([Peer("primary", 5432), Peer("replica", 5432)])
    assert both == {"primary": "127.0.0.1", "replica": "127.0.0.2"}

    reserved = bind_addresses([Peer("api", 8080)], reserved_ports={8080})
    assert reserved == {"api": "127.0.0.2"}

    proxies = bind_addresses([Peer("agent", 3128), Peer("verifier", 3129)])
    assert proxies == {"agent": "127.0.0.2", "verifier": "127.0.0.2"}

    service = bind_addresses([Peer("db", 5432), Peer("db", 6379)])
    assert service == {"db": "127.0.0.1"}

    service_with_reserved_port = bind_addresses(
        [Peer("db", 5432), Peer("db", 6379)],
        reserved_ports={5432},
    )
    assert service_with_reserved_port == {"db": "127.0.0.2"}

    with pytest.raises(ValueError, match="declares port 5432 twice"):
        bind_addresses([Peer("db", 5432), Peer("db", 5432)])


async def test_workspace_route_is_bound_once_and_removed_on_stop(tmp_path: Path) -> None:
    from hud.environment import Environment

    env = Environment()
    workspace = env.workspace(tmp_path / "root", track_files=False)
    workspace._bwrap = cast("Any", object())
    env._started = True
    route = WorkspaceRoute("ssh", "inference.hud.so", 443)

    env.bind_workspace_routes([route, route])
    env.bind_workspace_routes([route])

    assert workspace.peers == (Peer("inference.hud.so", 443, target=("inference.hud.so", 443)),)
    await env.stop()
    assert workspace.peers == ()


def test_workspace_route_rejects_ip_literals() -> None:
    with pytest.raises(ValueError, match="hostname"):
        WorkspaceRoute("ssh", "127.0.0.1", 443)


def test_workspace_names_are_added_to_the_substrates_hosts_rather_than_replacing_it() -> None:
    """Dropping the substrate's entries would cost the workspace localhost."""
    from hud.environment.egress import Peer, hosts_text

    text = hosts_text(
        [Peer("db", 5432)],
        "127.0.0.1\tlocalhost\n::1\tip6-localhost\n",
        local_aliases=["main"],
    )

    assert "127.0.0.1\tlocalhost" in text
    assert "::1\tip6-localhost" in text
    assert "127.0.0.1\tmain" in text
    assert text.endswith("127.0.0.1\tdb\n")

    same_service = hosts_text(
        [Peer("db", 5432), Peer("db", 6379)],
        "",
    )
    assert same_service == "127.0.0.1\tdb\n"


@pytest.mark.asyncio
async def test_a_declared_peer_is_a_name_sessions_resolve(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A peer reached only by port is not at the address the task expects, so
    the workspace carries its own hosts file — and only when it has a network
    of its own, since otherwise the service is already at its real address."""
    from hud.environment.egress import Peer

    ws = Workspace(
        tmp_path / "root",
        peers=[Peer("db", 5432)],
        local_aliases=["main"],
        ports=[5432],
        allowed_hosts={"pypi.org"},
        credentials_dir=tmp_path / "protected" / "session-keys",
        hosts_path=tmp_path / "runtime" / "hosts",
    )
    monkeypatch.setattr(ws, "_bwrap", Bubblewrap("/usr/bin/bwrap"))
    ws._prepare_runtime()

    argv = ws.bwrap_argv(["true"])
    hosts = Path(argv[argv.index("/etc/hosts") - 1])
    assert hosts == tmp_path / "runtime" / "hosts"
    assert argv[argv.index("/etc/hosts") - 2] == "--ro-bind"
    hosts_content = await asyncio.to_thread(hosts.read_text)
    assert "127.0.0.1\tmain\n" in hosts_content
    assert "127.0.0.2\tdb\n" in hosts_content
    # Bound over /etc/hosts, so every session reads it whatever its identity.
    assert (await asyncio.to_thread(hosts.stat)).st_mode & 0o044

    sharing = Workspace(tmp_path / "shared", peers=[Peer("db", 5432)], network=True)
    monkeypatch.setattr(sharing, "_bwrap", Bubblewrap("/usr/bin/bwrap"))
    sharing._prepare_runtime()
    assert "/etc/hosts" not in sharing.bwrap_argv(["true"])

    await ws.stop()
    await sharing.stop()


def test_private_workspace_preserves_localhost_without_peers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    hosts = tmp_path / "hosts"
    ws = Workspace(
        tmp_path / "root",
        allowed_hosts=set(),
        hosts_path=hosts,
    )
    monkeypatch.setattr(ws, "_bwrap", Bubblewrap("/usr/bin/bwrap"))

    ws._prepare_runtime()

    assert "/etc/hosts" in ws.bwrap_argv(["true"])
    assert "localhost" in hosts.read_text("utf-8")


def test_a_peer_is_reached_directly_rather_than_through_the_proxy() -> None:
    """The proxy resolves names out on the substrate, where a peer's name
    means nothing and its address is something else entirely."""
    from hud.environment.egress import Egress, Peer

    egress = Egress(
        "/tmp/unused",
        {"pypi.org"},
        [Peer("db", 5432), Peer("replica", 5432)],
        local_aliases=["main"],
    )
    bypass = egress.environment()["no_proxy"].split(",")

    assert "main" in bypass
    assert "db" in bypass and "replica" in bypass
    assert "127.0.0.1" in bypass and "127.0.0.2" in bypass


def test_bridge_socket_paths_are_configuration_not_process_arguments() -> None:
    from hud.environment.egress import Egress, Peer

    egress = Egress(
        "/media/hud/session-keys",
        {"pypi.org"},
        [Peer("db", 5432)],
        reserved_ports={5432},
    )

    visitor = Path("/media/hud/session-keys/visitor/egress.sock")
    argv, config = egress.bridge_command(visitor_socket=visitor)

    assert "/media/hud/session-keys" not in " ".join(argv)
    assert "/media/hud/session-keys" in config.decode()
    routes = json.loads(config)
    assert ["127.0.0.1", 3129, str(visitor)] in routes
    assert ["127.0.0.2", 5432, "/media/hud/session-keys/peer-0.sock"] in routes


def test_bridge_ignores_workspace_imports_and_closes_reset_connections(tmp_path: Path) -> None:
    from hud.environment.egress import Egress

    workspace = tmp_path / "workspace"
    package = workspace / "hud" / "environment"
    package.mkdir(parents=True)
    (workspace / "hud" / "__init__.py").write_text("")
    (package / "__init__.py").write_text("")
    marker = workspace / "imported"
    (package / "utils.py").write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).write_text('imported')\n"
    )

    sockets = Path(tempfile.mkdtemp(dir="/tmp"))
    egress = Egress(sockets, {"example.com"})
    upstream_listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    upstream_listener.bind(str(egress.socket_path))
    upstream_listener.listen()
    upstream_listener.settimeout(5)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as port_holder:
        port_holder.bind(("127.0.0.1", 0))
        port = port_holder.getsockname()[1]
    argv, config = egress.bridge_command(port=port)
    process = subprocess.Popen(
        argv,
        cwd=workspace,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    client: socket.socket | None = None
    upstream: socket.socket | None = None
    try:
        assert process.stdin is not None and process.stdout is not None
        process.stdin.write(config)
        process.stdin.flush()
        assert process.stdout.readline() == b"ready\n"
        assert not marker.exists()

        client = socket.create_connection(("127.0.0.1", port), timeout=5)
        upstream, _ = upstream_listener.accept()
        upstream.settimeout(1)
        client.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
        client.close()
        client = None

        assert upstream.recv(1) == b""
    finally:
        if client is not None:
            client.close()
        if upstream is not None:
            upstream.close()
        upstream_listener.close()
        if process.poll() is None:
            process.terminate()
        process.wait(timeout=5)
        shutil.rmtree(sockets, ignore_errors=True)


def test_bridge_drains_reverse_direction_after_half_close(tmp_path: Path) -> None:
    from hud.environment.egress import Egress

    sockets = Path(tempfile.mkdtemp(dir="/tmp"))
    egress = Egress(sockets, {"example.com"})
    upstream_listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    upstream_listener.bind(str(egress.socket_path))
    upstream_listener.listen()
    upstream_listener.settimeout(5)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as port_holder:
        port_holder.bind(("127.0.0.1", 0))
        port = port_holder.getsockname()[1]
    argv, config = egress.bridge_command(port=port)
    process = subprocess.Popen(
        argv,
        cwd=tmp_path,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    client: socket.socket | None = None
    upstream: socket.socket | None = None
    try:
        assert process.stdin is not None and process.stdout is not None
        process.stdin.write(config)
        process.stdin.flush()
        assert process.stdout.readline() == b"ready\n"

        client = socket.create_connection(("127.0.0.1", port), timeout=5)
        client.settimeout(5)
        upstream, _ = upstream_listener.accept()
        upstream.settimeout(5)

        request = b"complete request"
        client.sendall(request)
        client.shutdown(socket.SHUT_WR)
        assert upstream.recv(65536) == request
        assert upstream.recv(1) == b""

        response = b"complete response" * 4096
        upstream.sendall(response)
        upstream.shutdown(socket.SHUT_WR)
        received = bytearray()
        while chunk := client.recv(65536):
            received.extend(chunk)
        assert received == response
    finally:
        if client is not None:
            client.close()
        if upstream is not None:
            upstream.close()
        upstream_listener.close()
        if process.poll() is None:
            process.terminate()
        process.wait(timeout=5)
        shutil.rmtree(sockets, ignore_errors=True)


def _half_close_backend(
    response: bytes,
) -> tuple[socket.socket, threading.Thread, int, list[bytes]]:
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    listener.listen()
    listener.settimeout(5)
    port = listener.getsockname()[1]
    requests: list[bytes] = []

    def serve() -> None:
        with listener:
            connection, _ = listener.accept()
            with connection:
                chunks: list[bytes] = []
                while chunk := connection.recv(65536):
                    chunks.append(chunk)
                requests.append(b"".join(chunks))
                connection.sendall(response)

    thread = threading.Thread(target=serve)
    thread.start()
    return listener, thread, port, requests


def test_relay_drains_reverse_direction_when_write_shutdown_fails() -> None:
    from hud.environment.egress import _relay

    class ShutdownErrorSocket(socket.socket):
        def shutdown(self, how: int) -> None:
            raise OSError("injected shutdown failure")

    one, one_peer = socket.socketpair()
    other_socket, other_peer = socket.socketpair()
    other = ShutdownErrorSocket(fileno=other_socket.detach())
    one_peer.settimeout(5)
    relay_thread = threading.Thread(target=_relay, args=(one, other, 1.0))
    relay_thread.start()
    try:
        one_peer.shutdown(socket.SHUT_WR)
        response = b"complete response" * 64
        other_peer.sendall(response)
        other_peer.shutdown(socket.SHUT_WR)

        received = bytearray()
        while len(received) < len(response):
            received.extend(one_peer.recv(65536))
        assert received == response
    finally:
        one.close()
        one_peer.close()
        other.close()
        other_peer.close()
        relay_thread.join(5)
    assert not relay_thread.is_alive()


def test_connect_tunnel_drains_response_after_client_half_close(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hud.environment import egress as egress_mod
    from hud.environment.egress import Egress

    response = b"complete response" * 32768
    listener, thread, port, requests = _half_close_backend(response)

    def connect(_host: str, _port: int, timeout: float) -> socket.socket:
        return socket.create_connection(("127.0.0.1", port), timeout)

    monkeypatch.setattr(egress_mod, "_connect_public", connect)
    sockets = Path(tempfile.mkdtemp(dir="/tmp"))
    egress = Egress(sockets, {"example.com"})
    egress.start()
    try:
        client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        client.settimeout(5)
        client.connect(str(egress.socket_path))
        client.sendall(b"CONNECT example.com:443 HTTP/1.1\r\nHost: example.com:443\r\n\r\n")
        established = b""
        while b"\r\n\r\n" not in established:
            established += client.recv(4096)
        assert b"200 Connection established" in established

        request = b"request body"
        client.sendall(request)
        client.shutdown(socket.SHUT_WR)
        received = b""
        while chunk := client.recv(65536):
            received += chunk
        client.close()
    finally:
        egress.stop()
        listener.close()
        shutil.rmtree(sockets, ignore_errors=True)

    thread.join(5)
    assert not thread.is_alive()
    assert requests == [request]
    assert received == response


def test_peer_forward_drains_response_after_client_half_close() -> None:
    from hud.environment.egress import Egress

    response = b"complete response" * 32768
    listener, thread, port, requests = _half_close_backend(response)
    sockets = Path(tempfile.mkdtemp(dir="/tmp"))
    egress = Egress(sockets, set(), [Peer("service", port)])
    egress.start()
    try:
        client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        client.settimeout(5)
        client.connect(str(sockets / "peer-0.sock"))
        request = b"request body"
        client.sendall(request)
        client.shutdown(socket.SHUT_WR)
        received = b""
        while chunk := client.recv(65536):
            received += chunk
        client.close()
    finally:
        egress.stop()
        listener.close()
        shutil.rmtree(sockets, ignore_errors=True)

    thread.join(5)
    assert not thread.is_alive()
    assert requests == [request]
    assert received == response


def test_the_proxy_normalizes_http_framing_in_both_directions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import socket as socket_mod
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    from hud.environment import egress as egress_mod
    from hud.environment.egress import Egress

    class Chunked(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        request_body = b""
        request_host: str | None = None
        request_length: str | None = None
        request_transfer: str | None = None

        def do_POST(self) -> None:
            type(self).request_host = self.headers.get("Host")
            type(self).request_length = self.headers.get("Content-Length")
            type(self).request_transfer = self.headers.get("Transfer-Encoding")
            type(self).request_body = self.rfile.read(int(type(self).request_length or 0))
            self.send_response(200)
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            self.wfile.write(b"5\r\nhello\r\n0\r\n\r\n")

        def log_message(self, format: str, *args: Any) -> None:
            pass

    upstream = HTTPServer(("127.0.0.1", 0), Chunked)
    threading.Thread(target=upstream.serve_forever, daemon=True).start()
    port = upstream.server_address[1]
    monkeypatch.setattr(
        egress_mod,
        "_connect_public",
        lambda host, port, timeout: socket_mod.create_connection((host, port), timeout),
    )
    # Not the pytest tmp dir: a unix socket path is capped near 104 bytes.
    sockets = Path(tempfile.mkdtemp(dir="/tmp"))
    egress = Egress(sockets, {"127.0.0.1"})
    egress.start()
    try:
        client = socket_mod.socket(socket_mod.AF_UNIX, socket_mod.SOCK_STREAM)
        client.settimeout(10)
        client.connect(str(egress.socket_path))
        client.sendall(
            (
                f"POST http://127.0.0.1:{port}/ HTTP/1.1\r\n"
                "Host: forbidden.example\r\n"
                "Transfer-Encoding: chunked\r\n\r\n"
                "5\r\nhello\r\n6\r\n world\r\n0\r\n\r\n"
            ).encode()
        )
        received = b""
        while b"hello" not in received:
            chunk = client.recv(4096)
            if not chunk:
                break
            received += chunk
        client.close()
    finally:
        egress.stop()
        upstream.shutdown()
        upstream.server_close()
        shutil.rmtree(sockets, ignore_errors=True)

    headers, _, body = received.partition(b"\r\n\r\n")
    assert b"200" in headers
    assert b"transfer-encoding" not in headers.lower()
    assert body == b"hello"
    assert Chunked.request_body == b"hello world"
    assert Chunked.request_host == f"127.0.0.1:{port}"
    assert Chunked.request_length == "11"
    assert Chunked.request_transfer is None


def test_connection_relay_replaces_credentials_and_preserves_streaming() -> None:
    import http.client
    from http.server import BaseHTTPRequestHandler, HTTPServer

    class Upstream(BaseHTTPRequestHandler):
        authorization: str | None = None
        api_key: str | None = None
        body = b""

        def do_POST(self) -> None:
            type(self).authorization = self.headers.get("Authorization")
            type(self).api_key = self.headers.get("X-Api-Key")
            type(self).body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
            self.send_response(200)
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            self.wfile.write(b"5\r\nfirst\r\n6\r\nsecond\r\n0\r\n\r\n")

        def log_message(self, format: str, *args: Any) -> None:
            pass

    upstream = HTTPServer(("127.0.0.1", 0), Upstream)
    upstream_thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    upstream_thread.start()
    token = "scoped-runtime-token"
    connection = Connection(
        name="inference",
        capability="ssh",
        url=f"http://127.0.0.1:{upstream.server_address[1]}/v1",
        headers={"Authorization": f"Bearer {token}", "X-Api-Key": token},
    )
    relay = ConnectionRelay(connection)
    relay.start()
    try:
        client = http.client.HTTPConnection("127.0.0.1", relay.port, timeout=5)
        client.request(
            "POST",
            "/v1/messages",
            body=b"request",
            headers={
                "Authorization": "Bearer model-visible",
                "X-Api-Key": "model-visible",
            },
        )
        response = client.getresponse()
        assert response.status == 200
        assert response.read() == b"firstsecond"
        client.close()
    finally:
        relay.stop()
        upstream.shutdown()
        upstream.server_close()
        upstream_thread.join()

    assert Upstream.authorization == f"Bearer {token}"
    assert Upstream.api_key == token
    assert Upstream.body == b"request"


@pytest.mark.parametrize(
    "payload",
    [
        b"CONNECT 127.0.0.1:8765 HTTP/1.1\r\nHost: 127.0.0.1:8765\r\n\r\n",
        b"GET http://internal.example:8765/ HTTP/1.1\r\nHost: internal.example\r\n\r\n",
    ],
)
def test_public_egress_cannot_reach_substrate_addresses(
    payload: bytes, monkeypatch: pytest.MonkeyPatch
) -> None:
    import socket as socket_mod

    from hud.environment.egress import ANY_HOST, Egress

    monkeypatch.setattr(
        socket_mod,
        "getaddrinfo",
        lambda *args, **kwargs: [
            (
                socket_mod.AF_INET,
                socket_mod.SOCK_STREAM,
                socket_mod.IPPROTO_TCP,
                "",
                ("127.0.0.1", 8765),
            )
        ],
    )
    sockets = Path(tempfile.mkdtemp(dir="/tmp"))
    egress = Egress(sockets, {ANY_HOST})
    egress.start()
    try:
        client = socket_mod.socket(socket_mod.AF_UNIX, socket_mod.SOCK_STREAM)
        client.settimeout(5)
        client.connect(str(egress.socket_path))
        client.sendall(payload)
        response = client.recv(4096)
        client.close()
    finally:
        egress.stop()
        shutil.rmtree(sockets, ignore_errors=True)

    assert b"403" in response
    assert b"X-Proxy-Error: blocked-by-address" in response


def test_plain_http_upstream_failures_return_a_proxy_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hud.environment import egress as egress_mod
    from hud.environment.egress import Egress

    def unavailable(*_args: object, **_kwargs: object) -> socket.socket:
        raise ConnectionRefusedError

    monkeypatch.setattr(egress_mod, "_connect_public", unavailable)
    sockets = Path(tempfile.mkdtemp(dir="/tmp"))
    egress = Egress(sockets, {"example.com"})
    egress.start()
    try:
        client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        client.settimeout(5)
        client.connect(str(egress.socket_path))
        client.sendall(b"GET http://example.com/ HTTP/1.1\r\nHost: example.com\r\n\r\n")
        response = client.recv(4096)
        client.close()
    finally:
        egress.stop()
        shutil.rmtree(sockets, ignore_errors=True)

    assert b"502" in response
    assert b"X-Proxy-Error: upstream-failure" in response


def test_a_visitor_proxy_requires_its_ephemeral_credential() -> None:
    import socket as socket_mod

    from hud.environment.egress import Egress

    sockets = Path(tempfile.mkdtemp(dir="/tmp"))
    egress = Egress(sockets, {"example.com"}, token="secret")
    egress.start()
    try:
        client = socket_mod.socket(socket_mod.AF_UNIX, socket_mod.SOCK_STREAM)
        client.settimeout(5)
        client.connect(str(egress.socket_path))
        client.sendall(b"GET http://example.com/ HTTP/1.1\r\nHost: example.com\r\n\r\n")
        response = client.recv(4096)
        client.close()
    finally:
        egress.stop()
        shutil.rmtree(sockets, ignore_errors=True)

    assert b"407 Proxy Authentication Required" in response
    assert egress.environment()["http_proxy"].startswith("http://hud:secret@")


def test_a_workspace_that_reaches_no_host_is_told_of_no_proxy() -> None:
    """Pointing a client at a proxy that was never started turns "this task
    has no network" into a connection failure on the first hop."""
    from hud.environment.egress import Egress, Peer

    assert Egress("/tmp/unused", set(), [Peer("db", 5432)]).environment() == {}
    assert Egress("/tmp/unused", {"pypi.org"}).environment()["https_proxy"].endswith(":3128")


def test_a_host_is_permitted_by_name_or_as_a_subdomain() -> None:
    from hud.environment.egress import ANY_HOST, permitted

    assert permitted("pypi.org", {"pypi.org"})
    assert permitted("files.pypi.org", {"pypi.org"})  # a subdomain of what was named
    assert not permitted("notpypi.org", {"pypi.org"})  # not a subdomain, a different host
    assert not permitted("pypi.org.evil.com", {"pypi.org"})
    assert not permitted("anything", set())  # declared unreachable
    assert permitted("anything", {ANY_HOST})
    assert not permitted(None, {ANY_HOST})


def test_shell_uid_wraps_sessions_in_setpriv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _wall(monkeypatch)
    ws = Workspace(tmp_path / "root", shell_uid=1000)
    argv = ws.shell_argv("echo hi")
    # Absolute path: a bare name would resolve through the session PATH,
    # which the agent can influence — that lookup happens before the drop.
    # --no-new-privs: a setuid binary must not let the shell regain root.
    assert argv[:8] == [
        "/usr/bin/setpriv",
        "--reuid",
        "1000",
        "--regid",
        "1000",
        "--clear-groups",
        "--no-new-privs",
        "--",
    ]
    # The session env rides `env -i` *inside* the setpriv wrapper, so it only
    # takes effect after the drop.
    assert argv[8].endswith("/env") and argv[9] == "-i"
    assert "echo hi" in argv


def test_shell_identity_keeps_the_declared_primary_group(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _wall(monkeypatch)
    ws = Workspace(tmp_path / "root", shell_uid=1000, shell_gid=2000)

    argv = ws.shell_argv("id")

    assert argv[argv.index("--reuid") + 1] == "1000"
    assert argv[argv.index("--regid") + 1] == "2000"


def test_caller_env_is_injected_only_after_the_drop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An agent-influenced var like LD_PRELOAD must not be in the environment
    of the root-run setpriv; it may only reach the post-drop shell."""
    _wall(monkeypatch)
    ws = Workspace(tmp_path / "root", shell_uid=1000, env={"LD_PRELOAD": "/workspace/evil.so"})
    argv = ws.shell_argv("echo hi")
    assert argv[0] == "/usr/bin/setpriv"
    assert "LD_PRELOAD=/workspace/evil.so" in argv[argv.index("-i") :]


@pytest.mark.asyncio
async def test_wall_handoff_is_top_level_only_and_never_walks_the_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The handoff must be O(1): only the workspace dir is chowned. Recursing
    over baked content (node_modules, a venv) on the serving path would delay
    the control-port bind past the deploy readiness probe."""
    _wall(monkeypatch)
    handed: list[str] = []
    monkeypatch.setattr(os, "lchown", lambda p, u, g: handed.append(os.fsdecode(p)))
    monkeypatch.setattr(os, "walk", lambda *a, **k: pytest.fail("handoff must not walk the tree"))
    monkeypatch.setattr(os, "fwalk", lambda *a, **k: pytest.fail("handoff must not walk the tree"))

    root = tmp_path / "root"
    (root / "pkg").mkdir(parents=True)
    (root / "pkg" / "mod.py").write_text("x = 1\n")

    ws = Workspace(root, shell_uid=1000)
    await ws.start()
    try:
        assert [Path(p).name for p in handed] == ["root"]
    finally:
        await ws.stop()


@pytest.mark.asyncio
async def test_wall_handoff_failure_refuses_to_serve(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed ownership handoff leaves a workspace the agent can't write;
    the server must fail loudly instead of serving it."""
    _wall(monkeypatch)

    def deny(p: object, u: int, g: int) -> None:
        raise PermissionError("operation not permitted")

    monkeypatch.setattr(os, "lchown", deny)
    ws = Workspace(tmp_path / "root", shell_uid=1000)
    with pytest.raises(PermissionError):
        await ws.start()


def test_shell_uid_is_a_noop_off_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(Workspace, "_drops_privileges", lambda self: False)
    ws = Workspace(tmp_path / "root", shell_uid=1000)
    assert "setpriv" not in ws.shell_argv("echo hi")
    assert ws._session_env() is None


def test_without_shell_uid_argv_is_unchanged(tmp_path: Path) -> None:
    ws = Workspace(tmp_path / "root")
    assert "setpriv" not in ws.shell_argv("echo hi")


@pytest.mark.asyncio
async def test_session_wrapper_environment_contains_no_server_secrets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class SpawnCaptured(Exception):
        def __init__(self, env: dict[str, str] | None) -> None:
            self.env = env

    async def capture_spawn(*_args: str, **kwargs: Any) -> None:
        raise SpawnCaptured(kwargs.get("env"))

    monkeypatch.setenv("HUD_API_KEY", "super-secret")
    ws = Workspace(tmp_path / "root")
    monkeypatch.setattr(ws, "sandbox_pid", AsyncMock(return_value=7))
    ws._namespace = cast("Any", SimpleNamespace(spawn=capture_spawn))
    process = SimpleNamespace(term_type=None, command="true", env={})

    with pytest.raises(SpawnCaptured) as captured:
        await ws._handle_process(cast("Any", process))

    assert captured.value.env == {"PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin")}


@pytest.mark.asyncio
async def test_namespace_wait_status_is_forwarded_to_ssh_client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def wait_until_cancelled() -> None:
        await asyncio.Event().wait()

    def empty_reader() -> asyncio.StreamReader:
        reader = asyncio.StreamReader()
        reader.feed_eof()
        return reader

    child_channel = SimpleNamespace(close=Mock(), wait_closed=AsyncMock())
    child_process = SimpleNamespace(
        stdin=SimpleNamespace(write_eof=Mock()),
        stdout=empty_reader(),
        stderr=empty_reader(),
        wait=AsyncMock(return_value=SimpleNamespace(returncode=None)),
        returncode=None,
        channel=child_channel,
    )
    child = namespace_mod.NamespaceProcess(cast("Any", child_process))
    namespace = SimpleNamespace(spawn=AsyncMock(return_value=child))
    channel = SimpleNamespace(
        wait_closed=wait_until_cancelled,
        is_closing=Mock(return_value=False),
    )
    process = SimpleNamespace(
        term_type=None,
        command="true",
        stdin=SimpleNamespace(read=AsyncMock(return_value=b"")),
        stdout=SimpleNamespace(write=Mock(), drain=AsyncMock()),
        stderr=SimpleNamespace(write=Mock(), drain=AsyncMock()),
        channel=channel,
        exit=Mock(),
    )
    ws = Workspace(tmp_path / "root")
    monkeypatch.setattr(ws, "sandbox_pid", AsyncMock(return_value=7))
    ws._namespace = cast("Any", namespace)

    await ws._handle_process(cast("Any", process))

    process.exit.assert_called_once_with(255)
    child_channel.close.assert_not_called()


@pytest.mark.asyncio
async def test_root_without_working_drop_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Serving as root while unable to drop must refuse rather than run agents
    as root."""
    monkeypatch.setattr("hud.environment.workspace._is_root", lambda: True)
    monkeypatch.setattr(Workspace, "_drops_privileges", lambda self: False)
    ws = Workspace(tmp_path / "root", shell_uid=1000)
    with pytest.raises(RuntimeError, match="privileges cannot be dropped"):
        await ws.start()


def test_credentials_dir_is_private_and_unpredictable(tmp_path: Path) -> None:
    ws = Workspace(tmp_path / "root")
    creds = ws._credentials_dir()
    assert creds.is_relative_to(Path(tempfile.gettempdir()))
    assert not creds.is_relative_to(ws.root)
    # mkdtemp yields 0700 and a fresh name each call (no shared parent to hijack).
    assert (creds.stat().st_mode & 0o777) == 0o700
    assert ws._credentials_dir() == creds  # cached per instance


def test_usable_bwrap_reports_unusable_installs(monkeypatch) -> None:
    """An installed bwrap that cannot create namespaces must not be used."""
    import subprocess

    from hud.environment import workspace as ws

    monkeypatch.setattr(ws, "_bwrap_usable", None)
    monkeypatch.setattr(ws.shutil, "which", lambda _name: "/usr/bin/bwrap")
    monkeypatch.setattr(
        ws.subprocess,
        "run",
        lambda *a, **k: subprocess.CompletedProcess(a[0], 1, b"", b"No permissions"),
    )

    assert ws.usable_bwrap() is None


def test_windows_job_owns_the_process_tree() -> None:
    def first_thread(_snapshot: int, entry_pointer: Any) -> bool:
        entry_pointer._obj.owner_process_id = 123
        entry_pointer._obj.thread_id = 456
        return True

    kernel32 = SimpleNamespace(
        CreateJobObjectW=Mock(return_value=42),
        SetInformationJobObject=Mock(return_value=True),
        AssignProcessToJobObject=Mock(return_value=True),
        CreateToolhelp32Snapshot=Mock(return_value=43),
        Thread32First=Mock(side_effect=first_thread),
        Thread32Next=Mock(return_value=False),
        OpenThread=Mock(return_value=44),
        ResumeThread=Mock(return_value=1),
        CloseHandle=Mock(return_value=True),
        GetLastError=Mock(return_value=0),
    )
    job = workspace_mod._WindowsJob(kernel32)
    child = SimpleNamespace(_handle=99, pid=123)

    job.assign(cast("Any", child))
    job.resume(cast("Any", child))
    job.close()
    job.close()

    set_limits = kernel32.SetInformationJobObject.call_args.args
    assert set_limits[1] == workspace_mod._JOB_OBJECT_EXTENDED_LIMIT_INFORMATION
    assert (
        set_limits[2]._obj.basic_limit_information.limit_flags
        == workspace_mod._JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    )
    kernel32.AssignProcessToJobObject.assert_called_once_with(42, 99)
    kernel32.CreateToolhelp32Snapshot.assert_called_once_with(workspace_mod._TH32CS_SNAPTHREAD, 0)
    kernel32.OpenThread.assert_called_once_with(workspace_mod._THREAD_SUSPEND_RESUME, False, 456)
    kernel32.ResumeThread.assert_called_once_with(44)
    assert [call.args[0] for call in kernel32.CloseHandle.call_args_list] == [44, 43, 42]


@pytest.mark.asyncio
async def test_windows_channel_close_terminates_the_process_job(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exited = threading.Event()

    class Child:
        _handle = 99
        stdout = None
        stderr = None
        returncode: int | None = None

        def communicate(self) -> tuple[bytes, bytes]:
            assert exited.wait(1)
            self.returncode = -1
            return b"", b""

        def kill(self) -> None:
            self.returncode = -9
            exited.set()

    child = Child()
    job = SimpleNamespace(
        assign=Mock(),
        resume=Mock(),
        close=Mock(side_effect=exited.set),
    )
    channel = SimpleNamespace(
        wait_closed=AsyncMock(),
        is_closing=Mock(return_value=True),
    )
    process = SimpleNamespace(
        term_type=None,
        command="cmd /c echo hello",
        channel=channel,
        stdout=SimpleNamespace(write=Mock()),
        stderr=SimpleNamespace(write=Mock()),
        exit=Mock(),
    )
    ws = Workspace(tmp_path / "root")
    popen = Mock(return_value=child)
    monkeypatch.setattr(workspace_mod.sys, "platform", "win32")
    monkeypatch.setattr(workspace_mod, "_WindowsJob", Mock(return_value=job))
    monkeypatch.setattr(workspace_mod.subprocess, "Popen", popen)
    monkeypatch.setattr(ws, "sandbox_pid", AsyncMock(return_value=None))

    await ws._handle_process(cast("Any", process))

    job.assign.assert_called_once_with(child)
    job.resume.assert_called_once_with(child)
    job.close.assert_called_once()
    assert popen.call_args.kwargs["creationflags"] == workspace_mod._CREATE_SUSPENDED
    process.exit.assert_not_called()


def test_usable_bwrap_stages_pid_creation_when_direct_mounting_is_blocked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import subprocess

    from hud.environment import workspace as ws

    monkeypatch.setattr(ws, "_bwrap_usable", None)
    binaries = {
        "bwrap": "/usr/bin/bwrap",
        "unshare": "/usr/bin/unshare",
        "true": "/usr/bin/true",
    }
    monkeypatch.setattr(ws.shutil, "which", binaries.get)
    calls: list[list[str]] = []

    def run(argv: list[str], **_: object) -> subprocess.CompletedProcess[bytes]:
        calls.append(argv)
        return subprocess.CompletedProcess(argv, int(len(calls) == 1), b"", b"blocked")

    monkeypatch.setattr(ws.subprocess, "run", run)

    assert ws.usable_bwrap() == Bubblewrap("/usr/bin/bwrap", pid_unshare="/usr/bin/unshare")
    assert "--unshare-pid" in calls[0]
    assert calls[1][:4] == [
        "/usr/bin/unshare",
        "--kill-child=KILL",
        "--pid",
        "--mount-proc",
    ]
    assert "--unshare-pid" not in calls[1]


def test_staged_bwrap_keeps_user_isolation_and_uses_the_staged_proc(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ws = Workspace(tmp_path / "root")
    monkeypatch.setattr(
        ws,
        "_bwrap",
        Bubblewrap("/usr/bin/bwrap", pid_unshare="/usr/bin/unshare"),
    )

    argv = ws.bwrap_argv(["true"])

    assert argv[:5] == [
        "/usr/bin/unshare",
        "--kill-child=KILL",
        "--pid",
        "--mount-proc",
        "/usr/bin/bwrap",
    ]
    assert "--unshare-user-try" in argv
    assert "--unshare-pid" not in argv
    assert "--proc" in argv

    joined = ws.bwrap_argv(["true"], isolate_processes=False, isolate_users=False)
    assert joined[0] == "/usr/bin/bwrap"
    assert "/usr/bin/unshare" not in joined
    assert "--proc" in joined
    dev = joined.index("--dev-bind")
    assert joined[dev : dev + 3] == ["--dev-bind", "/dev", "/dev"]


def test_required_isolation_refuses_when_unavailable(monkeypatch, tmp_path) -> None:
    from hud.environment import workspace as ws

    monkeypatch.setattr(ws, "usable_bwrap", lambda: None)

    with pytest.raises(RuntimeError, match="isolation was required"):
        ws.Workspace(tmp_path, require_isolation=True)


@pytest.mark.asyncio
async def test_identity_map_reads_a_chunked_info_document() -> None:
    """bwrap's info JSON arrives in as many chunks as the pipe delivers; a
    reader that stops at the first chunk parses a truncated document and the
    sandbox never starts (observed as every session failing on a live box)."""
    info_read, info_write = os.pipe()
    block_read, block_write = os.pipe()
    document = b'{\n "child-pid": 4242,\n "other": "field"\n}\n'

    def write_in_chunks() -> None:
        os.write(info_write, document[:21])  # cut mid-document, line 2
        time.sleep(0.05)
        os.write(info_write, document[21:])
        os.close(info_write)

    writer = threading.Thread(target=write_in_chunks)
    writer.start()
    try:
        with mock.patch.object(namespace_mod, "_map_identities") as mapped:
            pid = await namespace_mod.install_identity_map(info_read, block_write)
        assert pid == 4242
        mapped.assert_called_once_with(4242)
        assert os.read(block_read, 1) == b"\n"  # the sandbox was released
    finally:
        writer.join()
        for fd in (info_read, block_read, block_write):
            with contextlib.suppress(OSError):
                os.close(fd)


@pytest.mark.asyncio
async def test_identity_map_resolves_a_staged_sandbox_to_its_parent_pid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    info_read, info_write = os.pipe()
    block_read, block_write = os.pipe()
    os.write(info_write, b'{"child-pid": 14}')
    os.close(info_write)
    children = {
        "/proc/100/task/100/children": "102 ",
        "/proc/102/task/102/children": "121 ",
    }

    def read_text(path: Path, *args: object, **kwargs: object) -> str:
        return children[str(path)]

    monkeypatch.setattr(Path, "read_text", read_text)
    try:
        with mock.patch.object(namespace_mod, "_map_identities") as mapped:
            pid = await namespace_mod.install_identity_map(
                info_read,
                block_write,
                launcher_pid=100,
                launcher_depth=2,
            )
        assert pid == 121
        mapped.assert_called_once_with(121)
        assert os.read(block_read, 1) == b"\n"
    finally:
        for fd in (info_read, block_read, block_write):
            with contextlib.suppress(OSError):
                os.close(fd)


def test_identity_map_preserves_every_id_available_to_the_container(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    maps = {
        "/proc/self/uid_map": "0 100000 131072\n",
        "/proc/self/gid_map": "0 200000 65536\n70000 300000 1000\n",
    }
    writes: dict[str, str] = {}

    def read_text(path: Path) -> str:
        return maps[str(path)]

    def write_text(path: Path, value: str) -> int:
        writes[str(path)] = value
        return len(value)

    monkeypatch.setattr(Path, "read_text", read_text)
    monkeypatch.setattr(Path, "write_text", write_text)

    namespace_mod._map_identities(42)

    assert writes["/proc/42/uid_map"] == "0 0 131072\n"
    assert writes["/proc/42/gid_map"] == "0 0 65536\n70000 70000 1000\n"


def test_the_proxy_refuses_to_relay_a_header_it_cannot_represent() -> None:
    """An upstream header is remote text, and a folded value keeps its CRLF
    through http.client. Relayed verbatim it would carry headers of its own
    into the response the workspace reads, so a field outside the grammar
    fails the whole response rather than being quietly repaired."""
    assert _field("Content-Type", "text/plain") == ("Content-Type", "text/plain")
    assert _field("X-Meta", "") == ("X-Meta", "")
    for name, value in (
        ("X-Evil", "a\r\n b"),
        ("X-Evil", "a\nX-Injected: 1"),
        ("X-Evil", "a\x00b"),
        ("Bad Name", "fine"),
        ("X-Evil\r\nX-Injected", "fine"),
    ):
        with pytest.raises(_Unrelayable):
            _field(name, value)


def test_private_workspace_maps_its_own_hostname_to_loopback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(socket, "gethostname", lambda: "container-self")
    ws = Workspace(
        tmp_path / "root",
        allowed_hosts=set(),
        local_aliases=["main"],
        credentials_dir=tmp_path / "keys",
        hosts_path=tmp_path / "hosts",
    )

    generated = ws._write_hosts().read_text("utf-8")

    assert "127.0.0.1\tmain" in generated
    assert "127.0.0.1\tcontainer-self" in generated


def test_child_pid_discovery_falls_back_to_proc_stat(monkeypatch: pytest.MonkeyPatch) -> None:
    entries = [Path("/proc/101"), Path("/proc/102"), Path("/proc/self")]

    def read_text(path: Path, *args: object, **kwargs: object) -> str:
        if path.name == "children":
            raise OSError("unavailable")
        if str(path) == "/proc/101/stat":
            return "101 (child one) S 100 0 0 0"
        if str(path) == "/proc/102/stat":
            return "102 (other) S 99 0 0 0"
        raise OSError("gone")

    monkeypatch.setattr(Path, "iterdir", lambda path: iter(entries))
    monkeypatch.setattr(Path, "read_text", read_text)

    assert namespace_mod._child_pids(100) == [101]
