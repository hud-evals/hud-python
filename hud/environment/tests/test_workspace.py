"""Workspace contract tests: credential placement and the shell_uid wall."""

from __future__ import annotations

import asyncio
import itertools
import os
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import asyncssh
import pytest

from hud.capabilities import SSHClient
from hud.environment import workspace as workspace_mod
from hud.environment.workspace import Mount, Workspace

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
            assert await client.listdir(".") == ["hello world.txt"]
            # Absolute paths anchor to the workspace, like the old SFTP chroot.
            await client.write_text("/REPORT.md", "done")
            assert (tmp_path / "root" / "REPORT.md").read_text() == "done"
            assert await client.read_text("/REPORT.md") == "done"
            assert "REPORT.md" in await client.listdir("/")
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
async def test_a_timed_out_command_keeps_what_it_printed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The output is the evidence of how far it got — reporting only that the
    deadline passed throws that away."""
    monkeypatch.setattr(workspace_mod, "_COMMAND_TIMEOUT", 1.0)
    ws = Workspace(tmp_path / "root")
    await ws.start()
    try:
        async with await _connect(ws) as conn:
            result = await conn.run("echo progress-so-far; sleep 30", check=False)
    finally:
        await ws.stop()

    assert "progress-so-far" in str(result.stdout)
    assert "timed out" in str(result.stderr)
    assert result.exit_status == 1


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
    monkeypatch.setattr(ws, "_bwrap", "/usr/bin/bwrap")
    argv = ws.shell_argv("echo hi", env={"PER_CALL": "1"})

    sandbox_env = _sandbox_env(argv)
    assert "HUD_API_KEY" not in sandbox_env
    assert sandbox_env["CUSTOM"] == "1"
    assert "PATH" in sandbox_env
    assert sandbox_env["PER_CALL"] == "1"


def test_bwrap_inherits_host_env_when_not_walled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HUD_SENTINEL", "visible")
    ws = Workspace(tmp_path / "root")
    monkeypatch.setattr(ws, "_bwrap", "/usr/bin/bwrap")
    argv = ws.bwrap_argv(["bash", "-lc", "true"])
    assert _sandbox_env(argv)["HUD_SENTINEL"] == "visible"


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
    monkeypatch.setattr(ws, "_bwrap", "/usr/bin/bwrap")
    _wall(monkeypatch)

    for argv in (ws.shell_argv("echo hi"), ws.shell_argv(), ws.bwrap_argv(["true"])):
        assert not _POST_0_4_BWRAP_OPTIONS.intersection(argv)


def test_sessions_join_one_sandbox_rather_than_each_making_its_own(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two commands must land in the same namespaces, or a process the first
    backgrounds is gone by the second."""
    ws = Workspace(tmp_path / "root", guest_path="/app", network=True)
    monkeypatch.setattr(ws, "_bwrap", "/usr/bin/bwrap")

    first = ws.enter_argv(4321, "start-a-server &")
    second = ws.enter_argv(4321, "curl localhost")

    assert first[0].endswith("/nsenter") and second[0].endswith("/nsenter")
    # Same target, so the same live namespaces — not a fresh sandbox per command.
    assert first[first.index("--target") + 1] == "4321"
    assert second[second.index("--target") + 1] == "4321"
    assert "--pid" in first and "--mount" in first and "--user" in first
    # The user namespace has to be joined first: it is what confers the
    # privilege to join the others in a container given no extra capability.
    assert first.index("--user") < min(first.index("--mount"), first.index("--pid"))
    assert "--wd=/app" in first


@pytest.mark.asyncio
async def test_concurrent_sessions_share_one_sandbox(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An agent issues parallel tool calls; if each started its own sandbox,
    what one backgrounds would be invisible to the next."""
    ws = Workspace(tmp_path / "root")
    monkeypatch.setattr(ws, "_bwrap", "/usr/bin/bwrap")
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


def test_a_sharing_sandbox_is_not_rejoined_by_network_namespace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Joining bwrap's user namespace forfeits authority over the container's
    netns, so asking for it fails the session outright; a severed sandbox owns
    its netns and must be joined or the network comes back."""
    shared = Workspace(tmp_path / "shared", network=True)
    severed = Workspace(tmp_path / "severed", network=False)
    for ws in (shared, severed):
        monkeypatch.setattr(ws, "_bwrap", "/usr/bin/bwrap")

    assert "--net" not in shared.enter_argv(11, "true")
    assert "--net" in severed.enter_argv(11, "true")


def test_the_sandbox_reports_readiness_before_sessions_join_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """bwrap names the child pid before that child has built its mount
    namespace, so the pid alone is not proof the sandbox can run anything."""
    ws = Workspace(tmp_path / "root")
    monkeypatch.setattr(ws, "_bwrap", "/usr/bin/bwrap")
    argv = ws.bwrap_argv(["sh", "-c", "echo ready"], info_fd=7)

    assert argv[argv.index("--info-fd") + 1] == "7"
    # The signal comes from the payload, which bwrap runs only after setup.
    assert argv[-1] == "echo ready"


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


def test_required_isolation_refuses_when_unavailable(monkeypatch, tmp_path) -> None:
    from hud.environment import workspace as ws

    monkeypatch.setattr(ws, "usable_bwrap", lambda: None)

    with pytest.raises(RuntimeError, match="isolation was required"):
        ws.Workspace(tmp_path, require_isolation=True)


@pytest.mark.asyncio
async def test_a_symlinked_root_publishes_both_spellings(tmp_path: Path) -> None:
    """A workspace addressed through a symlink (macOS /tmp -> /private/tmp) serves the
    real path, so it must publish the caller's spelling too or clients re-anchor it."""
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real, target_is_directory=True)

    ws = Workspace(link)
    await ws.start()
    try:
        cap = ws.capability()
        assert cap.params["cwd"] == real.as_posix()
        assert cap.params["cwd_aliases"] == [link.as_posix()]

        client = SSHClient(cap, cast("Any", None))
        assert client.map_path(f"{link}/calc.py") == f"{real}/calc.py"
    finally:
        await ws.stop()


@pytest.mark.asyncio
async def test_a_plain_root_publishes_no_alias(tmp_path: Path) -> None:
    ws = Workspace(tmp_path / "root")
    await ws.start()
    try:
        assert "cwd_aliases" not in ws.capability().params
    finally:
        await ws.stop()
