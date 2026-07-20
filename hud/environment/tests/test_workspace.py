"""Workspace contract tests: credential placement and the shell_uid wall."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import asyncssh
import pytest

from hud.capabilities import SSHClient
from hud.environment.workspace import Workspace

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


def test_bwrap_drops_host_env_when_walled(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The bwrap path must not re-inject host secrets via --setenv, while
    per-call env overrides still reach the sandbox."""
    monkeypatch.setenv("HUD_API_KEY", "super-secret")
    _wall(monkeypatch)

    ws = Workspace(tmp_path / "root", shell_uid=1000, env={"CUSTOM": "1"})
    monkeypatch.setattr(ws, "_bwrap", "/usr/bin/bwrap")
    argv = ws.shell_argv("echo hi", env={"PER_CALL": "1"})

    setenv_keys = {argv[i + 1] for i, tok in enumerate(argv) if tok == "--setenv"}
    assert "HUD_API_KEY" not in setenv_keys
    assert "CUSTOM" in setenv_keys and "PATH" in setenv_keys
    assert "PER_CALL" in setenv_keys


def test_bwrap_inherits_host_env_when_not_walled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HUD_SENTINEL", "visible")
    ws = Workspace(tmp_path / "root")
    monkeypatch.setattr(ws, "_bwrap", "/usr/bin/bwrap")
    argv = ws.bwrap_argv(["bash", "-lc", "true"])
    setenv_keys = {argv[i + 1] for i, tok in enumerate(argv) if tok == "--setenv"}
    assert "HUD_SENTINEL" in setenv_keys


def test_shell_uid_wraps_sessions_in_setpriv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _wall(monkeypatch)
    ws = Workspace(tmp_path / "root", shell_uid=1000)
    argv = ws.shell_argv("echo hi")
    # Absolute path: a bare name would resolve through the session PATH,
    # which the agent can influence — that lookup happens before the drop.
    assert argv[:7] == [
        "/usr/bin/setpriv",
        "--reuid",
        "1000",
        "--regid",
        "1000",
        "--clear-groups",
        "--",
    ]
    assert "echo hi" in argv


@pytest.mark.asyncio
async def test_wall_hands_preexisting_contents_to_the_dropped_uid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Contents baked in before serve (e.g. a Docker COPY as root) must become
    editable from the dropped shell, not just the top-level directory."""
    _wall(monkeypatch)
    handed: list[str] = []
    monkeypatch.setattr(os, "lchown", lambda p, u, g: handed.append(os.fsdecode(p)))

    root = tmp_path / "root"
    (root / "pkg").mkdir(parents=True)
    (root / "pkg" / "mod.py").write_text("x = 1\n")

    ws = Workspace(root, shell_uid=1000)
    await ws.start()
    try:
        names = {Path(p).name for p in handed}
        assert {"root", "pkg", "mod.py"} <= names
    finally:
        await ws.stop()


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
