"""Workspace contract tests: credential placement and the shell_uid wall."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import asyncssh
import pytest

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
    """The root is the agent's surface (shell cwd, SFTP chroot); key material
    must not be readable or writable through it."""
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
            assert result.stdout is not None and "ok" in result.stdout
    finally:
        await ws.stop()
    assert not key_path.exists()


@pytest.mark.asyncio
async def test_sftp_writes_are_handed_to_the_shell_uid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SFTP runs in the serving process; with shell_uid set (and the server
    root), files it creates must end up owned by the dropped uid."""
    claimed: list[tuple[str, int, int]] = []
    monkeypatch.setattr(os, "chown", lambda p, u, g: claimed.append((os.fsdecode(p), u, g)))
    monkeypatch.setattr("hud.environment.workspace._serving_as_root", lambda: True)

    ws = Workspace(tmp_path / "root", shell_uid=1000)
    await ws.start()
    try:
        async with await _connect(ws) as conn, conn.start_sftp_client() as sftp:
            local = tmp_path / "hello.txt"
            local.write_text("hi")
            await sftp.put(str(local), "/hello.txt")
            await sftp.mkdir("/subdir")
    finally:
        await ws.stop()

    owners = {Path(path).name: (uid, gid) for path, uid, gid in claimed}
    assert owners.get("hello.txt") == (1000, 1000)
    assert owners.get("subdir") == (1000, 1000)


@pytest.mark.asyncio
async def test_sftp_overwrites_do_not_rechown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    claimed: list[str] = []
    monkeypatch.setattr(os, "chown", lambda p, u, g: claimed.append(os.fsdecode(p)))
    monkeypatch.setattr("hud.environment.workspace._serving_as_root", lambda: True)

    root = tmp_path / "root"
    root.mkdir()
    (root / "existing.txt").write_text("before")
    ws = Workspace(root, shell_uid=1000)
    await ws.start()
    try:
        async with await _connect(ws) as conn, conn.start_sftp_client() as sftp:
            local = tmp_path / "existing.txt"
            local.write_text("after")
            await sftp.put(str(local), "/existing.txt")
    finally:
        await ws.stop()
    assert claimed == []


def test_shell_uid_wraps_sessions_in_setpriv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("hud.environment.workspace._serving_as_root", lambda: True)
    ws = Workspace(tmp_path / "root", shell_uid=1000)
    argv = ws.shell_argv("echo hi")
    assert argv[:7] == ["setpriv", "--reuid", "1000", "--regid", "1000", "--clear-groups", "--"]
    assert "echo hi" in argv


def test_shell_uid_is_a_noop_off_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("hud.environment.workspace._serving_as_root", lambda: False)
    ws = Workspace(tmp_path / "root", shell_uid=1000)
    assert "setpriv" not in ws.shell_argv("echo hi")
    assert "setpriv" not in ws.shell_argv("echo hi", cwd=str(tmp_path))


def test_without_shell_uid_argv_is_unchanged(tmp_path: Path) -> None:
    ws = Workspace(tmp_path / "root")
    assert "setpriv" not in ws.shell_argv("echo hi")


def test_credentials_dir_is_private(tmp_path: Path) -> None:
    ws = Workspace(tmp_path / "root")
    creds = ws._credentials_dir()
    assert creds.is_relative_to(Path(tempfile.gettempdir()))
    assert (creds.stat().st_mode & 0o777) == 0o700
