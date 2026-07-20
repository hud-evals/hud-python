"""Workspace contract tests: credential placement and the shell_uid wall."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import asyncssh
import pytest

from hud.environment.workspace import Workspace, _PrefixSFTPServer

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
            stdout = result.stdout
            assert isinstance(stdout, str) and "ok" in stdout
    finally:
        await ws.stop()
    assert not key_path.exists()


@pytest.mark.asyncio
async def test_sftp_writes_are_handed_to_the_shell_uid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SFTP runs in the serving process; when privileges are dropped, files it
    creates must end up owned by the dropped uid."""
    claimed: list[tuple[str, int, int]] = []
    monkeypatch.setattr(os, "chown", lambda p, u, g: claimed.append((os.fsdecode(p), u, g)))
    monkeypatch.setattr(Workspace, "_drops_privileges", lambda self: True)

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
    # The workspace root itself is handed over so the dropped uid can write to it.
    assert owners.get("root") == (1000, 1000)


@pytest.mark.asyncio
async def test_sftp_overwrites_do_not_rechown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    claimed: list[str] = []
    monkeypatch.setattr(os, "chown", lambda p, u, g: claimed.append(os.fsdecode(p)))
    monkeypatch.setattr(Workspace, "_drops_privileges", lambda self: True)

    root = tmp_path / "root"
    root.mkdir()
    (root / "existing.txt").write_text("before")
    ws = Workspace(root, shell_uid=1000)
    await ws.start()
    claimed.clear()  # ignore the root chown done at prepare time
    try:
        async with await _connect(ws) as conn, conn.start_sftp_client() as sftp:
            local = tmp_path / "existing.txt"
            local.write_text("after")
            await sftp.put(str(local), "/existing.txt")
    finally:
        await ws.stop()
    assert claimed == []


@pytest.mark.asyncio
async def test_dropped_session_env_excludes_server_secrets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A dropped shell must not inherit the server's environment (secrets)."""
    monkeypatch.setattr(Workspace, "_drops_privileges", lambda self: True)
    monkeypatch.setenv("HUD_API_KEY", "super-secret")

    ws = Workspace(tmp_path / "root", shell_uid=1000, env={"CUSTOM": "1"})
    session_env = ws._session_env()
    assert session_env is not None
    assert "HUD_API_KEY" not in session_env
    assert session_env["CUSTOM"] == "1"
    assert "PATH" in session_env


def _sftp_server(root: Path, *, chown_uid: int | None) -> _PrefixSFTPServer:
    # map_path doesn't touch the channel; None is enough to exercise it.
    return _PrefixSFTPServer(
        None,  # type: ignore[arg-type]
        chroot=str(root).encode(),
        guest_prefix=b"/workspace",
        chown_uid=chown_uid,
    )


def test_sftp_map_path_blocks_symlink_escape_under_the_wall(tmp_path: Path) -> None:
    """Every SFTP op resolves paths through map_path; under the wall it must
    refuse ones whose real target escapes the workspace (e.g. an on-disk
    ``escape -> /host/secret`` the dropped shell created), which a raw
    ``open`` would otherwise follow as root."""
    root = tmp_path / "root"
    root.mkdir()
    secret = tmp_path / "secret.txt"
    secret.write_text("host-only")
    (root / "escape").symlink_to(secret)
    (root / "ok.txt").write_text("inside")

    srv = _sftp_server(root, chown_uid=1000)
    with pytest.raises(asyncssh.SFTPNoSuchFile):
        srv.map_path(b"/escape")
    # Paths that stay inside are unaffected.
    assert srv.map_path(b"/ok.txt") == str(root / "ok.txt").encode()


def test_sftp_map_path_leaves_escape_to_asyncssh_without_the_wall(tmp_path: Path) -> None:
    """Without the wall, the containment guard is off — asyncssh's own
    prefix chroot governs, so behavior for normal workspaces is unchanged."""
    root = tmp_path / "root"
    root.mkdir()
    (root / "escape").symlink_to(tmp_path / "secret.txt")
    srv = _sftp_server(root, chown_uid=None)
    assert srv.map_path(b"/escape") == str(root / "escape").encode()


def test_bwrap_drops_host_env_when_walled(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The bwrap path must not re-inject host secrets via --setenv."""
    monkeypatch.setenv("HUD_API_KEY", "super-secret")
    monkeypatch.setattr(Workspace, "_drops_privileges", lambda self: True)

    ws = Workspace(tmp_path / "root", shell_uid=1000, env={"CUSTOM": "1"})
    monkeypatch.setattr(ws, "_bwrap", "/usr/bin/bwrap")
    argv = ws.shell_argv("echo hi")

    setenv_keys = {argv[i + 1] for i, tok in enumerate(argv) if tok == "--setenv"}
    assert "HUD_API_KEY" not in setenv_keys
    assert "CUSTOM" in setenv_keys and "PATH" in setenv_keys


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
    monkeypatch.setattr(Workspace, "_drops_privileges", lambda self: True)
    ws = Workspace(tmp_path / "root", shell_uid=1000)
    argv = ws.shell_argv("echo hi")
    assert argv[:7] == ["setpriv", "--reuid", "1000", "--regid", "1000", "--clear-groups", "--"]
    assert "echo hi" in argv


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
