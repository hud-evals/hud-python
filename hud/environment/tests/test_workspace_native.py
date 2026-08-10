"""Native Workspace SSH file-write contracts."""

from __future__ import annotations

import asyncio
import os
import sys
from typing import TYPE_CHECKING

import asyncssh
import pytest

from hud.capabilities import SSHClient
from hud.capabilities.ssh import SSHConnectionError
from hud.environment.workspace import Workspace, _sftp_server_path

if TYPE_CHECKING:
    from pathlib import Path, PurePath

pytestmark = [
    pytest.mark.skipif(sys.platform == "win32", reason="POSIX workspace semantics"),
    pytest.mark.skipif(_sftp_server_path() is None, reason="OpenSSH sftp-server is unavailable"),
]


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


def _staged_files(root: Path) -> list[Path]:
    return list(root.glob(".hud-write.*"))


async def test_cancelled_file_write_keeps_existing_destination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    destination = root / "destination.txt"
    destination.write_text("old", encoding="utf-8")
    commit_ready = asyncio.Event()

    async def block_commit(
        client: asyncssh.SFTPClient,
        staged: str | bytes | PurePath,
        target: str | bytes | PurePath,
    ) -> None:
        del client, staged, target
        commit_ready.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(asyncssh.SFTPClient, "posix_rename", block_commit)
    ws = Workspace(root)
    await ws.start()
    try:
        async with await _connect(ws) as connection:
            write = asyncio.create_task(
                SSHClient(ws.capability(), connection).write_text("destination.txt", "new")
            )
            await commit_ready.wait()

            assert destination.read_text(encoding="utf-8") == "old"
            write.cancel()
            with pytest.raises(asyncio.CancelledError):
                await write
    finally:
        await ws.stop()

    assert destination.read_text(encoding="utf-8") == "old"
    assert not _staged_files(root)


async def test_timed_out_file_write_keeps_existing_destination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    destination = root / "destination.txt"
    destination.write_text("old", encoding="utf-8")

    async def block_commit(
        client: asyncssh.SFTPClient,
        staged: str | bytes | PurePath,
        target: str | bytes | PurePath,
    ) -> None:
        del client, staged, target
        await asyncio.Event().wait()

    monkeypatch.setattr(asyncssh.SFTPClient, "posix_rename", block_commit)
    ws = Workspace(root)
    await ws.start()
    try:
        async with await _connect(ws) as connection:
            with pytest.raises(TimeoutError):
                await SSHClient(ws.capability(), connection).write_text(
                    "destination.txt",
                    "new",
                    timeout_s=0.05,
                )
    finally:
        await ws.stop()

    assert destination.read_text(encoding="utf-8") == "old"
    assert not _staged_files(root)


async def test_connection_loss_before_commit_keeps_existing_destination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    destination = root / "destination.txt"
    destination.write_text("old", encoding="utf-8")
    commit_ready = asyncio.Event()
    connection: asyncssh.SSHClientConnection
    original_commit = asyncssh.SFTPClient.posix_rename

    async def lose_before_commit(
        client: asyncssh.SFTPClient,
        staged: str | bytes | PurePath,
        target: str | bytes | PurePath,
    ) -> None:
        commit_ready.set()
        await connection.wait_closed()
        await original_commit(client, staged, target)

    monkeypatch.setattr(asyncssh.SFTPClient, "posix_rename", lose_before_commit)
    ws = Workspace(root)
    await ws.start()
    connection = await _connect(ws)
    try:
        write = asyncio.create_task(
            SSHClient(ws.capability(), connection).write_text("destination.txt", "new")
        )
        await commit_ready.wait()
        connection.abort()

        with pytest.raises(SSHConnectionError, match="lost during file operation"):
            await write
    finally:
        connection.abort()
        await connection.wait_closed()
        await ws.stop()

    assert destination.read_text(encoding="utf-8") == "old"


async def test_file_write_atomically_replaces_symlink_target_with_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    target = root / "target with [spaces]*.txt"
    target.write_text("old", encoding="utf-8")
    target.chmod(0o640)
    metadata = (target.stat().st_mode, target.stat().st_uid, target.stat().st_gid)
    link = root / "link.txt"
    link.symlink_to(target.name)
    commit_ready = asyncio.Event()
    allow_commit = asyncio.Event()
    original_commit = asyncssh.SFTPClient.posix_rename

    async def block_commit(
        client: asyncssh.SFTPClient,
        staged: str | bytes | PurePath,
        destination: str | bytes | PurePath,
    ) -> None:
        commit_ready.set()
        await allow_commit.wait()
        await original_commit(client, staged, destination)

    monkeypatch.setattr(asyncssh.SFTPClient, "posix_rename", block_commit)
    ws = Workspace(root)
    await ws.start()
    try:
        async with await _connect(ws) as connection:
            write = asyncio.create_task(
                SSHClient(ws.capability(), connection).write_text(
                    "link.txt",
                    "complete new content",
                )
            )
            await commit_ready.wait()

            assert target.read_text(encoding="utf-8") == "old"
            assert link.is_symlink()
            staged = _staged_files(root)
            assert len(staged) == 1
            assert staged[0].read_text(encoding="utf-8") == "complete new content"

            allow_commit.set()
            await write
    finally:
        await ws.stop()

    assert target.read_text(encoding="utf-8") == "complete new content"
    assert link.is_symlink()
    assert (target.stat().st_mode, target.stat().st_uid, target.stat().st_gid) == metadata
    assert not _staged_files(root)


async def test_new_file_write_observes_the_process_umask(tmp_path: Path) -> None:
    root = tmp_path / "root"
    ws = Workspace(root)
    previous_umask = os.umask(0o027)
    try:
        await ws.start()
        async with await _connect(ws) as connection:
            await SSHClient(ws.capability(), connection).write_text("new.txt", "content")
    finally:
        os.umask(previous_umask)
        await ws.stop()

    assert (root / "new.txt").stat().st_mode & 0o777 == 0o640
