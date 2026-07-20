"""SSHClient — asyncssh connection wrapper."""

from __future__ import annotations

import base64
import posixpath
import shlex
from typing import Any, ClassVar, Self
from urllib.parse import urlsplit

import asyncssh

from .base import Capability, CapabilityClient


class SSHClient(CapabilityClient):
    """Thin asyncssh wrapper. Exposes the raw connection via ``conn``."""

    protocol: ClassVar[str] = "ssh/2"

    def __init__(self, capability: Capability, conn: asyncssh.SSHClientConnection) -> None:
        self.capability = capability
        self._conn = conn

    @classmethod
    async def connect(cls, cap: Capability) -> Self:
        parts = urlsplit(cap.url)
        if parts.hostname is None or parts.port is None:
            raise ValueError(f"ssh capability missing host or port: {cap.url!r}")
        # Key content travels in the binding (works across network
        # namespaces); a key path only works on a shared filesystem.
        client_keys: list[Any] | None = None
        if client_key := cap.params.get("client_key"):
            client_keys = [asyncssh.import_private_key(client_key)]
        elif client_key_path := cap.params.get("client_key_path"):
            client_keys = [client_key_path]
        conn = await asyncssh.connect(
            host=parts.hostname,
            port=parts.port,
            username=cap.params.get("user", "agent"),
            client_keys=client_keys,
            known_hosts=None,
        )
        return cls(cap, conn)

    @property
    def conn(self) -> asyncssh.SSHClientConnection:
        """Raw asyncssh connection for commands and port forwarding."""
        return self._conn

    def map_path(self, path: str) -> str:
        """Anchor a path to the session cwd (the served workspace).

        The old SFTP subsystem was chrooted, so harness tools address files as
        ``/REPORT.md`` meaning workspace-relative. The exec channel sees the
        real filesystem; replicate the chroot: strip the cwd prefix if present,
        normalize the remainder against ``/`` (clamping ``..`` at the root,
        exactly as a chroot does), and re-anchor under the cwd. Idempotent.
        """
        cwd = str(self.capability.params.get("cwd", "")).rstrip("/")
        if not cwd:
            return path
        if self._is_windows:
            # The workspace publishes cwd via as_posix() (e.g. "C:/work") but
            # callers pass native paths ("C:\work\file.txt"); NTFS paths are
            # case-insensitive.
            path = path.replace("\\", "/")
            if path.lower() == cwd.lower() or path.lower().startswith(cwd.lower() + "/"):
                path = path[len(cwd) :]
            elif len(path) >= 2 and path[1] == ":" and path[0].isalpha():
                # Drive-absolute outside the workspace: anchor like the chroot.
                path = path[2:]
        elif path == cwd or path.startswith(cwd + "/"):
            path = path[len(cwd) :]
        normalized = posixpath.normpath("/" + path.lstrip("/"))
        return cwd if normalized == "/" else cwd + normalized

    async def read_text(self, path: str) -> str:
        """Read a UTF-8 text file through the exec channel."""
        path = self.map_path(path)
        if self._is_windows:
            quoted = _powershell_quote(path)
            command = (
                "powershell -NoProfile -NonInteractive -Command "
                f'"[Convert]::ToBase64String([IO.File]::ReadAllBytes({quoted}))"'
            )
            result = await self._conn.run(command, check=True)
            return base64.b64decode(_stdout(result)).decode("utf-8", errors="replace")
        # encoding=None transports raw bytes: a strict connection-level UTF-8
        # decode would raise on files with invalid UTF-8 instead of replacing.
        result = await self._conn.run(f"cat -- {shlex.quote(path)}", check=True, encoding=None)
        return _decode(result.stdout)

    async def write_text(self, path: str, content: str) -> None:
        """Write UTF-8 text through the exec channel without command interpolation."""
        path = self.map_path(path)
        if self._is_windows:
            quoted = _powershell_quote(path)
            await self._conn.run(
                "powershell -NoProfile -NonInteractive -Command "
                f'"[IO.File]::WriteAllBytes({quoted},[byte[]]@())"',
                check=True,
            )
            raw = content.encode("utf-8")
            for offset in range(0, len(raw), 6144):
                payload = base64.b64encode(raw[offset : offset + 6144]).decode("ascii")
                await self._conn.run(
                    "powershell -NoProfile -NonInteractive -Command "
                    f"\"$b=[Convert]::FromBase64String('{payload}');"
                    f"$f=[IO.File]::Open({quoted},[IO.FileMode]::Append,"
                    "[IO.FileAccess]::Write,[IO.FileShare]::Read);"
                    'try{$f.Write($b,0,$b.Length)}finally{$f.Dispose()}"',
                    check=True,
                )
            return
        await self._conn.run(f"cat > {shlex.quote(path)}", input=content, check=True)

    async def listdir(self, path: str) -> list[str]:
        """List direct children through the exec channel."""
        path = self.map_path(path)
        if self._is_windows:
            quoted = _powershell_quote(path)
            command = (
                "powershell -NoProfile -NonInteractive -Command "
                f'"Get-ChildItem -Force -Name -LiteralPath {quoted}"'
            )
            result = await self._conn.run(command, check=True)
            listing = _stdout(result)
        else:
            command = f"ls -1A -- {shlex.quote(path)}"
            result = await self._conn.run(command, check=True, encoding=None)
            listing = _decode(result.stdout)
        return sorted(line for line in listing.splitlines() if line not in (".", ".."))

    @property
    def _is_windows(self) -> bool:
        return self.capability.params.get("shell") in ("cmd", "powershell")

    async def close(self) -> None:
        self._conn.close()
        await self._conn.wait_closed()


def _stdout(result: asyncssh.SSHCompletedProcess) -> str:
    return result.stdout if isinstance(result.stdout, str) else ""


def _decode(raw: object) -> str:
    if isinstance(raw, bytes):
        return raw.decode("utf-8", errors="replace")
    return raw if isinstance(raw, str) else ""


def _powershell_quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


__all__ = ["SSHClient"]
