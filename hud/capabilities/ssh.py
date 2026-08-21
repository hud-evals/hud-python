"""SSHClient — asyncssh connection wrapper."""

from __future__ import annotations

import asyncio
import base64
import contextlib
import shlex
from typing import Any, ClassVar, Self
from urllib.parse import urlsplit

import asyncssh

from .base import Capability, CapabilityClient

SSH_RECONNECT_ATTEMPTS = 3
SSH_RECONNECT_BASE_DELAY_S = 0.25
SSH_SESSION_CLOSE_TIMEOUT_S = 5.0


class SSHConnectionError(ConnectionError):
    """SSH transport was unavailable for an operation."""


class SSHClient(CapabilityClient):
    """Thin asyncssh wrapper. Exposes the raw connection via ``conn``.

    File helpers pass paths to the session verbatim: relative paths resolve
    against the session cwd, absolute paths mean what they say. The namespace
    the session runs in is the only path truth — file helpers and shell
    commands must never disagree about what a path names.
    """

    protocol: ClassVar[str] = "ssh/2"

    def __init__(
        self,
        capability: Capability,
        conn: asyncssh.SSHClientConnection,
        *,
        default_timeout_s: float | None = None,
    ) -> None:
        self.capability = capability
        self._conn = conn
        self.default_timeout_s = default_timeout_s
        self._reconnect_lock = asyncio.Lock()
        self._closing = False

    @classmethod
    async def connect(cls, cap: Capability, *, default_timeout_s: float | None = None) -> Self:
        try:
            connection = await cls._connect(cap)
        except (ConnectionError, asyncssh.ConnectionLost) as exc:
            raise SSHConnectionError("SSH connection failed during handshake") from exc
        return cls(cap, connection, default_timeout_s=default_timeout_s)

    @staticmethod
    async def _connect(cap: Capability) -> asyncssh.SSHClientConnection:
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
        return await asyncssh.connect(
            host=parts.hostname,
            port=parts.port,
            username=cap.params.get("user", "agent"),
            client_keys=client_keys,
            known_hosts=None,
            errors="replace",
            keepalive_interval=15,
            keepalive_count_max=4,
        )

    @property
    def conn(self) -> asyncssh.SSHClientConnection:
        """Raw asyncssh connection for commands and port forwarding."""
        return self._conn

    async def create_process(self, command: str) -> asyncssh.SSHClientProcess[bytes]:
        """Open a binary exec channel after restoring a dropped SSH transport."""
        try:
            conn = await self._connection()
            return await conn.create_process(command, encoding=None)
        except asyncssh.ChannelOpenError as exc:
            raise SSHConnectionError("SSH server rejected the session") from exc
        except asyncssh.ConnectionLost as exc:
            raise SSHConnectionError("SSH connection lost while opening the session") from exc
        except SSHConnectionError:
            raise
        except (OSError, asyncssh.Error) as exc:
            if _is_closed(self._conn):
                raise SSHConnectionError("SSH connection lost while opening the session") from exc
            raise

    async def run(self, *args: object, **kwargs: Any) -> asyncssh.SSHCompletedProcess:
        """Run one command, reconnecting first when the transport is closed."""
        run_kwargs = dict(kwargs)
        timeout = self._effective_timeout(run_kwargs.pop("timeout", None))
        check = bool(run_kwargs.pop("check", False))
        conn: asyncssh.SSHClientConnection | None = None
        process = None
        try:
            async with asyncio.timeout(timeout):
                conn = await self._connection()
                process = await conn.create_process(*args, **run_kwargs)
                completed = await process.wait(check=check, timeout=None)
            if completed.returncode is None:
                conn.close()
                raise SSHConnectionError("SSH command ended without an exit status")
            return completed
        except (TimeoutError, asyncio.CancelledError):
            if process is not None:
                try:
                    process.terminate()
                except (OSError, asyncssh.Error):
                    process.close()
                try:
                    async with asyncio.timeout(SSH_SESSION_CLOSE_TIMEOUT_S):
                        await process.wait_closed()
                except (OSError, TimeoutError, asyncssh.Error):
                    process.close()
                    with contextlib.suppress(OSError, TimeoutError, asyncssh.Error):
                        async with asyncio.timeout(SSH_SESSION_CLOSE_TIMEOUT_S):
                            await process.wait_closed()
            raise
        except asyncssh.ChannelOpenError as exc:
            raise SSHConnectionError("SSH server rejected the session") from exc
        except asyncssh.ConnectionLost as exc:
            raise SSHConnectionError("SSH connection lost during operation") from exc
        except SSHConnectionError:
            raise
        except (OSError, asyncssh.Error) as exc:
            if conn is not None and _is_closed(conn):
                raise SSHConnectionError("SSH connection lost during operation") from exc
            raise

    async def _connection(self) -> asyncssh.SSHClientConnection:
        if self._closing:
            raise SSHConnectionError("SSH client is closed")
        if not _is_closed(self._conn):
            return self._conn

        async with self._reconnect_lock:
            if self._closing:
                raise SSHConnectionError("SSH client is closed")
            if not _is_closed(self._conn):
                return self._conn

            last_error: Exception | None = None
            for attempt in range(SSH_RECONNECT_ATTEMPTS):
                if self._closing:
                    raise SSHConnectionError("SSH client is closed")
                try:
                    conn = await self._connect(self.capability)
                except (OSError, asyncssh.Error) as exc:
                    last_error = exc
                    if self._closing:
                        raise SSHConnectionError("SSH client is closed") from exc
                    if attempt + 1 < SSH_RECONNECT_ATTEMPTS:
                        await asyncio.sleep(SSH_RECONNECT_BASE_DELAY_S * 2**attempt)
                    continue
                if self._closing:
                    conn.close()
                    await conn.wait_closed()
                    raise SSHConnectionError("SSH client is closed")
                self._conn = conn
                return conn
            raise SSHConnectionError(
                f"SSH reconnect failed after {SSH_RECONNECT_ATTEMPTS} attempts"
            ) from last_error

    async def read_text(self, path: str, *, timeout_s: float | None = None) -> str:
        """Read a UTF-8 text file through the exec channel."""
        timeout_s = self._effective_timeout(timeout_s)
        if self._is_windows:
            quoted = _powershell_quote(path)
            script = f"[Convert]::ToBase64String([IO.File]::ReadAllBytes({quoted}))"
            result = await self.run(_powershell(script), check=True, timeout=timeout_s)
            return base64.b64decode(_stdout(result)).decode("utf-8", errors="replace")
        # encoding=None transports raw bytes: a strict connection-level UTF-8
        # decode would raise on files with invalid UTF-8 instead of replacing.
        result = await self.run(
            f"cat -- {shlex.quote(path)}",
            check=True,
            encoding=None,
            timeout=timeout_s,
        )
        return _decode(result.stdout)

    async def write_text(
        self,
        path: str,
        content: str,
        *,
        timeout_s: float | None = None,
    ) -> None:
        """Write UTF-8 text through the exec channel without command interpolation."""
        timeout_s = self._effective_timeout(timeout_s)
        if self._is_windows:
            loop = asyncio.get_running_loop()
            deadline = loop.time() + timeout_s if timeout_s is not None else None

            def remaining_timeout() -> float | None:
                return None if deadline is None else max(0.0, deadline - loop.time())

            quoted = _powershell_quote(path)
            truncate = f"[IO.File]::WriteAllBytes({quoted},[byte[]]@())"
            await self.run(_powershell(truncate), check=True, timeout=remaining_timeout())
            raw = content.encode("utf-8")
            for offset in range(0, len(raw), 6144):
                payload = base64.b64encode(raw[offset : offset + 6144]).decode("ascii")
                script = (
                    f"$b=[Convert]::FromBase64String('{payload}');"
                    f"$f=[IO.File]::Open({quoted},[IO.FileMode]::Append,"
                    "[IO.FileAccess]::Write,[IO.FileShare]::Read);"
                    "try{$f.Write($b,0,$b.Length)}finally{$f.Dispose()}"
                )
                await self.run(_powershell(script), check=True, timeout=remaining_timeout())
            return
        run_kwargs: dict[str, Any] = {"input": content}
        if not content:
            # AsyncSSH treats empty input as absent; DEVNULL still delivers EOF.
            run_kwargs["stdin"] = asyncssh.DEVNULL
        await self.run(f"cat > {shlex.quote(path)}", check=True, timeout=timeout_s, **run_kwargs)

    async def listdir(self, path: str, *, timeout_s: float | None = None) -> list[str]:
        """List direct children through the exec channel."""
        timeout_s = self._effective_timeout(timeout_s)
        if self._is_windows:
            script = f"Get-ChildItem -Force -Name -LiteralPath {_powershell_quote(path)}"
            result = await self.run(_powershell(script), check=True, timeout=timeout_s)
            listing = _stdout(result)
        else:
            command = f"ls -1A -- {shlex.quote(path)}"
            result = await self.run(command, check=True, encoding=None, timeout=timeout_s)
            listing = _decode(result.stdout)
        return sorted(line for line in listing.splitlines() if line not in (".", ".."))

    def _effective_timeout(self, timeout_s: float | None) -> float | None:
        return self.default_timeout_s if timeout_s is None else timeout_s

    @property
    def _is_windows(self) -> bool:
        return self.capability.params.get("shell") in ("cmd", "powershell")

    async def close(self) -> None:
        self._closing = True
        async with self._reconnect_lock:
            self._conn.close()
            await self._conn.wait_closed()


def _is_closed(conn: asyncssh.SSHClientConnection) -> bool:
    return conn.is_closed()


def _powershell(script: str) -> str:
    """Wrap a script for remote execution with no quoting at all.

    ``-Command "..."`` breaks against the built-in Windows workspace: its exec
    handler splits with ``shlex.split(posix=False)`` (quotes retained) and
    ``subprocess`` re-quotes, so PowerShell sees a string literal instead of a
    script. ``-EncodedCommand`` carries the script as base64 UTF-16LE — no
    quotes to mangle anywhere.
    """
    encoded = base64.b64encode(script.encode("utf-16-le")).decode("ascii")
    return f"powershell -NoProfile -NonInteractive -EncodedCommand {encoded}"


def _stdout(result: asyncssh.SSHCompletedProcess) -> str:
    return result.stdout if isinstance(result.stdout, str) else ""


def _decode(raw: object) -> str:
    if isinstance(raw, bytes):
        return raw.decode("utf-8", errors="replace")
    return raw if isinstance(raw, str) else ""


def _powershell_quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


__all__ = ["SSHClient"]
