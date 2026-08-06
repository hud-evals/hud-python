"""Trusted namespace host for Workspace processes."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import pty
import shutil
import socket
import struct
import sys
from pathlib import Path
from typing import Any, Literal

import asyncssh

from hud.utils.process import ProcessGroup, ProcessResult, create_process_group_exec

_AF_NETLINK = getattr(socket, "AF_NETLINK", 16)
_NETLINK_ROUTE = getattr(socket, "NETLINK_ROUTE", 0)


async def read_bwrap_pid(info_read: int) -> int:
    """Read the child pid from bwrap's possibly chunked info document."""
    raw = b""
    document: dict[str, Any] | None = None
    async with asyncio.timeout(30.0):
        while chunk := await asyncio.to_thread(os.read, info_read, 4096):
            raw += chunk
            with contextlib.suppress(json.JSONDecodeError):
                document = json.loads(raw)
                break
    if document is None:
        raise RuntimeError("bubblewrap did not report its child pid")
    return int(document["child-pid"])


class NamespaceProcess:
    """A process launched by a :class:`NamespaceHost`."""

    def __init__(self, process: asyncssh.SSHClientProcess[bytes]) -> None:
        self._process = process
        self.stdin = process.stdin
        self.stdout = process.stdout
        self.stderr = process.stderr

    @property
    def returncode(self) -> int | None:
        return self._process.returncode

    async def wait(self) -> int:
        result = await self._process.wait()
        return result.returncode if result.returncode is not None else 255

    async def terminate(self) -> None:
        self._process.channel.close()
        with contextlib.suppress(Exception):
            await self._process.wait_closed()

    async def complete(self, *, max_wait: float | None = None) -> ProcessResult:
        try:
            result = await self._process.wait(timeout=max_wait)
        except asyncssh.TimeoutError as exc:
            await self.terminate()
            stdout, stderr = exc.stdout or b"", exc.stderr or b""
            assert isinstance(stdout, bytes) and isinstance(stderr, bytes)
            return ProcessResult(exc.returncode, stdout, stderr, True)
        stdout, stderr = result.stdout or b"", result.stderr or b""
        assert isinstance(stdout, bytes) and isinstance(stderr, bytes)
        return ProcessResult(result.returncode, stdout, stderr)

    async def resize(self, width: int, height: int, pixwidth: int, pixheight: int) -> None:
        self._process.change_terminal_size(width, height, pixwidth, pixheight)


class NamespaceHost:
    """Client for the trusted process which owns a Workspace namespace."""

    def __init__(self, socket_path: Path) -> None:
        self.socket_path = socket_path
        self._connection: asyncssh.SSHClientConnection | None = None

    async def connect(self) -> None:
        if self._connection is not None:
            return
        sock = socket.socket(socket.AF_UNIX)
        sock.setblocking(False)
        try:
            await asyncio.get_running_loop().sock_connect(sock, str(self.socket_path))
            self._connection = await asyncssh.connect(
                sock=sock,
                username="hud",
                known_hosts=None,
                encoding=None,
            )
        except BaseException:
            sock.close()
            raise

    async def close(self) -> None:
        connection, self._connection = self._connection, None
        if connection is not None:
            connection.close()
            await connection.wait_closed()

    async def spawn(
        self,
        argv: list[str],
        *,
        cwd: Path,
        env: dict[str, str],
        mount_view: Literal["workspace", "host"] = "workspace",
        tty: bool = False,
        terminal_size: tuple[int, int, int, int] = (80, 24, 0, 0),
        persistent: bool = False,
    ) -> NamespaceProcess:
        connection = self._require_connection()
        request = json.dumps(
            {
                "argv": argv,
                "cwd": str(cwd),
                "env": env,
                "mount_view": mount_view,
                "persistent": persistent,
            }
        )
        if tty:
            process = await connection.create_process(
                request,
                request_pty=True,
                term_type="xterm",
                term_size=terminal_size,
                encoding=None,
            )
        else:
            process = await connection.create_process(request, encoding=None)
        return NamespaceProcess(process)

    def _require_connection(self) -> asyncssh.SSHClientConnection:
        if self._connection is None:
            raise RuntimeError("workspace namespace host is not connected")
        return self._connection


class _NoAuth(asyncssh.SSHServer):
    def begin_auth(self, username: str) -> bool:
        return False


class _NamespaceHost:
    def __init__(
        self,
        socket_path: Path,
        *,
        setup_loopback: bool,
        holder_argv: list[str],
        bwrap: str,
        launcher_depth: int,
    ) -> None:
        self.socket_path = socket_path
        self.setup_loopback = setup_loopback
        self.holder_argv = holder_argv
        self.bwrap = bwrap
        self.launcher_depth = launcher_depth
        self.holder: ProcessGroup | None = None
        self.holder_pid: int | None = None

    async def serve(self) -> None:
        server: asyncssh.SSHAcceptor | None = None
        try:
            if self.setup_loopback:
                self._enable_loopback()
            self.holder_pid = await self._start_holder()
            with contextlib.suppress(FileNotFoundError):
                self.socket_path.unlink()
            listener = socket.socket(socket.AF_UNIX)
            listener.bind(str(self.socket_path))
            listener.listen()
            listener.setblocking(False)
            self.socket_path.chmod(0o600)
            server = await asyncssh.listen(
                sock=listener,
                server_host_keys=[asyncssh.generate_private_key("ssh-ed25519")],
                server_factory=_NoAuth,
                process_factory=self._handle,
                encoding=None,
            )
            sys.stdout.write(json.dumps({"holder_pid": self.holder_pid}) + "\n")
            sys.stdout.flush()
            await asyncio.Event().wait()
        finally:
            if server is not None:
                server.close()
                await server.wait_closed()
            await self.stop()

    async def stop(self) -> None:
        if self.holder is not None:
            await self.holder.terminate()
            self.holder = None
            self.holder_pid = None
        with contextlib.suppress(FileNotFoundError):
            self.socket_path.unlink()

    async def _handle(self, process: asyncssh.SSHServerProcess[bytes]) -> None:
        try:
            if process.command is None:
                raise ValueError("spawn request required")
            request: dict[str, Any] = json.loads(process.command)
            process.exit(await self._spawn(request, process))
        except Exception as exc:
            process.stderr.write(str(exc).encode())
            process.exit(1)
        await process.wait_closed()

    async def _start_holder(self) -> int:
        read_fd, write_fd = os.pipe()
        try:
            os.set_inheritable(write_fd, True)
            argv = list(self.holder_argv)
            index = argv.index(self.bwrap) + 1
            argv[index:index] = ["--info-fd", str(write_fd)]
            self.holder = await create_process_group_exec(
                *argv,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                pass_fds=(write_fd,),
            )
            os.close(write_fd)
            write_fd = -1
            pid = await read_bwrap_pid(read_fd)
            if self.launcher_depth:
                pid = self.holder.process.pid
            for _ in range(self.launcher_depth):
                children = Path(f"/proc/{pid}/task/{pid}/children").read_text().split()
                if len(children) != 1:
                    raise RuntimeError(f"sandbox launcher {pid} has {len(children)} children")
                pid = int(children[0])
            assert self.holder.stdout is not None
            if await asyncio.wait_for(self.holder.stdout.readline(), 30.0) != b"ready\n":
                raise RuntimeError(await self._holder_error())
            return pid
        except BaseException:
            if self.holder is not None:
                await self.holder.terminate()
                self.holder = None
            raise
        finally:
            os.close(read_fd)
            if write_fd != -1:
                os.close(write_fd)

    async def _holder_error(self) -> str:
        if self.holder is None or self.holder.stderr is None:
            return "sandbox holder did not become ready"
        detail = await self.holder.stderr.read(2048)
        return detail.decode(errors="replace").strip() or "sandbox holder did not become ready"

    async def _spawn(
        self,
        request: dict[str, Any],
        channel: asyncssh.SSHServerProcess[bytes],
    ) -> int:
        if self.holder_pid is None:
            raise RuntimeError("workspace holder is not running")
        mount_view = request["mount_view"]
        if mount_view not in ("workspace", "host"):
            raise ValueError(f"unknown mount view {mount_view!r}")
        command_prefix: list[str] = []
        if mount_view == "host":
            unshare = shutil.which("unshare")
            if unshare is None:
                raise RuntimeError("trusted workspace commands require unshare")
            command_prefix = [
                unshare,
                "--mount",
                "--propagation",
                "private",
                "--mount-proc",
                "--",
            ]
        argv = [
            shutil.which("nsenter") or "/usr/bin/nsenter",
            "--target",
            str(self.holder_pid),
            *(("--mount",) if mount_view == "workspace" else ()),
            "--pid",
            "--uts",
            "--ipc",
            "--preserve-credentials",
            "--",
            *command_prefix,
            *request["argv"],
        ]
        if channel.term_type:
            master_fd, slave_fd = pty.openpty()
            process = await create_process_group_exec(
                *argv,
                stdin=slave_fd,
                stdout=slave_fd,
                stderr=slave_fd,
                cwd=request["cwd"],
                env=request["env"],
            )
            os.close(slave_fd)
            await channel.redirect(
                stdin=os.dup(master_fd),
                stdout=os.dup(master_fd),
                send_eof=False,
            )
            os.close(master_fd)
        else:
            process = await create_process_group_exec(
                *argv,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=request["cwd"],
                env=request["env"],
            )
            assert process.process.stdin is not None
            assert process.stdout is not None and process.stderr is not None
            await channel.redirect(
                stdin=process.process.stdin,
                stdout=process.stdout,
                stderr=process.stderr,
                send_eof=False,
            )
        wait_task = asyncio.create_task(process.wait())
        closed_task = asyncio.create_task(channel.channel.wait_closed())
        try:
            done, _ = await asyncio.wait(
                (wait_task, closed_task),
                return_when=asyncio.FIRST_COMPLETED,
            )
            if closed_task in done and not wait_task.done():
                await process.terminate()
            returncode = await wait_task
            if not request["persistent"]:
                await process.terminate()
            return returncode
        finally:
            wait_task.cancel()
            closed_task.cancel()
            await asyncio.gather(wait_task, closed_task, return_exceptions=True)

    @staticmethod
    def _enable_loopback() -> None:
        request = struct.pack(
            "IHHIIBBHiII",
            32,
            16,
            5,
            1,
            os.getpid(),
            socket.AF_UNSPEC,
            0,
            0,
            socket.if_nametoindex("lo"),
            1,
            1,
        )
        with socket.socket(_AF_NETLINK, socket.SOCK_RAW, _NETLINK_ROUTE) as route:
            route.send(request)
            reply = route.recv(4096)
        _, message_type, _, _, _ = struct.unpack_from("IHHII", reply)
        if message_type == 2:
            error = struct.unpack_from("i", reply, 16)[0]
            if error:
                raise OSError(-error, os.strerror(-error))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("socket", type=Path)
    parser.add_argument("--setup-loopback", action="store_true")
    args = parser.parse_args()
    config = json.loads(sys.stdin.buffer.readline())
    asyncio.run(
        _NamespaceHost(
            args.socket,
            setup_loopback=args.setup_loopback,
            holder_argv=config["holder_argv"],
            bwrap=config["bwrap"],
            launcher_depth=config["launcher_depth"],
        ).serve()
    )


if __name__ == "__main__":
    main()
