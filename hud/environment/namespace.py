"""Trusted namespace host for Workspace processes."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import logging
import os
import pty
import shutil
import socket
import struct
import sys
from pathlib import Path
from typing import Any, Literal

import asyncssh
from typing_extensions import override

from hud.environment.utils import splice
from hud.utils.process import ProcessGroup, ProcessResult, create_process_group_exec

_AF_NETLINK = getattr(socket, "AF_NETLINK", 16)
_NETLINK_ROUTE = getattr(socket, "NETLINK_ROUTE", 0)
LOGGER = logging.getLogger("hud.environment.namespace")


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


async def install_identity_map(
    info_read: int,
    block_write: int,
    *,
    launcher_pid: int | None = None,
    launcher_depth: int = 0,
) -> int:
    """Map every available host identity into a blocked bwrap user namespace."""
    pid = await read_bwrap_pid(info_read)
    if launcher_pid is not None:
        pid = launcher_pid
        for _ in range(launcher_depth):
            children_file = Path(f"/proc/{pid}/task/{pid}/children")
            children = (await asyncio.to_thread(children_file.read_text)).split()
            if len(children) != 1:
                raise RuntimeError(f"sandbox launcher {pid} has {len(children)} children")
            pid = int(children[0])
    await asyncio.to_thread(_map_identities, pid)
    await asyncio.to_thread(os.write, block_write, b"\n")
    return pid


def _map_identities(pid: int) -> None:
    proc = Path(f"/proc/{pid}")
    with contextlib.suppress(OSError):
        (proc / "setgroups").write_text("allow")
    for name, own in (("uid_map", os.geteuid()), ("gid_map", os.getegid())):
        identity_map = ""
        for line in (Path("/proc/self") / name).read_text().splitlines():
            start, _, length = line.split()
            identity_map += f"{start} {start} {length}\n"
        try:
            (proc / name).write_text(identity_map)
        except OSError:
            if name == "gid_map":
                with contextlib.suppress(OSError):
                    (proc / "setgroups").write_text("deny")
            try:
                (proc / name).write_text(f"{own} {own} 1")
            except OSError:
                LOGGER.warning("could not map ids into the sandbox (%s)", name)


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
        self._listeners: list[asyncio.AbstractServer] = []
        self._handlers: set[asyncio.Task[None]] = set()

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
        for listener in self._listeners:
            listener.close()
        for handler in self._handlers:
            handler.cancel()
        await asyncio.gather(*self._handlers, return_exceptions=True)
        self._handlers.clear()
        for listener in self._listeners:
            await listener.wait_closed()
        self._listeners.clear()
        connection, self._connection = self._connection, None
        if connection is not None:
            connection.close()
            await connection.wait_closed()

    async def forward(self, port: int) -> None:
        async def accept(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
            task = asyncio.current_task()
            if task is not None:
                self._handlers.add(task)
            try:
                await _to_unix(reader, writer, _port_socket(self.socket_path, port))
            finally:
                if task is not None:
                    self._handlers.discard(task)

        self._listeners.append(
            await asyncio.start_server(
                accept,
                "0.0.0.0",  # noqa: S104 - publish into the substrate network
                port,
            )
        )

    async def spawn(
        self,
        argv: list[str],
        *,
        cwd: Path,
        env: dict[str, str],
        mount_view: Literal["workspace", "host"] = "workspace",
        identity: int | tuple[int, int] | None = None,
        tty: bool = False,
        terminal_size: tuple[int, int, int, int] = (80, 24, 0, 0),
        persistent: bool = False,
        scope: Literal["session", "environment"] = "session",
    ) -> NamespaceProcess:
        connection = self._require_connection()
        request = json.dumps(
            {
                "argv": argv,
                "cwd": str(cwd),
                "env": env,
                "mount_view": mount_view,
                "identity": identity,
                "persistent": persistent,
                "scope": scope,
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

    async def terminate_sessions(self) -> None:
        connection = self._require_connection()
        result = await connection.run(
            json.dumps({"operation": "terminate_sessions"}),
            check=False,
            encoding=None,
        )
        if result.returncode != 0:
            stderr = result.stderr or b""
            assert isinstance(stderr, bytes)
            detail = stderr.decode("utf-8", "replace").strip()
            raise RuntimeError(detail or "failed to terminate workspace sessions")

    def _require_connection(self) -> asyncssh.SSHClientConnection:
        if self._connection is None:
            raise RuntimeError("workspace namespace host is not connected")
        return self._connection


class _NoAuth(asyncssh.SSHServer):
    @override
    def begin_auth(self, username: str) -> bool:
        return False


def _port_socket(control_socket: Path, port: int) -> Path:
    return control_socket.with_name(f"port-{port}.sock")


async def _to_unix(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    path: Path,
) -> None:
    try:
        peer_reader, peer_writer = await asyncio.open_unix_connection(path)
    except OSError:
        writer.close()
        await writer.wait_closed()
        return
    await splice((reader, writer), (peer_reader, peer_writer))


async def _to_port(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    port: int,
) -> None:
    try:
        peer_reader, peer_writer = await asyncio.open_connection("127.0.0.1", port)
    except OSError:
        writer.close()
        await writer.wait_closed()
        return
    await splice((reader, writer), (peer_reader, peer_writer))


class _NamespaceHost:
    def __init__(
        self,
        socket_path: Path,
        *,
        setup_loopback: bool,
        holder_argv: list[str],
        bwrap: str,
        launcher_depth: int,
        map_identities: bool,
        ports: frozenset[int],
    ) -> None:
        self.socket_path = socket_path
        self.setup_loopback = setup_loopback
        self.holder_argv = holder_argv
        self.bwrap = bwrap
        self.launcher_depth = launcher_depth
        self.map_identities = map_identities
        self.ports = ports
        self.holders: dict[Literal["session", "environment"], tuple[ProcessGroup, int]] = {}
        self.forwarders: list[asyncio.AbstractServer] = []

    async def serve(self) -> None:
        server: asyncssh.SSHAcceptor | None = None
        try:
            if self.setup_loopback:
                self._enable_loopback()
            self.holders["session"] = await self._start_holder()
            self.holders["environment"] = await self._start_holder()
            with contextlib.suppress(FileNotFoundError):
                self.socket_path.unlink()
            listener = socket.socket(socket.AF_UNIX)
            listener.bind(str(self.socket_path))
            listener.listen()
            listener.setblocking(False)
            self.socket_path.chmod(0o600)
            for port in sorted(self.ports):
                path = _port_socket(self.socket_path, port)
                with contextlib.suppress(FileNotFoundError):
                    path.unlink()
                self.forwarders.append(
                    await asyncio.start_unix_server(
                        lambda reader, writer, port=port: _to_port(
                            reader,
                            writer,
                            port,
                        ),
                        path,
                    )
                )
                path.chmod(0o600)
            server = await asyncssh.listen(
                sock=listener,
                server_host_keys=[asyncssh.generate_private_key("ssh-ed25519")],
                server_factory=_NoAuth,
                process_factory=self._handle,
                encoding=None,
            )
            sys.stdout.write(json.dumps({"holder_pid": self.holders["session"][1]}) + "\n")
            sys.stdout.flush()
            await asyncio.Event().wait()
        finally:
            if server is not None:
                server.close()
                await server.wait_closed()
            await self.stop()

    async def stop(self) -> None:
        for server in self.forwarders:
            server.close()
        for server in self.forwarders:
            await server.wait_closed()
        self.forwarders.clear()
        for port in self.ports:
            with contextlib.suppress(FileNotFoundError):
                _port_socket(self.socket_path, port).unlink()
        holders = tuple(self.holders.values())
        self.holders.clear()
        await asyncio.gather(*(holder.terminate() for holder, _ in holders))
        with contextlib.suppress(FileNotFoundError):
            self.socket_path.unlink()

    async def _handle(self, process: asyncssh.SSHServerProcess[bytes]) -> None:
        try:
            if process.command is None:
                raise ValueError("spawn request required")
            request: dict[str, Any] = json.loads(process.command)
            if request.get("operation") == "terminate_sessions":
                await self._terminate_sessions()
                process.exit(0)
            else:
                process.exit(await self._spawn(request, process))
        except Exception as exc:
            process.stderr.write(str(exc).encode())
            process.exit(1)
        await process.wait_closed()

    async def _start_holder(self) -> tuple[ProcessGroup, int]:
        read_fd, write_fd = os.pipe()
        block_read = block_write = -1
        holder: ProcessGroup | None = None
        try:
            os.set_inheritable(write_fd, True)
            argv = list(self.holder_argv)
            index = argv.index(self.bwrap) + 1
            argv[index:index] = ["--info-fd", str(write_fd)]
            if self.map_identities:
                block_read, block_write = os.pipe()
                os.set_inheritable(block_read, True)
                user_index = argv.index("--unshare-user-try")
                argv[user_index] = "--unshare-user"
                argv[user_index + 1 : user_index + 1] = [
                    "--userns-block-fd",
                    str(block_read),
                ]
            holder = await create_process_group_exec(
                *argv,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                pass_fds=(write_fd, *((block_read,) if self.map_identities else ())),
            )
            os.close(write_fd)
            write_fd = -1
            if self.map_identities:
                os.close(block_read)
                block_read = -1
                pid = await install_identity_map(
                    read_fd,
                    block_write,
                    launcher_pid=holder.process.pid,
                    launcher_depth=self.launcher_depth,
                )
                os.close(block_write)
                block_write = -1
            else:
                pid = await read_bwrap_pid(read_fd)
            assert holder.stdout is not None
            if await asyncio.wait_for(holder.stdout.readline(), 30.0) != b"ready\n":
                raise RuntimeError(await self._holder_error(holder))
            return holder, pid
        except BaseException as exc:
            if holder is not None:
                with contextlib.suppress(Exception):
                    await holder.terminate()
                if holder.stderr is not None:
                    with contextlib.suppress(Exception):
                        detail = (
                            (await asyncio.wait_for(holder.stderr.read(2048), 1.0))
                            .decode(errors="replace")
                            .strip()
                        )
                        if detail:
                            exc.add_note(f"sandbox holder stderr: {detail}")
            raise
        finally:
            os.close(read_fd)
            for descriptor in (write_fd, block_read, block_write):
                if descriptor != -1:
                    os.close(descriptor)

    async def _holder_error(self, holder: ProcessGroup) -> str:
        if holder.stderr is None:
            return "sandbox holder did not become ready"
        detail = await holder.stderr.read(2048)
        return detail.decode(errors="replace").strip() or "sandbox holder did not become ready"

    async def _terminate_sessions(self) -> None:
        held = self.holders.pop("session", None)
        if held is not None:
            holder, _ = held
            await holder.terminate()

    async def _spawn(
        self,
        request: dict[str, Any],
        channel: asyncssh.SSHServerProcess[bytes],
    ) -> int:
        mount_view = request["mount_view"]
        if mount_view not in ("workspace", "host"):
            raise ValueError(f"unknown mount view {mount_view!r}")
        scope = request["scope"]
        if scope not in ("session", "environment"):
            raise ValueError(f"unknown process scope {scope!r}")
        if scope == "environment" and mount_view != "host":
            raise ValueError("environment processes require the host mount view")
        identity = request["identity"]
        identity_argv: tuple[str, ...] = ()
        if isinstance(identity, int):
            identity_argv = ("--setgid", str(identity), "--setuid", str(identity))
        elif identity is not None:
            uid, gid = map(int, identity)
            identity_argv = ("--setgid", str(gid), "--setuid", str(uid))
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
                *identity_argv,
                "--",
            ]
        held = self.holders.get(scope)
        if held is None:
            raise RuntimeError(f"workspace {scope} holder is not running")
        _, holder_pid = held
        argv = [
            shutil.which("nsenter") or "/usr/bin/nsenter",
            "--target",
            str(holder_pid),
            *(("--mount",) if mount_view == "workspace" else ()),
            "--pid",
            "--uts",
            "--ipc",
            *(
                ("--preserve-credentials",)
                if identity is None or mount_view == "host"
                else identity_argv
            ),
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
    parser.add_argument("--port", action="append", type=int, default=[])
    args = parser.parse_args()
    config = json.loads(sys.stdin.buffer.readline())
    asyncio.run(
        _NamespaceHost(
            args.socket,
            setup_loopback=args.setup_loopback,
            holder_argv=config["holder_argv"],
            bwrap=config["bwrap"],
            launcher_depth=config["launcher_depth"],
            map_identities=config["map_identities"],
            ports=frozenset(args.port),
        ).serve()
    )


if __name__ == "__main__":
    main()
