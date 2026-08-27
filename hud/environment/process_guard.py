"""Linux process-bound network connection enforcement."""

from __future__ import annotations

import argparse
import array
import asyncio
import contextlib
import ctypes
import errno
import fcntl
import ipaddress
import os
import platform
import select
import socket
import struct
import subprocess
import sys
import threading
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Collection, Sequence

_PR_SET_NO_NEW_PRIVS = 38
_SECCOMP_SET_MODE_FILTER = 1
_SECCOMP_FILTER_FLAG_NEW_LISTENER = 8
_SECCOMP_RET_KILL_PROCESS = 0x80000000
_SECCOMP_RET_USER_NOTIF = 0x7FC00000
_SECCOMP_RET_ERRNO = 0x00050000
_SECCOMP_RET_ALLOW = 0x7FFF0000

_BPF_LD_W_ABS = 0x20
_BPF_JMP_JEQ_K = 0x15
_BPF_RET_K = 0x06

_SECCOMP_IOCTL_NOTIF_RECV = 0xC0502100
_SECCOMP_IOCTL_NOTIF_SEND = 0xC0182101
_PIDFD_GETFD_SYSCALL = 438
_REGISTER_ADDRESS = b"\0hud-process-connection-register"
_SANDBOX_SOCKET = "/tmp/.hud-process-connection/control.sock"  # noqa: S108
_READY_TIMEOUT_SECONDS = 10.0

_LIBC = ctypes.CDLL(None, use_errno=True)
_supported: bool | None = None


class _SockFilter(ctypes.Structure):
    _fields_ = [
        ("code", ctypes.c_ushort),
        ("jt", ctypes.c_ubyte),
        ("jf", ctypes.c_ubyte),
        ("k", ctypes.c_uint),
    ]


class _SockFprog(ctypes.Structure):
    _fields_ = [("length", ctypes.c_ushort), ("filters", ctypes.POINTER(_SockFilter))]


class _SeccompData(ctypes.Structure):
    _fields_ = [
        ("nr", ctypes.c_int),
        ("arch", ctypes.c_uint),
        ("instruction_pointer", ctypes.c_ulonglong),
        ("args", ctypes.c_ulonglong * 6),
    ]


class _SeccompNotif(ctypes.Structure):
    _fields_ = [
        ("id", ctypes.c_ulonglong),
        ("pid", ctypes.c_uint),
        ("flags", ctypes.c_uint),
        ("data", _SeccompData),
    ]


class _SeccompNotifResp(ctypes.Structure):
    _fields_ = [
        ("id", ctypes.c_ulonglong),
        ("val", ctypes.c_longlong),
        ("error", ctypes.c_int),
        ("flags", ctypes.c_uint),
    ]


def _architecture() -> tuple[int, int, int, int, int]:
    machine = platform.machine()
    if machine == "x86_64":
        return 0xC000003E, 317, 42, 425, 426
    if machine in {"aarch64", "arm64"}:
        return 0xC00000B7, 277, 203, 425, 426
    raise RuntimeError(f"process-bound connections do not support {machine!r}")


def _install_connect_listener() -> int:
    audit_arch, seccomp_syscall, connect_syscall, io_uring_setup, io_uring_enter = _architecture()
    instructions = (_SockFilter * 11)(
        _SockFilter(_BPF_LD_W_ABS, 0, 0, 4),
        _SockFilter(_BPF_JMP_JEQ_K, 1, 0, audit_arch),
        _SockFilter(_BPF_RET_K, 0, 0, _SECCOMP_RET_KILL_PROCESS),
        _SockFilter(_BPF_LD_W_ABS, 0, 0, 0),
        _SockFilter(_BPF_JMP_JEQ_K, 0, 1, connect_syscall),
        _SockFilter(_BPF_RET_K, 0, 0, _SECCOMP_RET_USER_NOTIF),
        _SockFilter(_BPF_JMP_JEQ_K, 0, 1, io_uring_setup),
        _SockFilter(_BPF_RET_K, 0, 0, _SECCOMP_RET_ERRNO | errno.EPERM),
        _SockFilter(_BPF_JMP_JEQ_K, 0, 1, io_uring_enter),
        _SockFilter(_BPF_RET_K, 0, 0, _SECCOMP_RET_ERRNO | errno.EPERM),
        _SockFilter(_BPF_RET_K, 0, 0, _SECCOMP_RET_ALLOW),
    )
    program = _SockFprog(len(instructions), instructions)
    if _LIBC.prctl(_PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "prctl(PR_SET_NO_NEW_PRIVS)")
    listener = _LIBC.syscall(
        seccomp_syscall,
        _SECCOMP_SET_MODE_FILTER,
        _SECCOMP_FILTER_FLAG_NEW_LISTENER,
        ctypes.byref(program),
    )
    if listener < 0:
        raise OSError(ctypes.get_errno(), "seccomp(NEW_LISTENER)")
    return int(listener)


def process_connections_supported() -> bool:
    """Whether this substrate can install and broker seccomp notifications."""
    global _supported
    if _supported is not None:
        return _supported
    if sys.platform != "linux" or not hasattr(os, "pidfd_open"):
        _supported = False
        return False
    probe = subprocess.run(
        [sys.executable, "-m", __name__, "--probe"],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=10,
    )
    _supported = probe.returncode == 0
    return _supported


def _send_fd(channel: socket.socket, descriptor: int) -> None:
    descriptors = array.array("i", [descriptor])
    channel.sendmsg([b"R"], [(socket.SOL_SOCKET, socket.SCM_RIGHTS, descriptors)])


def _receive_fd(channel: socket.socket) -> int:
    _, ancillary, _, _ = channel.recvmsg(1, socket.CMSG_SPACE(array.array("i").itemsize))
    for level, kind, data in ancillary:
        if level == socket.SOL_SOCKET and kind == socket.SCM_RIGHTS:
            descriptors = array.array("i")
            descriptors.frombytes(data[: descriptors.itemsize])
            return descriptors[0]
    raise RuntimeError("guard process did not send its seccomp listener")


def _tgid(pid: int) -> int:
    with Path(f"/proc/{pid}/status").open(encoding="ascii") as status:
        for line in status:
            if line.startswith("Tgid:"):
                return int(line.split()[1])
    raise RuntimeError(f"process {pid} has no Tgid")


def _read_process(pid: int, address: int, length: int) -> bytes:
    descriptor = os.open(f"/proc/{pid}/mem", os.O_RDONLY)
    try:
        data = os.pread(descriptor, length, address)
    finally:
        os.close(descriptor)
    if len(data) != length:
        raise OSError(errno.EFAULT, "short process memory read")
    return data


def _destination(raw: bytes) -> tuple[str, int] | None:
    if len(raw) < 2:
        return None
    family = struct.unpack_from("H", raw)[0]
    if family == socket.AF_INET and len(raw) >= 8:
        return socket.inet_ntop(socket.AF_INET, raw[4:8]), struct.unpack_from("!H", raw, 2)[0]
    if family == socket.AF_INET6 and len(raw) >= 24:
        address = ipaddress.IPv6Address(raw[8:24])
        host = str(address.ipv4_mapped or address)
        return host, struct.unpack_from("!H", raw, 2)[0]
    return None


def _emulate_connect(tgid: int, descriptor: int, address: bytes) -> int:
    pidfd_open = getattr(os, "pidfd_open", None)
    if pidfd_open is None:
        return -errno.ENOSYS
    pidfd = int(pidfd_open(tgid))
    try:
        duplicate = _LIBC.syscall(_PIDFD_GETFD_SYSCALL, pidfd, descriptor, 0)
        if duplicate < 0:
            return -ctypes.get_errno()
    finally:
        os.close(pidfd)
    try:
        buffer = ctypes.create_string_buffer(address)
        if _LIBC.connect(duplicate, ctypes.byref(buffer), len(address)) == 0:
            return 0
        return -ctypes.get_errno()
    finally:
        os.close(duplicate)


class ProcessConnectionGuard:
    """Broker connects for one trusted process while constraining descendants."""

    def __init__(
        self,
        directory: Path,
        protected: Collection[tuple[str, int]],
        allowed: Collection[tuple[str, int]],
    ) -> None:
        self.directory = directory
        self.socket_path = directory / "control.sock"
        self.protected = frozenset(protected)
        self.allowed = frozenset(allowed)
        if not self.allowed <= self.protected:
            raise ValueError("allowed process connections must be protected destinations")
        self._server: socket.socket | None = None
        self._listener: int | None = None
        self._stop_read, self._stop_write = os.pipe()
        self._ready = threading.Event()
        self._error: BaseException | None = None
        self._thread: threading.Thread | None = None

    @property
    def sandbox_socket(self) -> str:
        return _SANDBOX_SOCKET

    def start(self) -> None:
        self.directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        self._server = socket.socket(socket.AF_UNIX)
        self._server.bind(str(self.socket_path))
        self._server.listen(1)
        self.socket_path.chmod(0o600)
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    async def wait_ready(self) -> None:
        ready = await asyncio.to_thread(self._ready.wait, _READY_TIMEOUT_SECONDS)
        if not ready:
            raise TimeoutError("process connection guard did not become ready")
        if self._error is not None:
            raise RuntimeError("process connection guard failed") from self._error

    def close(self) -> None:
        with contextlib.suppress(OSError):
            os.write(self._stop_write, b"x")
        if self._thread is not None:
            self._thread.join(timeout=2)
            if self._thread.is_alive():
                if self._server is not None:
                    self._server.close()
                if self._listener is not None:
                    os.close(self._listener)
                    self._listener = None
                self._thread.join(timeout=2)
            self._thread = None
        if self._server is not None:
            self._server.close()
            self._server = None
        if self._listener is not None:
            os.close(self._listener)
            self._listener = None
        for descriptor in (self._stop_read, self._stop_write):
            with contextlib.suppress(OSError):
                os.close(descriptor)
        with contextlib.suppress(FileNotFoundError):
            self.socket_path.unlink()
        with contextlib.suppress(OSError):
            self.directory.rmdir()

    def _serve(self) -> None:
        try:
            assert self._server is not None
            poller = select.poll()
            poller.register(self._server, select.POLLIN)
            poller.register(self._stop_read, select.POLLIN)
            ready = {descriptor for descriptor, _ in poller.poll()}
            if self._stop_read in ready:
                return
            channel, _ = self._server.accept()
            self._server.close()
            self._server = None
            with channel:
                self._listener = _receive_fd(channel)
            with contextlib.suppress(FileNotFoundError):
                self.socket_path.unlink()
            self._broker()
        except BaseException as exc:
            self._error = exc
            self._ready.set()

    def _broker(self) -> None:
        assert self._listener is not None
        poller = select.poll()
        poller.register(self._listener, select.POLLIN)
        poller.register(self._stop_read, select.POLLIN)
        trusted_tgid: int | None = None
        while True:
            ready = {descriptor for descriptor, _ in poller.poll()}
            if self._stop_read in ready:
                return
            notification = _SeccompNotif()
            try:
                fcntl.ioctl(self._listener, _SECCOMP_IOCTL_NOTIF_RECV, notification)
            except OSError as exc:
                if exc.errno in {errno.EINTR, errno.ENOENT}:
                    continue
                raise
            response = _SeccompNotifResp(id=notification.id)
            try:
                process_tgid = _tgid(notification.pid)
                if trusted_tgid is None:
                    trusted_tgid = process_tgid
                    response.error = -errno.ECONNREFUSED
                    self._ready.set()
                else:
                    address = _read_process(
                        notification.pid,
                        notification.data.args[1],
                        notification.data.args[2],
                    )
                    target = _destination(address)
                    if target in self.protected and (
                        process_tgid != trusted_tgid or target not in self.allowed
                    ):
                        response.error = -errno.EPERM
                    else:
                        result = _emulate_connect(
                            process_tgid,
                            notification.data.args[0],
                            address,
                        )
                        response.error = result if result < 0 else 0
                        response.val = result if result >= 0 else 0
            except (OSError, RuntimeError):
                response.error = -errno.EPERM
            try:
                fcntl.ioctl(self._listener, _SECCOMP_IOCTL_NOTIF_SEND, response)
            except OSError as exc:
                if exc.errno != errno.ENOENT:
                    raise


def guarded_exec(socket_path: str, argv: Sequence[str]) -> None:
    """Install the guard, register this process, and replace it with ``argv``."""
    if not argv:
        raise ValueError("guarded execution requires a command")
    channel = socket.socket(socket.AF_UNIX)
    channel.connect(socket_path)
    listener = _install_connect_listener()
    try:
        _send_fd(channel, listener)
    finally:
        os.close(listener)
        channel.close()
    registration = socket.socket(socket.AF_UNIX)
    try:
        registration.connect(_REGISTER_ADDRESS)
    except ConnectionRefusedError:
        pass
    finally:
        registration.close()
    os.execvp(argv[0], list(argv))  # noqa: S606 - exact controller-built argv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("socket", nargs="?")
    parser.add_argument("argv", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.probe:
        listener = _install_connect_listener()
        os.close(listener)
        return
    if args.socket is None:
        parser.error("socket is required")
    argv = args.argv[1:] if args.argv[:1] == ["--"] else args.argv
    guarded_exec(args.socket, argv)


if __name__ == "__main__":
    main()

__all__ = ["ProcessConnectionGuard", "guarded_exec", "process_connections_supported"]
