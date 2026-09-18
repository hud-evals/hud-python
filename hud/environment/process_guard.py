"""Linux process-bound network connection enforcement."""

# ruff: noqa: UP045

from __future__ import annotations

import argparse
import array
import asyncio
import contextlib
import ctypes
import errno
import fcntl
import ipaddress
import json
import os
import platform
import select
import shutil
import signal
import socket
import struct
import subprocess
import sys
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Literal, NoReturn, Optional, cast

if TYPE_CHECKING:
    from collections.abc import Collection, Sequence

_PR_SET_NO_NEW_PRIVS = 38
_SECCOMP_SET_MODE_FILTER = 1
_SECCOMP_FILTER_FLAG_NEW_LISTENER = 8
_SECCOMP_RET_KILL_PROCESS = 0x80000000
_SECCOMP_RET_USER_NOTIF = 0x7FC00000
_SECCOMP_RET_TRACE = 0x7FF00000
_SECCOMP_RET_ERRNO = 0x00050000
_SECCOMP_RET_ALLOW = 0x7FFF0000

_BPF_LD_W_ABS = 0x20
_BPF_JMP_JEQ_K = 0x15
_BPF_RET_K = 0x06

_SECCOMP_IOCTL_NOTIF_RECV = 0xC0502100
_SECCOMP_IOCTL_NOTIF_SEND = 0xC0182101
_PIDFD_OPEN_SYSCALL = 434
_PIDFD_GETFD_SYSCALL = 438
_REGISTER_ADDRESS = b"\0hud-process-connection-register"
_SANDBOX_SOCKET = "/tmp/.hud-process-connection/control.sock"  # noqa: S108
_SANDBOX_HELPER = "/tmp/.hud-process-connection/process_guard.py"  # noqa: S108
_READY_TIMEOUT_SECONDS = 10.0

_PTRACE_TRACEME = 0
_PTRACE_CONT = 7
_PTRACE_GETREGS = 12
_PTRACE_SETREGS = 13
_PTRACE_SETOPTIONS = 0x4200
_PTRACE_O_TRACEFORK = 0x00000002
_PTRACE_O_TRACEVFORK = 0x00000004
_PTRACE_O_TRACECLONE = 0x00000008
_PTRACE_O_TRACEEXEC = 0x00000010
_PTRACE_O_TRACESECCOMP = 0x00000080
_PTRACE_O_EXITKILL = 0x00100000
_PTRACE_EVENT_SECCOMP = 7
_WAIT_ALL = getattr(os, "WALL", 0x40000000)
_MAX_SOCKADDR_BYTES = 128

_LIBC = ctypes.CDLL(None, use_errno=True)
_LIBC.ptrace.restype = ctypes.c_long
GuardBackend = Literal["notify", "ptrace"]
_backend: Optional[GuardBackend] = None
_backend_probed = False


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


class _UserRegsStruct(ctypes.Structure):
    _fields_ = [
        ("r15", ctypes.c_ulonglong),
        ("r14", ctypes.c_ulonglong),
        ("r13", ctypes.c_ulonglong),
        ("r12", ctypes.c_ulonglong),
        ("rbp", ctypes.c_ulonglong),
        ("rbx", ctypes.c_ulonglong),
        ("r11", ctypes.c_ulonglong),
        ("r10", ctypes.c_ulonglong),
        ("r9", ctypes.c_ulonglong),
        ("r8", ctypes.c_ulonglong),
        ("rax", ctypes.c_ulonglong),
        ("rcx", ctypes.c_ulonglong),
        ("rdx", ctypes.c_ulonglong),
        ("rsi", ctypes.c_ulonglong),
        ("rdi", ctypes.c_ulonglong),
        ("orig_rax", ctypes.c_ulonglong),
        ("rip", ctypes.c_ulonglong),
        ("cs", ctypes.c_ulonglong),
        ("eflags", ctypes.c_ulonglong),
        ("rsp", ctypes.c_ulonglong),
        ("ss", ctypes.c_ulonglong),
        ("fs_base", ctypes.c_ulonglong),
        ("gs_base", ctypes.c_ulonglong),
        ("ds", ctypes.c_ulonglong),
        ("es", ctypes.c_ulonglong),
        ("fs", ctypes.c_ulonglong),
        ("gs", ctypes.c_ulonglong),
    ]


def _architecture() -> tuple[int, int, int, int, int]:
    machine = platform.machine()
    if machine == "x86_64":
        return 0xC000003E, 317, 42, 425, 426
    if machine in {"aarch64", "arm64"}:
        return 0xC00000B7, 277, 203, 425, 426
    raise RuntimeError(f"process-bound connections do not support {machine!r}")


def _connect_filter(connect_action: int) -> tuple[_SockFprog, object]:
    audit_arch, _, connect_syscall, io_uring_setup, io_uring_enter = _architecture()
    instructions = (_SockFilter * 11)(
        _SockFilter(_BPF_LD_W_ABS, 0, 0, 4),
        _SockFilter(_BPF_JMP_JEQ_K, 1, 0, audit_arch),
        _SockFilter(_BPF_RET_K, 0, 0, _SECCOMP_RET_KILL_PROCESS),
        _SockFilter(_BPF_LD_W_ABS, 0, 0, 0),
        _SockFilter(_BPF_JMP_JEQ_K, 0, 1, connect_syscall),
        _SockFilter(_BPF_RET_K, 0, 0, connect_action),
        _SockFilter(_BPF_JMP_JEQ_K, 0, 1, io_uring_setup),
        _SockFilter(_BPF_RET_K, 0, 0, _SECCOMP_RET_ERRNO | errno.EPERM),
        _SockFilter(_BPF_JMP_JEQ_K, 0, 1, io_uring_enter),
        _SockFilter(_BPF_RET_K, 0, 0, _SECCOMP_RET_ERRNO | errno.EPERM),
        _SockFilter(_BPF_RET_K, 0, 0, _SECCOMP_RET_ALLOW),
    )
    return _SockFprog(len(instructions), instructions), instructions


def _install_connect_listener() -> int:
    _, seccomp_syscall, _, _, _ = _architecture()
    program, instructions = _connect_filter(_SECCOMP_RET_USER_NOTIF)
    if _LIBC.prctl(_PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "prctl(PR_SET_NO_NEW_PRIVS)")
    listener = _LIBC.syscall(
        seccomp_syscall,
        _SECCOMP_SET_MODE_FILTER,
        _SECCOMP_FILTER_FLAG_NEW_LISTENER,
        ctypes.byref(program),
    )
    del instructions
    if listener < 0:
        raise OSError(ctypes.get_errno(), "seccomp(NEW_LISTENER)")
    return int(listener)


def _install_connect_trace() -> None:
    _, seccomp_syscall, _, _, _ = _architecture()
    program, instructions = _connect_filter(_SECCOMP_RET_TRACE)
    if _LIBC.prctl(_PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "prctl(PR_SET_NO_NEW_PRIVS)")
    result = _LIBC.syscall(
        seccomp_syscall,
        _SECCOMP_SET_MODE_FILTER,
        0,
        ctypes.byref(program),
    )
    del instructions
    if result != 0:
        raise OSError(ctypes.get_errno(), "seccomp(TRACE)")


def _detected_backend() -> Optional[GuardBackend]:
    global _backend, _backend_probed
    if _backend_probed:
        return _backend
    _backend_probed = True
    if sys.platform != "linux":
        return None
    try:
        probe = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--probe"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    candidate = probe.stdout.strip()
    if probe.returncode == 0 and candidate in {"notify", "ptrace"}:
        _backend = cast("GuardBackend", candidate)
    return _backend


def process_connections_supported() -> bool:
    """Whether this substrate has a race-free process connection guard."""
    return _detected_backend() is not None


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


def _destination(raw: bytes) -> Optional[tuple[str, int]]:
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
    pidfd = _LIBC.syscall(_PIDFD_OPEN_SYSCALL, tgid, 0)
    if pidfd < 0:
        return -ctypes.get_errno()
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


def _ptrace(request: int, pid: int, address: object = 0, data: object = 0) -> int:
    ctypes.set_errno(0)
    result = _LIBC.ptrace(request, pid, address, data)
    if result == -1:
        error = ctypes.get_errno()
        if error:
            raise OSError(error, f"ptrace({request})")
    return int(result)


def _trace_options(pid: int) -> None:
    options = (
        _PTRACE_O_TRACEFORK
        | _PTRACE_O_TRACEVFORK
        | _PTRACE_O_TRACECLONE
        | _PTRACE_O_TRACEEXEC
        | _PTRACE_O_TRACESECCOMP
        | _PTRACE_O_EXITKILL
    )
    _ptrace(_PTRACE_SETOPTIONS, pid, 0, options)


def _trace_registers(pid: int) -> _UserRegsStruct:
    registers = _UserRegsStruct()
    _ptrace(_PTRACE_GETREGS, pid, 0, ctypes.byref(registers))
    return registers


def _complete_traced_connect(
    pid: int,
    trusted_tgid: int,
    protected: Collection[tuple[str, int]],
    allowed: Collection[tuple[str, int]],
) -> None:
    registers = _trace_registers(pid)
    process_tgid = _tgid(pid)
    length = int(registers.rdx)
    if length < 0 or length > _MAX_SOCKADDR_BYTES:
        result = -errno.EFAULT
    else:
        try:
            address = _read_process(pid, int(registers.rsi), length)
            target = _destination(address)
            if target in protected and (process_tgid != trusted_tgid or target not in allowed):
                result = -errno.EPERM
            else:
                result = _emulate_connect(process_tgid, int(registers.rdi), address)
        except (OSError, RuntimeError):
            result = -errno.EPERM
    registers.orig_rax = ctypes.c_ulonglong(-1).value
    registers.rax = ctypes.c_ulonglong(result).value
    _ptrace(_PTRACE_SETREGS, pid, 0, ctypes.byref(registers))


def _trace_loop(
    original: int,
    protected: Collection[tuple[str, int]],
    allowed: Collection[tuple[str, int]],
) -> int:
    while True:
        try:
            pid, status = os.waitpid(-1, _WAIT_ALL)
        except InterruptedError:
            continue
        if os.WIFEXITED(status) or os.WIFSIGNALED(status):
            if pid == original:
                return status
            continue
        if not os.WIFSTOPPED(status):
            continue
        event = status >> 16
        stop_signal = os.WSTOPSIG(status)
        if event == _PTRACE_EVENT_SECCOMP:
            _complete_traced_connect(pid, original, protected, allowed)
            deliver = 0
        elif event or stop_signal in {signal.SIGSTOP, signal.SIGTRAP}:
            deliver = 0
        else:
            deliver = stop_signal
        try:
            _ptrace(_PTRACE_CONT, pid, 0, deliver)
        except OSError as exc:
            if exc.errno != errno.ESRCH:
                raise


def _forward_signals(original: int) -> None:
    def forward(signum: int, _frame: object) -> None:
        with contextlib.suppress(ProcessLookupError):
            os.kill(original, signum)

    for name in ("SIGHUP", "SIGINT", "SIGQUIT", "SIGTERM", "SIGTSTP", "SIGCONT", "SIGWINCH"):
        if signum := getattr(signal, name, None):
            signal.signal(signum, forward)


def _exit_from_wait_status(status: int) -> NoReturn:
    if os.WIFEXITED(status):
        raise SystemExit(os.WEXITSTATUS(status))
    signum = os.WTERMSIG(status)
    signal.signal(signum, signal.SIG_DFL)
    os.kill(os.getpid(), signum)
    raise RuntimeError("failed to reproduce traced process signal exit")


def _trace_exec(
    channel: socket.socket,
    protected: Collection[tuple[str, int]],
    allowed: Collection[tuple[str, int]],
    argv: Sequence[str],
) -> NoReturn:
    if platform.machine() != "x86_64":
        raise RuntimeError("ptrace process connections currently require x86_64")
    original = os.fork()
    if original == 0:
        channel.close()
        try:
            _ptrace(_PTRACE_TRACEME, 0)
            os.kill(os.getpid(), signal.SIGSTOP)
            _install_connect_trace()
            os.execvp(argv[0], list(argv))  # noqa: S606 - exact controller-built argv
        except BaseException:
            os._exit(127)
    try:
        _, status = os.waitpid(original, 0)
        if not os.WIFSTOPPED(status):
            raise RuntimeError("guarded process did not stop for ptrace")
        _trace_options(original)
        _forward_signals(original)
        channel.sendall(b"R")
        channel.close()
        _ptrace(_PTRACE_CONT, original)
        _exit_from_wait_status(_trace_loop(original, protected, allowed))
    except BaseException:
        with contextlib.suppress(ProcessLookupError):
            os.kill(original, signal.SIGKILL)
        with contextlib.suppress(ChildProcessError):
            os.waitpid(original, 0)
        raise


def _receive_policy(channel: socket.socket) -> tuple[frozenset[tuple[str, int]], ...]:
    raw = bytearray()
    while not raw.endswith(b"\n"):
        chunk = channel.recv(65536 - len(raw))
        if not chunk:
            raise RuntimeError("guard broker closed before sending its policy")
        raw.extend(chunk)
        if len(raw) >= 65536:
            raise RuntimeError("guard policy exceeded 64 KiB")
    document = json.loads(raw)
    protected = frozenset((str(host), int(port)) for host, port in document["protected"])
    allowed = frozenset((str(host), int(port)) for host, port in document["allowed"])
    if not allowed <= protected:
        raise RuntimeError("guard policy allowed an unprotected destination")
    return protected, allowed


def _probe_ptrace() -> bool:
    if platform.machine() != "x86_64":
        return False
    original = os.fork()
    if original == 0:
        try:
            _ptrace(_PTRACE_TRACEME, 0)
            os.kill(os.getpid(), signal.SIGSTOP)
            _install_connect_trace()
            descriptor = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                descriptor.connect(("127.0.0.1", 9))
            except OSError as exc:
                os._exit(0 if exc.errno == errno.EPERM else 1)
            os._exit(1)
        except BaseException:
            os._exit(1)
    try:
        _, status = os.waitpid(original, 0)
        if not os.WIFSTOPPED(status):
            return False
        _trace_options(original)
        _ptrace(_PTRACE_CONT, original)
        status = _trace_loop(original, {("127.0.0.1", 9)}, set())
        return os.WIFEXITED(status) and os.WEXITSTATUS(status) == 0
    except (OSError, RuntimeError):
        with contextlib.suppress(ProcessLookupError):
            os.kill(original, signal.SIGKILL)
        with contextlib.suppress(ChildProcessError):
            os.waitpid(original, 0)
        return False


class ProcessConnectionGuard:
    """Broker connects for one trusted process while constraining descendants."""

    def __init__(
        self,
        directory: Path,
        protected: Collection[tuple[str, int]],
        allowed: Collection[tuple[str, int]],
        *,
        backend: Optional[GuardBackend] = None,
    ) -> None:
        self.directory = directory
        self.socket_path = directory / "control.sock"
        self.helper_path = directory / "process_guard.py"
        self.protected = frozenset(protected)
        self.allowed = frozenset(allowed)
        if not self.allowed <= self.protected:
            raise ValueError("allowed process connections must be protected destinations")
        selected_backend = backend or _detected_backend()
        if selected_backend is None:
            raise RuntimeError("process connection guards are unavailable on this substrate")
        self.backend: GuardBackend = selected_backend
        self._server: Optional[socket.socket] = None
        self._listener: Optional[int] = None
        self._stop_read, self._stop_write = os.pipe()
        self._ready = threading.Event()
        self._error: Optional[BaseException] = None
        self._thread: Optional[threading.Thread] = None

    @property
    def sandbox_socket(self) -> str:
        return _SANDBOX_SOCKET

    @property
    def sandbox_helper(self) -> str:
        return _SANDBOX_HELPER

    def start(self) -> None:
        self.directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        shutil.copyfile(Path(__file__), self.helper_path)
        self.helper_path.chmod(0o500)
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
        with contextlib.suppress(FileNotFoundError):
            self.helper_path.unlink()
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
                if self.backend == "notify":
                    self._listener = _receive_fd(channel)
                else:
                    policy = {
                        "protected": sorted([host, port] for host, port in self.protected),
                        "allowed": sorted([host, port] for host, port in self.allowed),
                    }
                    channel.sendall(json.dumps(policy, separators=(",", ":")).encode() + b"\n")
                    if channel.recv(1) != b"R":
                        raise RuntimeError("ptrace guard did not acknowledge its policy")
            with contextlib.suppress(FileNotFoundError):
                self.socket_path.unlink()
            if self.backend == "notify":
                self._broker()
            else:
                self._ready.set()
        except BaseException as exc:
            self._error = exc
            self._ready.set()

    def _broker(self) -> None:
        assert self._listener is not None
        poller = select.poll()
        poller.register(self._listener, select.POLLIN)
        poller.register(self._stop_read, select.POLLIN)
        trusted_tgid: Optional[int] = None
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
                    address_length = int(notification.data.args[2])
                    if address_length < 0 or address_length > _MAX_SOCKADDR_BYTES:
                        raise OSError(errno.EFAULT, "invalid sockaddr length")
                    address = _read_process(
                        notification.pid,
                        notification.data.args[1],
                        address_length,
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


def guarded_exec(backend: GuardBackend, socket_path: str, argv: Sequence[str]) -> None:
    """Install the guard, register this process, and replace it with ``argv``."""
    if not argv:
        raise ValueError("guarded execution requires a command")
    channel = socket.socket(socket.AF_UNIX)
    channel.connect(socket_path)
    if backend == "ptrace":
        protected, allowed = _receive_policy(channel)
        _trace_exec(channel, protected, allowed, argv)
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
    parser.add_argument("--probe", choices=("auto", "notify", "ptrace"), nargs="?", const="auto")
    parser.add_argument("--backend", choices=("notify", "ptrace"))
    parser.add_argument("socket", nargs="?")
    parser.add_argument("argv", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.probe:
        if args.probe in {"auto", "notify"}:
            try:
                listener = _install_connect_listener()
            except OSError:
                if args.probe == "notify":
                    raise SystemExit(1) from None
            else:
                os.close(listener)
                sys.stdout.write("notify\n")
                return
        if args.probe in {"auto", "ptrace"} and _probe_ptrace():
            sys.stdout.write("ptrace\n")
            return
        raise SystemExit(1)
    if args.backend is None:
        parser.error("--backend is required")
    if args.socket is None:
        parser.error("socket is required")
    argv = args.argv[1:] if args.argv[:1] == ["--"] else args.argv
    guarded_exec(args.backend, args.socket, argv)


if __name__ == "__main__":
    main()

__all__ = ["ProcessConnectionGuard", "guarded_exec", "process_connections_supported"]
