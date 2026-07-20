"""Workspace: a directory + bwrap-isolated SSH server."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import shutil
import socket
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import asyncssh

from hud.utils.process import create_process_group_exec

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from hud.capabilities import Capability

    from .file_tracker import FileTracker

LOGGER = logging.getLogger("hud.environment.workspace")

# Set once the first Workspace logs the missing-bwrap notice (avoid per-instance spam).
_warned_no_bwrap = False


def _is_root() -> bool:
    return sys.platform != "win32" and hasattr(os, "geteuid") and os.geteuid() == 0


# ─────────────────────────── mount declarations ───────────────────────────


MountKind = Literal["ro", "rw", "tmpfs", "symlink", "proc", "dev"]

# kind -> (normal flag, optional modifier, takes source)
_MOUNT_FLAGS: dict[MountKind, tuple[str, str | None, bool]] = {
    "ro": ("--ro-bind", "--ro-bind-try", True),
    "rw": ("--bind", "--bind-try", True),
    "symlink": ("--symlink", None, True),
    "tmpfs": ("--tmpfs", None, False),
    "proc": ("--proc", None, False),
    "dev": ("--dev", None, False),
}


@dataclass(slots=True, frozen=True)
class Mount:
    """One bwrap mount entry: ``Mount(kind, src=..., dst=..., optional=...)``."""

    kind: MountKind
    src: str = ""
    dst: str = ""
    optional: bool = False

    def to_bwrap_args(self) -> list[str]:
        normal, optional_flag, takes_src = _MOUNT_FLAGS[self.kind]
        flag = optional_flag if (self.optional and optional_flag) else normal
        return [flag, self.src, self.dst] if takes_src else [flag, self.dst]


# Most slim Linux distros merge ``/lib`` into ``/usr/lib`` via symlinks;
# we mirror that inside the namespace.
DEFAULT_SYSTEM_MOUNTS: tuple[Mount, ...] = (
    Mount("ro", src="/usr", dst="/usr"),
    Mount("ro", src="/etc", dst="/etc"),
    Mount("symlink", src="usr/lib", dst="/lib"),
    Mount("symlink", src="usr/lib64", dst="/lib64"),
    Mount("symlink", src="usr/bin", dst="/bin"),
    Mount("symlink", src="usr/sbin", dst="/sbin"),
    Mount("proc", dst="/proc"),
    Mount("dev", dst="/dev"),
    Mount("tmpfs", dst="/tmp"),  # noqa: S108 — namespace-local tmpfs, not a host tempdir
)


# ─────────────────────────── the workspace ───────────────────────────


_DEFAULT_USER = "agent"


class Workspace:
    """Directory + bwrap-isolated SSH.

    The standard shell daemon: ``env.workspace(root)`` attaches one to an
    :class:`~hud.environment.Environment`, which starts it and publishes its
    concrete ``ssh/2`` capability when the env serves. Construction is pure
    data — keys, sockets, and the root directory materialize only at serve
    time. Drive it directly (``start()`` / :meth:`capability` / ``stop()``)
    to publish the capability yourself.

    ``shell_uid`` drops agent sessions to that uid with ``setpriv`` when the
    serving process is root — the privilege wall for substrates where bwrap
    is unavailable and the env process holds secrets the agent must not read.
    No-op off root.
    """

    def __init__(
        self,
        root: Path | str,
        *,
        # bwrap configuration
        mounts: Sequence[Mount] = (),
        network: bool = False,
        env: Mapping[str, str] | None = None,
        system_mounts: Sequence[Mount] | None = None,
        guest_path: str = "/workspace",
        # ssh server configuration
        host: str = "127.0.0.1",
        port: int = 0,
        user: str = _DEFAULT_USER,
        host_key_path: Path | None = None,
        authorized_client_keys: list[Path] | None = None,
        track_files: bool = False,
        shell_uid: int | None = None,
    ) -> None:
        self.root: Path = Path(root).resolve()
        # Per-instance credential dir, materialized lazily (see _credentials_dir).
        self._cred_dir: Path | None = None

        # Path the root is mounted at inside the sandbox (and the default cwd).
        # Defaults to /workspace; set to the root's real path for callers that
        # need in-/out-of-sandbox paths to match (e.g. Harbor challenge dirs).
        self._guest_path = guest_path

        # bwrap state
        self.mounts: tuple[Mount, ...] = tuple(mounts)
        self.network = network
        self.env: dict[str, str] = dict(env or {})
        self._system_mounts: tuple[Mount, ...] = tuple(
            system_mounts if system_mounts is not None else DEFAULT_SYSTEM_MOUNTS,
        )
        self._bwrap = shutil.which("bwrap")
        # Without bwrap there is no `/workspace` mount — the sandbox *is* the real
        # directory, so address it by its real path. Otherwise `cd /workspace`
        # lands in a phantom dir and the editor/bash disagree on where files are.
        # Only override the default; respect an explicit guest_path.
        if self._bwrap is None and guest_path == "/workspace":
            self._guest_path = self.root.as_posix()
        # ssh config
        self._ssh_host = host
        self._ssh_port = port
        self._ssh_user = user
        self._shell_uid = shell_uid
        self._ssh_host_key_path = host_key_path
        self._ssh_authorized_client_keys = list(authorized_client_keys or [])
        self._acceptor: asyncssh.SSHAcceptor | None = None
        self._serve_task: asyncio.Task[None] | None = None
        self._client_key_path: Path | None = None
        self._host_key: asyncssh.SSHKey | None = None
        self._host_pubkey_str: str | None = None
        self._authorized_keys_path: Path | None = None
        self._sock: socket.socket | None = None
        self._bound_host: str | None = None
        self._bound_port: int | None = None
        # File tracking: an observation-only filetracking/1 server over the same
        # root. Materialized at start() when enabled.
        self._track_files = track_files
        self._file_tracker: FileTracker | None = None
        self._ft_server: asyncio.Server | None = None
        self._ft_host: str | None = None
        self._ft_port: int | None = None

    def _setpriv(self) -> str | None:
        """Absolute path to ``setpriv``, resolved via the *server's* PATH.

        Sessions must exec it by absolute path: session env can carry an
        agent-writable PATH, and a bare name resolved through it would run
        an agent-planted binary as root before the drop happens.
        """
        return shutil.which("setpriv")

    def _drops_privileges(self) -> bool:
        """Whether sessions are dropped to ``shell_uid`` on this host.

        Only when serving as root on Linux with ``setpriv`` present —
        ``setpriv`` is a util-linux command and the drop is meaningless off
        root. Off root the option is a documented no-op.
        """
        return (
            self._shell_uid is not None
            and _is_root()
            and sys.platform == "linux"
            and self._setpriv() is not None
        )

    def _prepare_runtime(self) -> None:
        """Materialize filesystem credentials and bind the SSH socket."""
        if self._sock is not None:
            return
        if self._shell_uid is not None and _is_root() and not self._drops_privileges():
            # Fail closed: serving as root while unable to drop would run every
            # agent shell as root, exactly what shell_uid exists to prevent.
            raise RuntimeError(
                "shell_uid is set and the server is root, but privileges cannot be dropped "
                "(setpriv is required on Linux). Refusing to serve agent shells as root."
            )
        if self._bwrap is None and sys.platform != "win32":
            # Once per process: repeating this for every Workspace is noise, and
            # on macOS (no bubblewrap exists) it is an expected state.
            global _warned_no_bwrap
            if not _warned_no_bwrap:
                _warned_no_bwrap = True
                log = LOGGER.warning if sys.platform == "linux" else LOGGER.info
                log(
                    "bwrap not on PATH; SSH sessions will run WITHOUT isolation. "
                    "Install bubblewrap, or run inside a Linux container that has it.",
                )
        self.root.mkdir(parents=True, exist_ok=True)
        if self._drops_privileges():
            # The workspace is the agent's surface: hand the whole tree to the
            # dropped uid, or contents baked in as root (e.g. a Docker COPY)
            # stay un-editable from the dropped shell. lchown so an in-tree
            # symlink can't redirect the chown outside the workspace.
            assert self._shell_uid is not None
            uid = self._shell_uid
            with contextlib.suppress(OSError):
                os.lchown(self.root, uid, uid)
            for dirpath, dirnames, filenames in os.walk(self.root):
                for entry in (*dirnames, *filenames):
                    with contextlib.suppress(OSError):
                        os.lchown(os.path.join(dirpath, entry), uid, uid)
        self._host_key, self._host_pubkey_str = self._load_or_generate_host_key()
        self._authorized_keys_path = self._ensure_authorized_keys_file()
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind((self._ssh_host, self._ssh_port))
        self._sock.listen(128)
        self._bound_host, self._bound_port = self._sock.getsockname()[:2]
        LOGGER.info(
            "Workspace SSH bound on %s as user %r (client key: %s)",
            self.ssh_url,
            self._ssh_user,
            self._client_key_path,
        )

    # ─── lifecycle ────────────────────────────────────────────────────

    async def _serve(self) -> None:
        """Run the asyncssh accept loop on the pre-bound socket."""
        self._prepare_runtime()
        assert self._sock is not None
        assert self._host_key is not None
        assert self._authorized_keys_path is not None
        self._acceptor = await asyncssh.listen(
            sock=self._sock,
            server_host_keys=[self._host_key],
            authorized_client_keys=str(self._authorized_keys_path),
            process_factory=self._handle_process,
            line_editor=False,
            keepalive_interval=30,
            encoding=None,
        )

    async def start(self) -> None:
        """Ensure the SSH accept loop is running. Idempotent.

        The first start prepares credentials and binds the socket, then ensures
        the async acceptor exists.
        """
        self._prepare_runtime()
        if self._serve_task is None and self._acceptor is None:
            self._serve_task = asyncio.get_event_loop().create_task(self._serve())
        # Yield so the acceptor binds before first use.
        await asyncio.sleep(0)
        if self._track_files and self._ft_server is None:
            await self._start_file_tracking()

    async def _start_file_tracking(self) -> None:
        """Take the baseline snapshot and bind the filetracking/1 server."""
        from .file_tracker import FileTracker, serve_file_tracking

        tracker = FileTracker(self.root)
        # The baseline walk is CPU-bound; keep it off the event loop.
        await asyncio.get_running_loop().run_in_executor(None, tracker.take_baseline)
        self._file_tracker = tracker
        self._ft_server = await serve_file_tracking(tracker, host=self._ssh_host)
        self._ft_host, self._ft_port = self._ft_server.sockets[0].getsockname()[:2]

    async def stop(self) -> None:
        """Stop accepting SSH sessions and release the socket.

        Credentials stay on disk; a later :meth:`start` re-binds (fresh port
        unless one was pinned) and reuses them.
        """
        if self._ft_server is not None:
            self._ft_server.close()
            with contextlib.suppress(Exception):
                await asyncio.wait_for(self._ft_server.wait_closed(), 5.0)
            self._ft_server = None
            self._ft_host = self._ft_port = None
            self._file_tracker = None
        if self._serve_task is not None:
            self._serve_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._serve_task
            self._serve_task = None
        if self._acceptor is not None:
            self._acceptor.close()
            # close() initiates shutdown; wait_closed() can hang on Windows when a
            # client connection lingers, so bound it rather than block teardown.
            with contextlib.suppress(Exception):
                await asyncio.wait_for(self._acceptor.wait_closed(), 5.0)
            self._acceptor = None
        elif self._sock is not None:
            self._sock.close()
        self._sock = None
        self._bound_host = None
        self._bound_port = None
        if self._cred_dir is not None:
            shutil.rmtree(self._cred_dir, ignore_errors=True)
            self._cred_dir = None

    # ─── ssh accessors / capability ───────────────────────────────────

    @property
    def ssh_url(self) -> str:
        """``ssh://host:port`` — prepared lazily on first access."""
        self._prepare_runtime()
        assert self._bound_host is not None
        assert self._bound_port is not None
        return f"ssh://{self._bound_host}:{self._bound_port}"

    @property
    def ssh_host_pubkey(self) -> str:
        """OpenSSH-format public host key (for harness ``known_hosts``)."""
        self._prepare_runtime()
        assert self._host_pubkey_str is not None
        return self._host_pubkey_str

    @property
    def ssh_client_key_path(self) -> Path | None:
        """Ephemeral client private key path (None if external keys supplied)."""
        self._prepare_runtime()
        return self._client_key_path

    @property
    def ssh_user(self) -> str:
        """SSH username."""
        return self._ssh_user

    def capability(self, name: str = "shell") -> Capability:
        """The concrete ``ssh`` capability — materializes keys + bind.

        Carries the managed client key's *content*, so the binding
        authenticates from anywhere the daemon is reachable — including a
        client on the other side of a container boundary.
        """
        from hud.capabilities import Capability

        key_path = self.ssh_client_key_path
        return Capability.ssh(
            name=name,
            url=self.ssh_url,
            user=self.ssh_user,
            host_pubkey=self.ssh_host_pubkey,
            client_key=key_path.read_text() if key_path else None,
            client_key_path=key_path,
            cwd=self._guest_path,
        )

    @property
    def tracks_files(self) -> bool:
        """Whether this workspace serves a ``filetracking/1`` capability."""
        return self._track_files

    def file_tracking_capability(self, name: str = "filetracking") -> Capability:
        """The concrete ``filetracking/1`` capability (requires ``track_files=True``)."""
        from hud.capabilities import Capability

        if self._ft_host is None or self._ft_port is None:
            raise RuntimeError("file tracking not started; call start() with track_files=True")
        return Capability(
            name=name,
            protocol="filetracking/1",
            url=f"tcp://{self._ft_host}:{self._ft_port}",
        )

    # ─── argv builders (public — useful if you want your own subprocess) ──

    @property
    def bwrap_available(self) -> bool:
        return self._bwrap is not None

    def bwrap_argv(
        self,
        command: list[str] | str,
        *,
        cwd: str | None = None,
        env: Mapping[str, str] | None = None,
        inherit_host_env: bool = True,
    ) -> list[str]:
        """Argv that runs ``command`` inside bwrap. Raises if bwrap unavailable.

        bwrap ``--clearenv`` then re-injects ``full_env`` via ``--setenv``, so
        with ``inherit_host_env=False`` the host environment (server secrets)
        is left out and only ``self.env`` + ``env`` reach the sandbox.
        """
        if self._bwrap is None:
            raise RuntimeError("bwrap not available on this host")
        target_cwd = cwd if cwd is not None else self._guest_path
        base_env = dict(os.environ) if inherit_host_env else {}
        full_env = {**base_env, **self.env, **(env or {})}
        argv: list[str] = [
            self._bwrap,
            "--die-with-parent",
            "--unshare-user-try",
            "--unshare-pid",
            "--unshare-ipc",
            "--unshare-uts",
            "--unshare-cgroup-try",
        ]
        if not self.network:
            argv.append("--unshare-net")
        for m in self._system_mounts:
            argv.extend(m.to_bwrap_args())
        argv.extend(["--bind", str(self.root), self._guest_path])
        for m in self.mounts:
            argv.extend(m.to_bwrap_args())
        argv.extend(["--chdir", target_cwd])
        argv.append("--clearenv")
        for k, v in full_env.items():
            argv.extend(["--setenv", k, v])
        argv.append("--")
        if isinstance(command, str):
            argv.extend(["bash", "-lc", command])
        else:
            argv.extend(command)
        return argv

    def shell_argv(
        self,
        command: str | None = None,
        *,
        cwd: str | None = None,
        env: Mapping[str, str] | None = None,
    ) -> list[str]:
        """Per-session shell argv (bwrap'd if available, else host shell).

        With ``shell_uid`` set and the serving process running as root, the
        whole session is wrapped in ``setpriv`` to drop to that uid.
        """
        if sys.platform == "win32":
            if command is not None:
                return ["cmd.exe", "/c", command]
            return ["cmd.exe"]
        if self._bwrap is not None:
            inner: list[str] | str = ["bash", "-lc", command] if command else ["bash", "-l"]
            if self._drops_privileges():
                # Don't let bwrap re-inject host secrets via --setenv; feed it
                # the same minimal environment as the non-bwrap dropped shell,
                # keeping explicit per-call overrides.
                walled_env = {**(self._session_env() or {}), **(env or {})}
                argv = self.bwrap_argv(inner, cwd=cwd, env=walled_env, inherit_host_env=False)
            else:
                argv = self.bwrap_argv(inner, cwd=cwd, env=env)
        elif command is not None:
            argv = ["bash", "-lc", command]
        else:
            argv = ["bash", "-l"]
        if self._drops_privileges():
            setpriv = self._setpriv()
            assert setpriv is not None  # guaranteed by _drops_privileges
            uid = str(self._shell_uid)
            argv = [setpriv, "--reuid", uid, "--regid", uid, "--clear-groups", "--", *argv]
        return argv

    # ─── ssh server internals ─────────────────────────────────────────

    def _credentials_dir(self) -> Path:
        """Key material lives outside the served root: the root is the agent's
        shell cwd and diff-tracking surface, so secrets don't belong in it.

        ``mkdtemp`` creates a fresh 0700 directory with an unpredictable name
        atomically, so a local user can't pre-place a symlink at the path to
        redirect the private keys the server writes here.
        """
        if self._cred_dir is None:
            self._cred_dir = Path(tempfile.mkdtemp(prefix="hud-workspace-creds-"))
        return self._cred_dir

    def _load_or_generate_host_key(self) -> tuple[asyncssh.SSHKey, str]:
        if self._ssh_host_key_path is not None:
            key = asyncssh.read_private_key(self._ssh_host_key_path)
        else:
            key_path = self._credentials_dir() / "host_ed25519"
            if key_path.exists():
                key = asyncssh.read_private_key(key_path)
            else:
                key = asyncssh.generate_private_key("ssh-ed25519")
                key.write_private_key(str(key_path))
                key.write_public_key(str(key_path.with_suffix(".pub")))
        return key, key.export_public_key().decode("ascii").strip()

    def _ensure_authorized_keys_file(self) -> Path:
        """Write the authorized_keys file asyncssh wants on disk."""
        creds = self._credentials_dir()
        auth_path = creds / "authorized_keys"
        pub_lines: list[str] = []

        if self._ssh_authorized_client_keys:
            pub_lines.extend(Path(p).read_text().strip() for p in self._ssh_authorized_client_keys)
        else:
            priv_path = creds / "client_ed25519"
            pub_path = priv_path.with_suffix(".pub")
            if not (priv_path.exists() and pub_path.exists()):
                client = asyncssh.generate_private_key("ssh-ed25519")
                client.write_private_key(str(priv_path))
                client.write_public_key(str(pub_path))
            pub_lines.append(pub_path.read_text().strip())
            self._client_key_path = priv_path

        auth_path.write_text("\n".join(pub_lines) + "\n", encoding="ascii")
        return auth_path

    def _session_env(self) -> dict[str, str] | None:
        """Environment for a shell session (non-bwrap path).

        When dropping privileges, the child would otherwise inherit the
        server's full environment — including any secrets the env process
        holds (the reason ``shell_uid`` exists) — so build a minimal, safe
        environment from scratch. Otherwise preserve the inherited-env
        behavior, layering ``self.env`` overrides.
        """
        if self._drops_privileges():
            base = {
                "PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
                # The server's HOME (/root) is unreadable by the dropped uid;
                # the workspace is the one directory guaranteed writable.
                "HOME": self._guest_path,
                "TERM": os.environ.get("TERM", "xterm"),
            }
            return {**base, **self.env}
        return {**os.environ, **self.env} if self.env else None

    async def _handle_process(self, process: asyncssh.SSHServerProcess[bytes]) -> None:
        argv = self.shell_argv(process.command)
        proc_env = self._session_env()

        if sys.platform == "win32":
            # On Windows, asyncio.create_subprocess_exec uses the ProactorEventLoop's
            # IOCP machinery for process-exit notification.  When the IOCP event fires
            # after the subprocess coroutine has already returned (a race that can
            # happen even when communicate() calls wait() internally), it corrupts
            # asyncssh's IOCP state and permanently breaks the SSH session.
            # Running subprocess.run() in a thread-pool executor sidesteps IOCP
            # entirely: the blocking WaitForSingleObject in the worker thread drains
            # the process exit before the Future resolves, leaving no pending events.
            #
            # Also: shell_argv() used to wrap the SSH command in ["cmd.exe", "/c",
            # command], but Python's list2cmdline would requote that, leaving a
            # trailing '"' on the last token. Fixed by splitting process.command
            # directly with shlex.split so list2cmdline never adds an extra layer.
            # Additionally, cmd.exe launched via CreateProcess does NOT search the
            # CWD for batch files (only PATH), so relative .bat paths are resolved
            # to absolute below.
            import functools
            import shlex
            import subprocess as _subprocess

            if process.command:
                try:
                    win_argv: list[str] = shlex.split(process.command, posix=False)
                except ValueError:
                    win_argv = ["cmd.exe", "/c", process.command]
                # cmd.exe launched via CreateProcess/subprocess does NOT search
                # the CWD for batch files — only directories on PATH. Resolve
                # relative .bat paths to absolute so cmd.exe finds them.
                if win_argv and win_argv[0].lower() in ("cmd", "cmd.exe"):
                    win_argv = [
                        str(self.root / arg)
                        if (arg.lower().endswith(".bat") and not os.path.isabs(arg))
                        else arg
                        for arg in win_argv
                    ]
            else:
                win_argv = ["cmd.exe"]

            try:
                loop = asyncio.get_running_loop()
                result = await asyncio.wait_for(
                    loop.run_in_executor(
                        None,
                        functools.partial(
                            _subprocess.run,
                            win_argv,
                            stdin=_subprocess.DEVNULL,
                            stdout=_subprocess.PIPE,
                            stderr=_subprocess.PIPE,
                            cwd=str(self.root),
                            env=proc_env,
                            timeout=3600,
                        ),
                    ),
                    timeout=3660.0,
                )
            except FileNotFoundError as exc:
                process.stderr.write(f"workspace: cannot spawn shell: {exc}\n".encode())
                process.exit(127)
                return
            except (TimeoutError, _subprocess.TimeoutExpired):
                process.stderr.write(b"workspace: command timed out after 3600s\n")
                process.exit(1)
                return

            if result.stdout:
                process.stdout.write(result.stdout)
            if result.stderr:
                process.stderr.write(result.stderr)
            process.exit(result.returncode)
            return

        try:
            sub = await create_process_group_exec(
                *argv,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(self.root),
                env=proc_env,
            )
        except FileNotFoundError as exc:
            process.stderr.write(f"workspace: cannot spawn shell: {exc}\n".encode())
            process.exit(127)
            return

        stdin = sub.process.stdin
        stdout = sub.stdout
        stderr = sub.stderr
        assert stdin is not None
        assert stdout is not None
        assert stderr is not None

        async def relay_stdin() -> None:
            try:
                while chunk := await process.stdin.read(65536):
                    stdin.write(chunk)
                    await stdin.drain()
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                stdin.close()

        stdin_task = asyncio.create_task(relay_stdin())
        stdout_task = asyncio.create_task(stdout.read())
        stderr_task = asyncio.create_task(stderr.read())
        try:
            async with asyncio.timeout(3600.0):
                _, stdout_data, stderr_data = await asyncio.gather(
                    sub.wait(),
                    stdout_task,
                    stderr_task,
                )
            await sub.terminate()
            stdin_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await stdin_task
        except TimeoutError:
            await sub.terminate()
            stdin_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await stdin_task
            process.stderr.write(b"workspace: command timed out after 3600s\n")
            process.exit(1)
            return
        except BaseException:
            await sub.terminate()
            stdin_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await stdin_task
            raise

        if stdout_data:
            process.stdout.write(stdout_data)
        if stderr_data:
            process.stderr.write(stderr_data)
        process.exit(sub.returncode if sub.returncode is not None else 0)


__all__ = [
    "DEFAULT_SYSTEM_MOUNTS",
    "Mount",
    "MountKind",
    "Workspace",
]
