"""Workspace: a directory + bwrap-isolated SSH server."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import shutil
import socket
import struct
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import asyncssh

from hud.utils.process import create_process_group_exec

if sys.platform != "win32":  # the pty a session runs on has no Windows analogue
    import fcntl
    import pty
    import termios

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from hud.capabilities import Capability

    from .file_tracker import FileTracker

LOGGER = logging.getLogger("hud.environment.workspace")

_COMMAND_TIMEOUT = 3600.0

# Set once the first Workspace logs the missing-bwrap notice (avoid per-instance spam).
#: bwrap's usability is a property of the host/container, so probe once per
#: process: an installed bwrap that cannot create namespaces (a container
#: whose seccomp profile blocks unprivileged userns) would otherwise fail
#: every session instead of falling back.
_bwrap_usable: bool | None = None


def usable_bwrap() -> str | None:
    """``bwrap``'s path if it can actually create namespaces here, else None."""
    global _bwrap_usable
    path = shutil.which("bwrap")
    if path is None:
        return None
    probe_binary = shutil.which("true")
    if probe_binary is None:
        return None
    if _bwrap_usable is None:
        try:
            probe = subprocess.run(
                # Mirrors a real session's *namespace* setup — mounting proc is
                # what an unprivileged container blocks first. The whole root
                # is bound so the probed binary keeps its loader; a narrower
                # mount set fails for the wrong reason. It does not prove the
                # installed bwrap parses every option a session passes, so
                # bwrap_argv stays within what bubblewrap 0.4 understands.
                [
                    path,
                    "--unshare-user-try",
                    "--unshare-pid",
                    "--ro-bind",
                    "/",
                    "/",
                    "--proc",
                    "/proc",
                    "--",
                    probe_binary,
                ],
                capture_output=True,
                timeout=15,
                check=False,
            )
            _bwrap_usable = probe.returncode == 0
            if not _bwrap_usable:
                LOGGER.warning(
                    "bwrap is installed but cannot create namespaces (%s); sessions will "
                    "run WITHOUT isolation. Allow unprivileged user namespaces (e.g. "
                    "docker --security-opt seccomp=unconfined) to enable it.",
                    probe.stderr.decode("utf-8", "replace").strip()[:120],
                )
        except (OSError, subprocess.SubprocessError):
            _bwrap_usable = False
    return path if _bwrap_usable else None


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

#: What the sandbox runs so it stays alive between sessions. It must outlast
#: every rollout without waking (a sandbox is discarded, never expired) and
#: come from the task's own image, since the serving venv is masked inside.
#: bwrap's reaper is pid 1 above it, so processes the agent orphans are reaped
#: rather than accumulating for the life of the sandbox.
#:
#: The line it prints first is the readiness signal, and it has to come from
#: in here: bwrap reports the child pid before that child has finished
#: building its mount namespace, so a session joining on the strength of the
#: pid alone can land in a root that is still half-assembled. The payload runs
#: only once setup is done, so its own output is the proof.
_SANDBOX_HOLDER = ["sh", "-c", "echo ready; exec sleep 2147483647"]
_SANDBOX_READY = b"ready\n"

#: What the sandbox's user namespace maps: the container's ids, unchanged.
_FULL_ID_RANGE = "0 0 65536"


def _without_harness_config(environ: Mapping[str, str]) -> dict[str, str]:
    """The serving process's environment, minus HUD's own configuration.

    A session runs in the *task's* environment, not the harness's. HUD's
    variables reaching it are a credential leak where they hold an API key,
    and a tell everywhere else: an agent that finds ``HUD_`` anything in its
    environment knows exactly what is running it. Variables the task or the
    caller declare are layered on afterwards and are unaffected — this drops
    only what the serving process happened to be configured with.
    """
    return {key: value for key, value in environ.items() if not key.startswith("HUD_")}


def _env_argv(env: Mapping[str, str]) -> list[str]:
    """``env -i`` and its assignments: an exact environment for what follows.

    Understood by every bubblewrap, unlike ``--clearenv``, which 0.4 lacks.
    """
    env_bin = shutil.which("env") or "/usr/bin/env"
    return [env_bin, "-i", *(f"{k}={v}" for k, v in env.items())]


async def install_identity_map(info_read: int, block_write: int) -> int:
    """Map ids into a bwrap held at ``--userns-block-fd``, and release it.

    The counterpart to spawning with ``--info-fd``/``--userns-block-fd``:
    bwrap reports the pid of the namespace it made and waits, this side says
    who its ids are, and only then does anything run in it. Returns that pid.
    """
    loop = asyncio.get_running_loop()
    raw = await asyncio.wait_for(loop.run_in_executor(None, os.read, info_read, 4096), 30.0)
    pid = int(json.loads(raw)["child-pid"]) if raw else 0
    if pid:
        _map_identities(pid)
    os.write(block_write, b"\n")
    return pid


def _map_identities(pid: int) -> None:
    """Give the sandbox the whole id space, not just the id that created it.

    Left to itself bwrap maps one id, so every file owned by anyone else is
    ``nobody`` inside — unreadable, unwritable, not even chownable by the
    sandbox's own root. That is most of an image whose task runs as a non-root
    user: the agent cannot write its own working directory, and no session can
    drop to an id the map does not contain. Writing the map from out here
    needs CAP_SETUID/CAP_SETGID in *this* namespace, which a container's root
    holds by default. Where the kernel refuses, fall back to the single id
    bwrap would have mapped: a narrow map is workable, an absent one is not.
    """
    proc = Path(f"/proc/{pid}")
    with contextlib.suppress(OSError):
        (proc / "setgroups").write_text("allow")
    for name, own in (("uid_map", os.geteuid()), ("gid_map", os.getegid())):
        try:
            (proc / name).write_text(_FULL_ID_RANGE)
        except OSError:
            if name == "gid_map":
                # Giving up supplementary groups is the kernel's price for
                # writing gid_map without the capability.
                with contextlib.suppress(OSError):
                    (proc / "setgroups").write_text("deny")
            try:
                (proc / name).write_text(f"{own} {own} 1")
            except OSError:
                LOGGER.warning("could not map ids into the sandbox (%s)", name)


def _open_pty(process: asyncssh.SSHServerProcess[bytes]) -> tuple[int, int]:
    """A terminal pair sized to what the client asked for: (master, slave)."""
    master_fd, slave_fd = pty.openpty()
    _set_winsize(slave_fd, *process.get_terminal_size())
    return master_fd, slave_fd


def _set_winsize(fd: int, width: int, height: int, pixwidth: int, pixheight: int) -> None:
    """Tell the terminal how big it is.

    Full-screen programs lay out against this and never re-measure, so a stale
    size leaves them drawing to the wrong shape for the rest of the session.
    """
    with contextlib.suppress(OSError):
        fcntl.ioctl(fd, termios.TIOCSWINSZ, struct.pack("HHHH", height, width, pixwidth, pixheight))


def _ctty_argv() -> list[str]:
    """Claim the session's terminal as its *controlling* terminal.

    tty file descriptors alone are not a terminal session: ``/dev/tty`` cannot
    be opened, and job control has no foreground process group to signal. Run
    inside the sandbox, ``setsid`` is not a process-group leader there and so
    execs in place rather than forking, which keeps the pid and exit status
    the caller is waiting on. Where the binary is absent (macOS ships none)
    the session still gets a working tty, only without a ctty.
    """
    setsid = shutil.which("setsid")
    return [setsid, "-c"] if setsid else []


async def _pty_streams(master_fd: int) -> tuple[Any, asyncio.StreamReader]:
    """Async ends of the terminal: something to write keystrokes to, and the
    screen output to read.

    Both sides go through the event loop rather than blocking reads, so one
    talkative program cannot stall the server, and ``drain`` gives the same
    backpressure the pipe path has.
    """
    loop = asyncio.get_running_loop()
    reader = asyncio.StreamReader()
    await loop.connect_read_pipe(
        lambda: asyncio.StreamReaderProtocol(reader), os.fdopen(master_fd, "rb", 0)
    )
    # A dup so the read and write ends own their own file objects; closing one
    # must not pull the terminal out from under the other.
    transport, protocol = await loop.connect_write_pipe(
        asyncio.streams.FlowControlMixin, os.fdopen(os.dup(master_fd), "wb", 0)
    )
    return asyncio.StreamWriter(transport, protocol, None, loop), reader


def _payload_argv(
    command: str | list[str] | None, env: Mapping[str, str], *, ctty: bool = False
) -> list[str]:
    """The session itself: a login shell (or an exact argv) under ``env``."""
    argv = [*(_ctty_argv() if ctty else []), *_env_argv(env)]
    if isinstance(command, str):
        return [*argv, "bash", "-lc", command]
    if command is None:
        return [*argv, "bash", "-l"]
    return argv + command


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
    No-op off root. Only the workspace directory itself is handed to the uid
    at start (O(1), on the serving path); pre-staged content is the author's
    to own via ``COPY --chown`` or task setup.
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
        require_isolation: bool = False,
        credentials_dir: Path | str | None = None,
        hand_over_root: bool = True,
    ) -> None:
        self.root: Path = Path(root).resolve()
        # Per-instance credential dir, materialized lazily (see _credentials_dir).
        self._cred_dir: Path | None = None
        self._configured_cred_dir = Path(credentials_dir) if credentials_dir else None

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
        self._bwrap = usable_bwrap()
        # Without bwrap there is no `/workspace` mount — the sandbox *is* the real
        # directory, so address it by its real path. Otherwise `cd /workspace`
        # lands in a phantom dir and the editor/bash disagree on where files are.
        # Only override the default; respect an explicit guest_path.
        if self._bwrap is None and guest_path == "/workspace":
            self._guest_path = self.root.as_posix()
        # The caller's spelling of the same directory when it differs from the real
        # path (macOS resolves /tmp to /private/tmp) and sessions run under the real
        # one. A client that knows only the real path treats the other spelling as a
        # workspace-relative address and re-anchors it somewhere that does not exist.
        given = Path(root).absolute().as_posix()
        real = self.root.as_posix()
        self._cwd_aliases = [given] if given != real and self._guest_path == real else []
        # ssh config
        self._ssh_host = host
        self._ssh_port = port
        self._ssh_user = user
        self._shell_uid = shell_uid
        # Whether the root is chowned to shell_uid at start. Off where the
        # image staged it already: whose it is, is the image's statement.
        self._hand_over_root = hand_over_root
        if require_isolation and self._bwrap is None:
            raise RuntimeError(
                "isolation was required but bwrap cannot sandbox here: install "
                "bubblewrap and allow unprivileged user namespaces (e.g. run the "
                "container with --security-opt seccomp=unconfined). Refusing to "
                "serve sessions that would silently run unisolated."
            )
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
        # The sandbox sessions run in, spawned on first use and held until
        # discard_sandbox(). Its namespaces are what makes a process the agent
        # backgrounds outlive the command that started it.
        self._sandbox: asyncio.subprocess.Process | None = None
        self._sandbox_init: int | None = None
        # Sessions start concurrently (an agent can issue parallel tool calls),
        # and two that each started a sandbox would not share one.
        self._sandbox_lock = asyncio.Lock()

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
        if self._bwrap is None and sys.platform != "win32" and shutil.which("bwrap") is None:
            # Once per process: repeating this for every Workspace is noise, and
            # on macOS (no bubblewrap exists) it is an expected state. The
            # present-but-unusable case is diagnosed by usable_bwrap itself.
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
            # Make the workspace dir itself writable by the dropped uid so a
            # fresh workspace is usable out of the box. This is O(1) and must
            # stay that way: it runs on the serving path, before the control
            # port binds. We deliberately do NOT recurse — a baked tree
            # (node_modules, a venv, a dataset) can hold hundreds of thousands
            # of files, and walking it here delays the bind past the deploy
            # readiness probe. Ownership of pre-staged content is the env
            # author's job, done where it's cheap and scoped: at build time
            # (`COPY --chown`) or in task setup over just what was staged.
            assert self._shell_uid is not None
            if self._hand_over_root:
                os.lchown(self.root, self._shell_uid, self._shell_uid)
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
        await self.discard_sandbox()
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
            cwd_aliases=self._cwd_aliases or None,
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
            params={"root": self.root.as_posix(), "setup_diff": True},
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
        info_fd: int | None = None,
        userns_block_fd: int | None = None,
        tty: bool = False,
    ) -> list[str]:
        """Argv that runs ``command`` inside bwrap. Raises if bwrap unavailable.

        The payload runs under ``env -i``, so it starts from exactly
        ``full_env``: with ``inherit_host_env=False`` the host environment
        (server secrets) is left out and only ``self.env`` + ``env`` reach the
        sandbox. Every option here is one bubblewrap 0.4 understands —
        ``--clearenv`` (0.5+) would abort each session on an older bwrap that
        :func:`usable_bwrap` cannot tell apart from a current one.
        """
        if self._bwrap is None:
            raise RuntimeError("bwrap not available on this host")
        target_cwd = cwd if cwd is not None else self._guest_path
        base_env = _without_harness_config(os.environ) if inherit_host_env else {}
        full_env = {**base_env, **self.env, **(env or {})}
        argv: list[str] = [
            self._bwrap,
            "--die-with-parent",
            # Blocking means this side installs the map, so the namespace has
            # to be ours to map: --unshare-user, not the best-effort form.
            "--unshare-user" if userns_block_fd is not None else "--unshare-user-try",
            "--unshare-pid",
            "--unshare-ipc",
            "--unshare-uts",
            "--unshare-cgroup-try",
        ]
        if not self.network:
            argv.append("--unshare-net")
        if info_fd is not None:
            argv.extend(["--info-fd", str(info_fd)])
        if userns_block_fd is not None:
            argv.extend(["--userns-block-fd", str(userns_block_fd)])
        for m in self._system_mounts:
            argv.extend(m.to_bwrap_args())
        argv.extend(["--bind", str(self.root), self._guest_path])
        for m in self.mounts:
            argv.extend(m.to_bwrap_args())
        argv.extend(["--chdir", target_cwd])
        argv.append("--")
        argv.extend(_payload_argv(command, full_env, ctty=tty))
        return argv

    def enter_argv(
        self,
        pid: int,
        command: str | None = None,
        *,
        cwd: str | None = None,
        env: Mapping[str, str] | None = None,
        tty: bool = False,
    ) -> list[str]:
        """Argv that runs ``command`` inside the sandbox *pid* belongs to.

        The counterpart to :meth:`bwrap_argv`, which *creates* a sandbox: this
        joins one that already exists, so successive commands share it. The
        user namespace is joined first — that is what grants the privileges to
        join the rest without any capability the container was not given.

        The network namespace is joined only when the sandbox has one of its
        own, which is exactly when the workspace severed the network. A
        sharing sandbox is already in this process's netns, and that netns
        belongs to an outer user namespace: once joined to bwrap's, we hold no
        authority there and rejoining fails outright.
        """
        nsenter = shutil.which("nsenter") or "/usr/bin/nsenter"
        argv = [
            nsenter,
            "--target",
            str(pid),
            "--user",
            "--mount",
            "--pid",
            "--uts",
            "--ipc",
            *(() if self.network else ("--net",)),
            f"--wd={cwd if cwd is not None else self._guest_path}",
            "--",
        ]
        # Unlike the bwrap path, the drop goes *inside*: joining namespaces
        # needs the privileges the dropped uid does not have.
        argv.extend(self._drop_argv())
        argv.extend(_payload_argv(command, self._full_env(env), ctty=tty))
        return argv

    def _full_env(self, env: Mapping[str, str] | None = None) -> dict[str, str]:
        """The environment a session starts from.

        Dropped sessions get the minimal one built for the wall; otherwise the
        serving process's environment carries through, less HUD's own.
        """
        if self._drops_privileges():
            return {**(self._session_env() or {}), **(env or {})}
        return {**_without_harness_config(os.environ), **self.env, **(env or {})}

    def _drop_argv(self) -> list[str]:
        """The ``setpriv`` prefix that drops to ``shell_uid``, if it applies."""
        if not self._drops_privileges():
            return []
        setpriv = self._setpriv()
        assert setpriv is not None  # guaranteed by _drops_privileges
        uid = str(self._shell_uid)
        # --no-new-privs: without it a setuid binary (or passwordless sudo)
        # inside the workspace would let the dropped shell regain root and
        # read the secrets the wall protects.
        return [setpriv, "--reuid", uid, "--regid", uid, "--clear-groups", "--no-new-privs", "--"]

    # ─── the sandbox sessions share ───────────────────────────────────────

    async def sandbox_pid(self) -> int | None:
        """The live sandbox's init pid, starting one if none is running.

        ``None`` where bwrap cannot sandbox: sessions then run directly, as
        they always have, and nothing persists between them beyond the files.
        """
        if self._bwrap is None:
            return None
        if (live := self._live_sandbox_pid()) is not None:
            return live
        async with self._sandbox_lock:
            # Another session may have started it while this one waited.
            if (live := self._live_sandbox_pid()) is not None:
                return live
            return await self._start_sandbox()

    def _live_sandbox_pid(self) -> int | None:
        if self._sandbox is None or self._sandbox.returncode is not None:
            return None
        assert self._sandbox_init is not None
        return self._sandbox_init

    async def _start_sandbox(self) -> int:
        """Spawn the holder whose namespaces sessions join, and learn its pid.

        The pid comes from bwrap's ``--info-fd`` rather than by searching for
        the holder: the pid that matters is the one *this* process can name,
        and bwrap is the only party that knows which of its forks that is.
        """
        read_fd, write_fd = os.pipe()
        block_read, block_write = os.pipe()
        try:
            os.set_inheritable(write_fd, True)
            os.set_inheritable(block_read, True)
            argv = self.bwrap_argv(_SANDBOX_HOLDER, info_fd=write_fd, userns_block_fd=block_read)
            self._sandbox = await asyncio.create_subprocess_exec(
                *argv,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                pass_fds=(write_fd, block_read),
            )
            os.close(write_fd)
            os.close(block_read)
            write_fd = block_read = -1
            # The sandbox is held at its own creation until its ids are
            # mapped: nothing runs in it, and nothing joins it, before then.
            pid = await install_identity_map(read_fd, block_write)
        finally:
            os.close(read_fd)
            os.close(block_write)
            for stray in (write_fd, block_read):
                if stray != -1:
                    os.close(stray)
        if not pid:
            raise RuntimeError(f"the sandbox holder did not start: {await self._sandbox_error()}")
        assert self._sandbox is not None and self._sandbox.stdout is not None
        try:
            signal = await asyncio.wait_for(self._sandbox.stdout.readline(), 30.0)
        except TimeoutError:
            signal = b""
        if signal != _SANDBOX_READY:
            reason = await self._sandbox_error()
            await self.discard_sandbox()
            raise RuntimeError(f"the sandbox never became ready: {reason}")
        self._sandbox_init = pid
        return pid

    async def _sandbox_error(self) -> str:
        """Whatever the sandbox said on the way down, for a failure message."""
        if self._sandbox is None or self._sandbox.stderr is None:
            return "no output"
        with contextlib.suppress(Exception):
            stderr = await asyncio.wait_for(self._sandbox.stderr.read(2048), 5.0)
            return stderr.decode(errors="replace").strip() or "no output"
        return "no output"

    async def discard_sandbox(self) -> None:
        """Tear the sandbox down, killing everything still running in it.

        The rollout boundary: one adapted image serves many rollouts, and a
        sandbox that outlives its own would hand the next agent the previous
        one's daemons. Killing the holder collapses the pid namespace, so
        every process the agent left behind goes with it — including ones it
        detached. A later session starts a fresh sandbox.
        """
        sandbox, self._sandbox, self._sandbox_init = self._sandbox, None, None
        if sandbox is None or sandbox.returncode is not None:
            return
        sandbox.kill()
        with contextlib.suppress(Exception):
            await asyncio.wait_for(sandbox.wait(), 10.0)

    def shell_argv(
        self,
        command: str | None = None,
        *,
        cwd: str | None = None,
        env: Mapping[str, str] | None = None,
        tty: bool = False,
    ) -> list[str]:
        """Per-session shell argv (bwrap'd if available, else host shell).

        With ``shell_uid`` set and the serving process running as root, the
        whole session is wrapped in ``setpriv`` to drop to that uid.

        ``cwd`` is the one argument the unsandboxed form cannot honour: there
        is no mount namespace to ``--chdir`` into, so the session runs wherever
        the caller starts the process — which for a :class:`Workspace` is its
        root, the same path ``_guest_path`` takes when bwrap is unavailable.
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
                argv = self.bwrap_argv(
                    inner, cwd=cwd, env=walled_env, inherit_host_env=False, tty=tty
                )
            else:
                argv = self.bwrap_argv(inner, cwd=cwd, env=env, tty=tty)
        else:
            # The same payload the sandboxed forms run. Built here too rather
            # than left as a bare shell, so that ``env`` and ``tty`` mean the
            # same thing however the session is placed — and so the session
            # env reaches the shell only *after* any drop: an LD_PRELOAD in it
            # must never be in the environment of the root-run setpriv.
            argv = _payload_argv(command, self._full_env(env), ctty=tty)
        if self._drops_privileges():
            argv = [*self._drop_argv(), *argv]
        return argv

    # ─── ssh server internals ─────────────────────────────────────────

    def _credentials_dir(self) -> Path:
        """Key material lives outside the served root: the root is the agent's
        shell cwd and diff-tracking surface, so secrets don't belong in it.

        ``mkdtemp`` creates a fresh 0700 directory with an unpredictable name
        atomically, so a local user can't pre-place a symlink at the path to
        redirect the private keys the server writes here. A caller that masks
        part of the filesystem from sessions should pass ``credentials_dir``
        pointing inside it: outside the served root is not the same as out of
        the session's reach, and these are the keys to the session itself.
        """
        if self._cred_dir is None:
            if self._configured_cred_dir is not None:
                self._configured_cred_dir.mkdir(parents=True, exist_ok=True)
                self._configured_cred_dir.chmod(0o700)
                self._cred_dir = self._configured_cred_dir
            else:
                # Named for what it holds, not for what put it there.
                self._cred_dir = Path(tempfile.mkdtemp(prefix="ssh-"))
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
        """Environment the agent's shell session sees.

        When dropping privileges, the child would otherwise inherit the
        server's full environment — including any secrets the env process
        holds (the reason ``shell_uid`` exists) — so build a minimal, safe
        environment from scratch. It reaches the shell only *after* the drop
        (``env -i`` inside setpriv, or bwrap ``--setenv``). Otherwise preserve
        the inherited-env behavior, layering ``self.env`` overrides.
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
        pid = await self.sandbox_pid()
        # Sessions start from an exact environment, so a terminal's TERM has to
        # be put there deliberately: without it curses and tput have no
        # terminal description and fall back or fail outright.
        term_type = process.term_type
        wants_tty = bool(term_type)
        session_env = {"TERM": term_type} if term_type else None
        argv = (
            self.shell_argv(process.command, env=session_env, tty=wants_tty)
            if pid is None
            else self.enter_argv(pid, process.command, env=session_env, tty=wants_tty)
        )
        if self._drops_privileges():
            # The pre-drop processes (setpriv, bwrap) run as root; caller env
            # like an LD_PRELOAD in self.env must not load into them. The
            # session env is injected after the drop by shell_argv.
            proc_env: dict[str, str] | None = {
                "PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
            }
        else:
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

        # A client that asked for a pty gets one: the child's std fds are the
        # terminal, so isatty() holds and curses/readline programs behave as
        # they would in a terminal. stderr merges into stdout, as on any tty.
        pty_pair = _open_pty(process) if wants_tty else None
        child_fds: dict[str, Any] = (
            {
                "stdin": asyncio.subprocess.PIPE,
                "stdout": asyncio.subprocess.PIPE,
                "stderr": asyncio.subprocess.PIPE,
            }
            if pty_pair is None
            else {"stdin": pty_pair[1], "stdout": pty_pair[1], "stderr": pty_pair[1]}
        )
        try:
            sub = await create_process_group_exec(
                *argv, **child_fds, cwd=str(self.root), env=proc_env
            )
        except FileNotFoundError as exc:
            if pty_pair is not None:
                os.close(pty_pair[0])
                os.close(pty_pair[1])
            process.stderr.write(f"workspace: cannot spawn shell: {exc}\n".encode())
            process.exit(127)
            return

        if pty_pair is not None:
            # The child holds the terminal now; this side keeps only the master.
            os.close(pty_pair[1])
            stdin_writer, stdout_reader = await _pty_streams(pty_pair[0])
            stderr_reader = None
        else:
            stdin_writer = sub.process.stdin
            stdout_reader = sub.stdout
            stderr_reader = sub.stderr
        assert stdin_writer is not None
        assert stdout_reader is not None

        async def relay_stdin() -> None:
            try:
                while True:
                    try:
                        chunk = await process.stdin.read(65536)
                    except asyncssh.TerminalSizeChanged as resized:
                        # A resize arrives as an exception on the read rather
                        # than as data. It is not an asyncssh.Error, so left
                        # alone it would escape this coroutine and take the
                        # session's keyboard with it.
                        if pty_pair is not None:
                            _set_winsize(
                                pty_pair[0],
                                resized.width,
                                resized.height,
                                resized.pixwidth,
                                resized.pixheight,
                            )
                        continue
                    if not chunk:
                        break
                    stdin_writer.write(chunk)
                    await stdin_writer.drain()
            except (asyncssh.Error, BrokenPipeError, ConnectionResetError, OSError):
                pass
            finally:
                with contextlib.suppress(Exception):
                    stdin_writer.close()

        async def relay_output(
            reader: asyncio.StreamReader, writer: asyncssh.SSHWriter[bytes]
        ) -> None:
            """Forward the child's output as it is produced.

            Streamed, not accumulated: an agent watching a build wants the
            lines while it runs, a session that never exits would otherwise
            say nothing at all, and a command killed at the timeout still
            keeps whatever it managed to print.
            """
            try:
                while chunk := await reader.read(65536):
                    writer.write(chunk)
                    await writer.drain()
            except (asyncssh.Error, BrokenPipeError, ConnectionResetError, OSError):
                # A pty master reads EIO once the child is gone: end of output,
                # not a failure.
                pass

        stdin_task = asyncio.create_task(relay_stdin())
        # One stream on a terminal, where stderr shares the tty, two otherwise.
        output_tasks = [asyncio.create_task(relay_output(stdout_reader, process.stdout))]
        if stderr_reader is not None:
            output_tasks.append(asyncio.create_task(relay_output(stderr_reader, process.stderr)))
        wait_task = asyncio.create_task(sub.wait())
        channel_closed_task = asyncio.create_task(process.channel.wait_closed())
        timed_out = False
        try:
            async with asyncio.timeout(_COMMAND_TIMEOUT):
                done, _ = await asyncio.wait(
                    (wait_task, channel_closed_task),
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if channel_closed_task in done and not wait_task.done():
                    return
                await wait_task
        except TimeoutError:
            timed_out = True
        finally:
            # A command that ran to completion inside the sandbox keeps its
            # process group: `some-server &` is how an agent starts something
            # it means to use in the *next* command, and killing the group
            # here would take it down with the shell that launched it. The
            # sandbox is the lifetime boundary instead — discarding it at the
            # end of the rollout collapses the pid namespace and everything
            # left in it. Nothing bounds a command that timed out, was
            # abandoned mid-flight, or ran with no sandbox at all, so those
            # are still torn down as a group.
            completed = wait_task.done() and not wait_task.cancelled()
            if pid is None or timed_out or not completed:
                await sub.terminate()
            stdin_task.cancel()
            wait_task.cancel()
            channel_closed_task.cancel()
            await asyncio.gather(
                stdin_task,
                wait_task,
                channel_closed_task,
                return_exceptions=True,
            )
            _, output_pending = await asyncio.wait(output_tasks, timeout=1.0)
            for task in output_pending:
                task.cancel()
            await asyncio.gather(*output_tasks, return_exceptions=True)

        if process.channel.is_closing():
            return
        if timed_out:
            # Whatever ran before the deadline has already been relayed; this
            # only says why it stopped.
            process.stderr.write(
                f"workspace: command timed out after {_COMMAND_TIMEOUT:g}s\n".encode()
            )
            process.exit(1)
            return
        process.exit(sub.returncode if sub.returncode is not None else 0)


__all__ = [
    "DEFAULT_SYSTEM_MOUNTS",
    "Mount",
    "MountKind",
    "Workspace",
]
