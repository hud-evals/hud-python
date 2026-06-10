"""Workspace: a directory + bwrap-isolated SSH server (bash + SFTP chroot)."""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import socket
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import asyncssh

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from hud.capabilities import Capability

LOGGER = logging.getLogger("hud.environment.workspace")

# Set once the first Workspace logs the missing-bwrap notice (avoid per-instance spam).
_warned_no_bwrap = False


# ─────────────────────────── mount declarations ───────────────────────────


MountKind = Literal["ro", "rw", "tmpfs", "symlink", "proc", "dev"]

# kind -> (normal-flag, optional-variant or None, takes-src)
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
    """Directory + bwrap-isolated SSH (bash + chroot'd SFTP)."""

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
    ) -> None:
        self.root: Path = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

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
        if self._bwrap is None and sys.platform != "win32":
            # Once per process: repeating this on every Workspace construction is
            # noise, and on macOS (no bubblewrap exists) it is an expected state.
            global _warned_no_bwrap
            if not _warned_no_bwrap:
                _warned_no_bwrap = True
                log = LOGGER.warning if sys.platform == "linux" else LOGGER.info
                log(
                    "bwrap not on PATH; SSH sessions will run WITHOUT isolation. "
                    "Install bubblewrap, or run inside a Linux container that has it.",
                )

        # ssh config
        self._ssh_host = host
        self._ssh_user = user
        self._ssh_host_key_path = host_key_path
        self._ssh_authorized_client_keys = list(authorized_client_keys or [])
        self._acceptor: asyncssh.SSHAcceptor | None = None
        self._client_key_path: Path | None = None

        # ─── synchronous spinup ───
        self._host_key, self._host_pubkey_str = self._load_or_generate_host_key()
        self._authorized_keys_path = self._ensure_authorized_keys_file()
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind((host, port))
        self._sock.listen(128)
        self._bound_host, self._bound_port = self._sock.getsockname()[:2]

        # Serve from a dedicated background event loop (daemon thread), so the
        # SSH server is live right after construction — module-level
        # ``Workspace(...)`` just works, with no ``@env.initialize`` /
        # ``await ws.start()`` boilerplate.
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._loop.run_forever,
            name=f"hud-workspace-ssh-{self._bound_port}",
            daemon=True,
        )
        self._thread.start()
        self._serve_future = asyncio.run_coroutine_threadsafe(self._serve(), self._loop)

        LOGGER.info(
            "Workspace SSH bound on %s as user %r (client key: %s)",
            self.ssh_url,
            self._ssh_user,
            self._client_key_path,
        )

    # ─── lifecycle ────────────────────────────────────────────────────

    async def _serve(self) -> None:
        """Run the asyncssh accept loop on the pre-bound socket."""
        self._acceptor = await asyncssh.listen(
            sock=self._sock,
            server_host_keys=[self._host_key],
            authorized_client_keys=str(self._authorized_keys_path),
            process_factory=self._handle_process,
            sftp_factory=self._sftp_factory,
            allow_scp=True,
            line_editor=False,
            keepalive_interval=30,
            encoding=None,
        )

    async def start(self) -> None:
        """Wait until the background SSH acceptor is up. Idempotent.

        The server starts on its own background loop at construction; this only
        surfaces a startup error early. Calling it is optional and kept for
        backward compatibility.
        """
        await asyncio.wrap_future(self._serve_future)

    # ─── ssh accessors / capability ───────────────────────────────────

    @property
    def ssh_url(self) -> str:
        """``ssh://host:port`` — available immediately after construction."""
        return f"ssh://{self._bound_host}:{self._bound_port}"

    @property
    def ssh_host_pubkey(self) -> str:
        """OpenSSH-format public host key (for harness ``known_hosts``)."""
        return self._host_pubkey_str

    @property
    def ssh_client_key_path(self) -> Path | None:
        """Ephemeral client private key path (None if external keys supplied)."""
        return self._client_key_path

    @property
    def ssh_user(self) -> str:
        """SSH username."""
        return self._ssh_user

    def capability(self, name: str = "shell") -> Capability:
        """The ``ssh`` capability for this workspace.

        Available at construction (url/keys are generated synchronously), so an env
        can declare it up front: ``Environment(..., capabilities=[ws.capability()])``.
        """
        from hud.capabilities import Capability

        return Capability.ssh(
            name=name,
            url=self.ssh_url,
            user=self.ssh_user,
            host_pubkey=self.ssh_host_pubkey,
            client_key_path=self.ssh_client_key_path,
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
    ) -> list[str]:
        """Argv that runs ``command`` inside bwrap. Raises if bwrap unavailable."""
        if self._bwrap is None:
            raise RuntimeError("bwrap not available on this host")
        target_cwd = cwd if cwd is not None else self._guest_path
        full_env = {**os.environ, **self.env, **(env or {})}
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
        """Per-session shell argv (bwrap'd if available, else host shell)."""
        if self._bwrap is not None:
            inner: list[str] | str = ["bash", "-lc", command] if command else ["bash", "-l"]
            return self.bwrap_argv(inner, cwd=cwd, env=env)
        if sys.platform == "win32":
            if command is not None:
                return ["cmd.exe", "/c", command]
            return ["cmd.exe"]
        if command is not None:
            return ["bash", "-lc", command]
        return ["bash", "-l"]

    # ─── ssh server internals ─────────────────────────────────────────

    def _credentials_dir(self) -> Path:
        d = self.root / ".hud" / "ssh"
        d.mkdir(parents=True, exist_ok=True)
        return d

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

    async def _handle_process(self, process: asyncssh.SSHServerProcess[bytes]) -> None:
        argv = self.shell_argv(process.command)
        try:
            sub = await asyncio.create_subprocess_exec(
                *argv,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(self.root),
            )
        except FileNotFoundError as exc:
            process.stderr.write(f"workspace: cannot spawn shell: {exc}\n".encode())
            process.exit(127)
            return

        # On Windows, process.redirect + sub.wait() hangs because asyncio
        # pipes don't signal EOF properly for cmd.exe subprocesses.
        # Use communicate() which handles this correctly.
        try:
            stdout_data, stderr_data = await sub.communicate(
                input=None,
            )
        except asyncio.CancelledError:
            sub.kill()
            await sub.wait()
            raise

        if stdout_data:
            process.stdout.write(stdout_data)
        if stderr_data:
            process.stderr.write(stderr_data)
        process.exit(sub.returncode if sub.returncode is not None else 0)

    def _sftp_factory(self, chan: asyncssh.SSHServerChannel[bytes]) -> asyncssh.SFTPServer:
        return asyncssh.SFTPServer(chan, chroot=str(self.root).encode())


__all__ = [
    "DEFAULT_SYSTEM_MOUNTS",
    "Mount",
    "MountKind",
    "Workspace",
]
