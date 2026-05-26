"""Workspace — a directory exposed to an agent over SSH (bwrap-isolated).

A ``Workspace`` is *one* thing: a directory on disk plus an SSH server that
gives the agent a bwrap-isolated bash + SFTP chroot'd to that directory.
Construct it, ``await workspace.start()`` once to bind the SSH listener,
then wire it into your ``Env`` by constructing a ``Capability.ssh(...)``
from the workspace's published URL and keys::

    workspace = Workspace(root="/tmp/coding")
    await workspace.start()
    env = Env(
        name="coding",
        capabilities=[Capability.ssh(
            url=workspace.ssh_url,
            host_pubkey=workspace.ssh_host_pubkey,
            client_key_path=workspace.ssh_client_key_path,
        )],
    )

The env-author manipulates the workspace as a normal directory — write
files with ``(workspace.root / "x.py").write_text(...)``, run commands with
``asyncio.create_subprocess_exec(...)``, etc. There's no ``exec`` /
``read_file`` helper because plain Python is just as good and there's no
benefit to a wrapper.

What the agent sees over SSH:

* A bash session inside a bwrap namespace where the only writable directory
  is ``/workspace`` (= ``workspace.root`` on the host). On non-Linux hosts
  where ``bwrap`` is missing, the session falls back to plain host bash
  (with a startup warning).
* SFTP rooted at ``/`` = ``workspace.root``. The agent can list, read, and
  write anywhere under it — but they can't escape.

Auth: ed25519 host + client keypairs are generated under
``<root>/.hud/ssh/`` on first start. The public host key and the path to
the ephemeral client private key are published in the capability ``params``
so a dev harness can connect immediately. Pass ``authorized_client_keys``
to use pre-existing keys instead (production).

Mounts: pass ``Mount`` instances to expose host paths inside the namespace
— e.g. ``Mount("ro", src="/opt/venv", dst="/opt/venv")`` to share a Python
environment. ``DEFAULT_SYSTEM_MOUNTS`` already covers ``/usr``, ``/etc``,
``/tmp``, ``/proc``, ``/dev`` and the standard ``/lib → /usr/lib`` symlinks.
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import asyncssh

LOGGER = logging.getLogger("hud.env.workspace")


# ─────────────────────────── mount declarations ───────────────────────────


MountKind = Literal["ro", "rw", "tmpfs", "symlink", "proc", "dev"]

# kind -> (normal-flag, optional-variant or None, takes-src)
_MOUNT_FLAGS: dict[MountKind, tuple[str, str | None, bool]] = {
    "ro":      ("--ro-bind", "--ro-bind-try", True),
    "rw":      ("--bind",    "--bind-try",    True),
    "symlink": ("--symlink", None,            True),
    "tmpfs":   ("--tmpfs",   None,            False),
    "proc":    ("--proc",    None,            False),
    "dev":     ("--dev",     None,            False),
}


@dataclass(slots=True, frozen=True)
class Mount:
    """One bwrap mount entry. Construct with kwargs; render with ``to_bwrap_args``.

    ::

        Mount("ro",      src="/usr",      dst="/usr")
        Mount("rw",      src="/data",     dst="/data",  optional=True)
        Mount("symlink", src="usr/lib",   dst="/lib")
        Mount("tmpfs",   dst="/tmp")
        Mount("proc",    dst="/proc")
        Mount("dev",     dst="/dev")
    """

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
    Mount("ro",      src="/usr",     dst="/usr"),
    Mount("ro",      src="/etc",     dst="/etc"),
    Mount("symlink", src="usr/lib",  dst="/lib"),
    Mount("symlink", src="usr/lib64", dst="/lib64"),
    Mount("symlink", src="usr/bin",  dst="/bin"),
    Mount("symlink", src="usr/sbin", dst="/sbin"),
    Mount("proc",    dst="/proc"),
    Mount("dev",     dst="/dev"),
    Mount("tmpfs",   dst="/tmp"),
)


# ─────────────────────────── the workspace ───────────────────────────


_DEFAULT_USER = "agent"


class Workspace:
    """A directory exposed to an agent over SSH (bwrap-isolated shell + SFTP)."""

    def __init__(
        self,
        root: Path | str,
        *,
        # bwrap configuration
        mounts: Sequence[Mount] = (),
        network: bool = False,
        env: Mapping[str, str] | None = None,
        system_mounts: Sequence[Mount] | None = None,
        # ssh server configuration
        host: str = "127.0.0.1",
        port: int = 0,
        user: str = _DEFAULT_USER,
        host_key_path: Path | None = None,
        authorized_client_keys: list[Path] | None = None,
    ) -> None:
        self.root: Path = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

        # bwrap state
        self.mounts: tuple[Mount, ...] = tuple(mounts)
        self.network = network
        self.env: dict[str, str] = dict(env or {})
        self._system_mounts: tuple[Mount, ...] = tuple(
            system_mounts if system_mounts is not None else DEFAULT_SYSTEM_MOUNTS,
        )
        self._bwrap = shutil.which("bwrap")
        if self._bwrap is None and sys.platform != "win32":
            LOGGER.warning(
                "bwrap not on PATH; SSH sessions will run WITHOUT isolation. "
                "Install bubblewrap, or run inside a Linux container that has it.",
            )

        # ssh state (set in start())
        self._ssh_host = host
        self._ssh_port = port
        self._ssh_user = user
        self._ssh_host_key_path = host_key_path
        self._ssh_authorized_client_keys = list(authorized_client_keys or [])
        self._acceptor: asyncssh.SSHAcceptor | None = None
        self._client_key_path: Path | None = None
        self._host_pubkey_str: str = ""

    # ─── lifecycle ────────────────────────────────────────────────────

    async def start(self) -> None:
        """Bind the SSH listener. Idempotent; call once after construction."""
        if self._acceptor is not None:
            return
        host_key, self._host_pubkey_str = self._load_or_generate_host_key()
        authorized_keys_path = self._ensure_authorized_keys_file()
        self._acceptor = await asyncssh.listen(
            host=self._ssh_host,
            port=self._ssh_port,
            server_host_keys=[host_key],
            authorized_client_keys=str(authorized_keys_path),
            process_factory=self._handle_process,
            sftp_factory=self._sftp_factory,
            allow_scp=True,
            line_editor=False,
            keepalive_interval=30,
            encoding=None,
        )
        LOGGER.info(
            "Workspace SSH listening on %s as user %r (client key: %s)",
            self.ssh_url, self._ssh_user, self._client_key_path,
        )

    # ─── ssh accessors / capability ───────────────────────────────────

    @property
    def ssh_url(self) -> str:
        """Network URL the agent connects to, e.g. ``ssh://127.0.0.1:54321``."""
        if self._acceptor is None:
            raise RuntimeError("Workspace not started; call `await workspace.start()` first")
        sock = self._acceptor.sockets[0].getsockname()
        return f"ssh://{sock[0]}:{sock[1]}"

    @property
    def ssh_host_pubkey(self) -> str:
        """OpenSSH-format public host key string for the harness's ``known_hosts``."""
        return self._host_pubkey_str

    @property
    def ssh_client_key_path(self) -> Path | None:
        """Path to the ephemeral client private key (None if external keys were supplied)."""
        return self._client_key_path

    @property
    def ssh_user(self) -> str:
        """SSH username the agent should connect as."""
        return self._ssh_user

    # ─── argv builders (public — useful if you want your own subprocess) ──

    @property
    def bwrap_available(self) -> bool:
        return self._bwrap is not None

    def bwrap_argv(
        self,
        command: list[str] | str,
        *,
        cwd: str = "/workspace",
        env: Mapping[str, str] | None = None,
    ) -> list[str]:
        """Build the argv that runs ``command`` inside the bwrap namespace.

        Raises if bwrap is unavailable — branch on ``bwrap_available``.
        """
        if self._bwrap is None:
            raise RuntimeError("bwrap not available on this host")
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
        argv.extend(["--bind", str(self.root), "/workspace"])
        for m in self.mounts:
            argv.extend(m.to_bwrap_args())
        argv.extend(["--chdir", cwd])
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
        cwd: str = "/workspace",
        env: Mapping[str, str] | None = None,
    ) -> list[str]:
        """Argv for the per-session shell (bwrap'd if available, host bash otherwise)."""
        if self._bwrap is not None:
            inner: list[str] | str = ["bash", "-lc", command] if command else ["bash", "-l"]
            return self.bwrap_argv(inner, cwd=cwd, env=env)
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
        """Materialise the authorized_keys file asyncssh wants on disk."""
        creds = self._credentials_dir()
        auth_path = creds / "authorized_keys"
        pub_lines: list[str] = []

        if self._ssh_authorized_client_keys:
            for p in self._ssh_authorized_client_keys:
                pub_lines.append(Path(p).read_text().strip())
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
            )
        except FileNotFoundError as exc:
            process.stderr.write(f"workspace: cannot spawn shell: {exc}\n".encode())
            process.exit(127)
            return

        await process.redirect(stdin=sub.stdin, stdout=sub.stdout, stderr=sub.stderr)
        try:
            exit_code = await sub.wait()
        except asyncio.CancelledError:
            sub.kill()
            await sub.wait()
            raise
        process.exit(exit_code)

    def _sftp_factory(self, chan: asyncssh.SSHServerChannel[bytes]) -> asyncssh.SFTPServer:
        return asyncssh.SFTPServer(chan, chroot=str(self.root).encode())


__all__ = [
    "DEFAULT_SYSTEM_MOUNTS",
    "Mount",
    "MountKind",
    "Workspace",
]
