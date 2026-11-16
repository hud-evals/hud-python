"""Workspace: the ``ssh/2`` shell capability over a ``Sandbox``."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import socket
import sys
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any, Generic, Protocol, cast

import asyncssh
from typing_extensions import TypeVar

from hud.capabilities.base import Capability
from hud.capabilities.ssh import PROCESS_CONNECTIONS_REQUEST
from hud.environment.file_tracker import FileTracker
from hud.environment.sandbox import Sandbox
from hud.environment.utils import forward_output
from hud.utils.deprecation import moved_to, resolve_retired
from hud.utils.exceptions import HudDeprecationWarning

if TYPE_CHECKING:
    from collections.abc import Sequence

    from hud.environment.sandbox import SessionProcess, TerminalSize

LOGGER = logging.getLogger("hud.environment.workspace")

_PROCESS_CLOSE_TIMEOUT_S = 5.0
_DEFAULT_USER = "agent"

# What v0.6.23 defined here and ``hud.environment.sandbox`` holds now.
_RETIRED = moved_to(
    "hud.environment.sandbox",
    "DEFAULT_SYSTEM_MOUNTS",
    "Bubblewrap",
    "Mount",
    "MountKind",
    "usable_bwrap",
)
# v0.6.23's ``Workspace`` members that its ``Sandbox`` holds now, by their names there.
_RETIRED_MEMBERS = {
    "add_peer": "add_peer",
    "add_process_connection": "add_process_connection",
    "addresses": "addresses",
    "allowed_hosts": "allowed_hosts",
    "bwrap_available": "bwrap_available",
    "discard_sandbox": "discard",
    "env": "env",
    "launch": "launch",
    "local_aliases": "local_aliases",
    "mounts": "mounts",
    "network": "network",
    "owns_netns": "owns_netns",
    "peers": "peers",
    "ports": "ports",
    "remove_peer": "remove_peer",
    "remove_process_connection": "remove_process_connection",
    "run": "run",
    "sandbox_pid": "pid",
    "supports_process_connections": "supports_process_connections",
    "terminate_sessions": "terminate_sessions",
    "visiting": "visiting",
}


def __getattr__(name: str) -> Any:
    return resolve_retired(__name__, name, _RETIRED)


class WorkspaceSandbox(Protocol):
    """What a ``Workspace`` needs from the sandbox its sessions run in."""

    @property
    def root(self) -> Path: ...

    @property
    def guest_path(self) -> str: ...

    @property
    def bwrap_available(self) -> bool: ...

    @property
    def supports_process_connections(self) -> bool: ...

    def prepare(self) -> None: ...

    async def start(self) -> None: ...

    async def stop(self) -> None: ...

    def credentials_dir(self) -> Path: ...

    async def spawn_session(
        self,
        command: str | None,
        *,
        term: str | None = None,
        terminal_size: TerminalSize = (80, 24, 0, 0),
        connections: Sequence[str] = (),
    ) -> SessionProcess: ...


_SandboxT = TypeVar("_SandboxT", bound=WorkspaceSandbox, default=Sandbox)


class _TrackedServer(asyncssh.SSHServer):
    """Keeps ``connections`` current with the connections a listener holds open."""

    def __init__(self, connections: set[asyncssh.SSHServerConnection]) -> None:
        self._connections = connections
        self._connection: asyncssh.SSHServerConnection | None = None

    def connection_made(self, conn: asyncssh.SSHServerConnection) -> None:
        self._connection = conn
        self._connections.add(conn)

    def connection_lost(self, exc: Exception | None) -> None:
        if self._connection is not None:
            self._connections.discard(self._connection)


class Workspace(Generic[_SandboxT]):
    """The agent's shell: an SSH server, published as an ``ssh/2`` capability, whose
    sessions run in ``sandbox``.

    ``env.workspace(root)`` attaches one over a bubblewrap ``Sandbox`` to an
    ``Environment``, which starts it and publishes the capability when the env
    serves. Construction writes nothing: keys, sockets and the sandbox root are
    created on first use. Drive it directly with ``start()``, ``capability()`` and
    ``stop()`` to publish the capability yourself.

    v0.6.23's ``Workspace(root, **sandbox_options)`` and the sandbox members it held
    (``run``, ``terminate_sessions``, ``discard_sandbox`` and the rest) keep working
    with a ``HudDeprecationWarning`` until hud v0.7.0.
    """

    def __init__(
        self,
        sandbox: _SandboxT | Path | str,
        *,
        host: str = "127.0.0.1",
        port: int = 0,
        user: str = _DEFAULT_USER,
        host_key_path: Path | None = None,
        authorized_client_keys: list[Path] | None = None,
        track_files: bool = False,
        **sandbox_options: Any,
    ) -> None:
        if isinstance(sandbox, Path | str):
            warnings.warn(
                "`Workspace(root, ...)` is deprecated and will be removed in hud v0.7.0; "
                "pass `Workspace(Sandbox(root, ...), ...)`, with the sandbox options on "
                "`Sandbox`.",
                HudDeprecationWarning,
                stacklevel=2,
            )
            sandbox = cast("_SandboxT", Sandbox(sandbox, **sandbox_options))
        elif sandbox_options:
            raise TypeError(
                f"Workspace got sandbox options {sorted(sandbox_options)}; pass them to "
                "the Sandbox it serves"
            )
        self.sandbox: _SandboxT = sandbox
        self._ssh_host = host
        self._ssh_port = port
        self._ssh_user = user
        self._ssh_host_key_path = host_key_path
        self._ssh_authorized_client_keys = list(authorized_client_keys or [])
        self._acceptor: asyncssh.SSHAcceptor | None = None
        self._connections: set[asyncssh.SSHServerConnection] = set()
        self._client_key_path: Path | None = None
        self._host_key: asyncssh.SSHKey | None = None
        self._host_pubkey_str: str | None = None
        self._authorized_keys_path: Path | None = None
        self._sock: socket.socket | None = None
        self._bound_host: str | None = None
        self._bound_port: int | None = None
        self._track_files = track_files
        self._file_tracker: FileTracker | None = None

    def __getattr__(self, name: str) -> Any:
        member = _RETIRED_MEMBERS.get(name)
        if member is None or name == "sandbox":
            raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")
        warnings.warn(
            f"`Workspace.{name}` is deprecated and will be removed in hud v0.7.0; use "
            f"`workspace.sandbox.{member}`.",
            HudDeprecationWarning,
            stacklevel=2,
        )
        return getattr(self.sandbox, member)

    @property
    def root(self) -> Path:
        """The sandbox's root, where file tracking looks."""
        return self.sandbox.root

    def _prepare_runtime(self) -> None:
        """Materialize the sandbox, the SSH credentials, and bind the socket."""
        if self._sock is not None:
            return
        self.sandbox.prepare()
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

    async def start(self) -> None:
        """Ensure the SSH accept loop is running. Idempotent.

        Returns only after every published workspace service is accepting clients.
        """
        await self.sandbox.start()
        try:
            self._prepare_runtime()
            if self._acceptor is None:
                assert self._sock is not None
                assert self._host_key is not None
                assert self._authorized_keys_path is not None
                self._acceptor = await asyncssh.listen(
                    sock=self._sock,
                    server_factory=lambda: _TrackedServer(self._connections),
                    server_host_keys=[self._host_key],
                    authorized_client_keys=str(self._authorized_keys_path),
                    process_factory=self._handle_process,
                    line_editor=False,
                    keepalive_interval=30,
                    encoding=None,
                )
            if self._track_files and self._file_tracker is None:
                await self._start_file_tracking()
        except BaseException:
            await self.stop()
            raise

    async def _start_file_tracking(self) -> None:
        """Take the baseline snapshot."""
        tracker = FileTracker(self.root)
        # The baseline walk is CPU-bound; keep it off the event loop.
        await asyncio.get_running_loop().run_in_executor(None, tracker.take_baseline)
        self._file_tracker = tracker

    async def stop(self) -> None:
        """Stop the sandbox and the SSH server, end its open connections, and release
        the socket and keys.

        A later ``start`` re-binds (fresh port unless one was pinned) with
        fresh credentials.
        """
        await self.sandbox.stop()
        self._file_tracker = None
        if self._acceptor is not None:
            self._acceptor.close()
            # Closing the listener leaves its connections open; access ends with them.
            for connection in list(self._connections):
                connection.close()
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
        return self._ssh_user

    def capability(self, name: str = "shell") -> Capability:
        """The concrete ``ssh`` capability — materializes keys + bind.

        Carries the managed client key's *content*, so the binding
        authenticates from anywhere the daemon is reachable — including a
        client on the other side of a container boundary.
        """
        key_path = self.ssh_client_key_path
        return Capability.ssh(
            name=name,
            url=self.ssh_url,
            user=self.ssh_user,
            host_pubkey=self.ssh_host_pubkey,
            client_key=key_path.read_text() if key_path else None,
            client_key_path=key_path,
            cwd=self.sandbox.guest_path,
            isolation="bwrap" if self.sandbox.bwrap_available else "none",
            process_connections=self.sandbox.supports_process_connections,
        )

    @property
    def tracks_files(self) -> bool:
        return self._track_files

    @property
    def file_tracker(self) -> FileTracker | None:
        """The tracker over ``root``, set while a tracking workspace is started."""
        return self._file_tracker

    def _load_or_generate_host_key(self) -> tuple[asyncssh.SSHKey, str]:
        if self._ssh_host_key_path is not None:
            key = asyncssh.read_private_key(self._ssh_host_key_path)
        else:
            key_path = self.sandbox.credentials_dir() / "host_ed25519"
            if key_path.exists():
                key = asyncssh.read_private_key(key_path)
            else:
                key = asyncssh.generate_private_key("ssh-ed25519")
                key.write_private_key(str(key_path))
                key.write_public_key(str(key_path.with_suffix(".pub")))
        return key, key.export_public_key().decode("ascii").strip()

    def _ensure_authorized_keys_file(self) -> Path:
        """Write the authorized_keys file asyncssh wants on disk."""
        creds = self.sandbox.credentials_dir()
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

    def _requested_process_connections(
        self,
        process: asyncssh.SSHServerProcess[bytes],
    ) -> tuple[str, ...]:
        raw = getattr(process, "env", {}).get(PROCESS_CONNECTIONS_REQUEST)
        if raw is None:
            return ()
        try:
            requested = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError("invalid process connection request") from exc
        if not isinstance(requested, list) or not all(isinstance(name, str) for name in requested):
            raise ValueError("process connection request must be a list of names")
        return tuple(dict.fromkeys(str(name) for name in requested))

    async def _handle_process(self, process: asyncssh.SSHServerProcess[bytes]) -> None:
        if sys.platform == "win32" and isinstance(self.sandbox, Sandbox):
            await self._handle_windows_process(process, self.sandbox)
            return
        try:
            session = await self.sandbox.spawn_session(
                process.command,
                term=process.term_type or None,
                terminal_size=process.get_terminal_size(),
                connections=self._requested_process_connections(process),
            )
        except Exception as exc:
            LOGGER.warning("workspace session setup failed: %s", exc)
            if not process.channel.is_closing():
                reason = "spawn" if isinstance(exc, FileNotFoundError) else "start"
                process.stderr.write(f"workspace: cannot {reason} shell: {exc}\n".encode())
                process.exit(127 if isinstance(exc, FileNotFoundError) else 1)
            return

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
                        await session.resize(
                            resized.width,
                            resized.height,
                            resized.pixwidth,
                            resized.pixheight,
                        )
                        continue
                    if not chunk:
                        break
                    session.stdin.write(chunk)
                    await session.stdin.drain()
            except asyncssh.SignalReceived:
                await session.terminate()
            except (asyncssh.Error, BrokenPipeError, ConnectionResetError, OSError):
                pass
            finally:
                with contextlib.suppress(Exception):
                    session.close_stdin()

        stdin_task = asyncio.create_task(relay_stdin())
        output_tasks = [asyncio.create_task(forward_output(session.stdout, process.stdout))]
        if session.stderr is not None:
            output_tasks.append(asyncio.create_task(forward_output(session.stderr, process.stderr)))
        wait_task = asyncio.create_task(session.wait())
        channel_closed_task = asyncio.create_task(process.channel.wait_closed())
        returncode: int | None = None
        try:
            done, _ = await asyncio.wait(
                (wait_task, channel_closed_task),
                return_when=asyncio.FIRST_COMPLETED,
            )
            if channel_closed_task in done and not wait_task.done():
                return
            returncode = await wait_task
        finally:
            # A command that ran to completion inside the held sandbox keeps its
            # process group: `some-server &` is how an agent starts something
            # it means to use in the *next* command, and killing the group
            # here would take it down with the shell that launched it. The
            # sandbox is the lifetime boundary instead — discarding it at the
            # end of the rollout collapses the pid namespace and everything
            # left in it. An abandoned command or one running without a held
            # sandbox is still torn down as a group.
            completed = wait_task.done() and not wait_task.cancelled()
            if not session.held or not completed:
                await session.terminate()
            stdin_task.cancel()
            wait_task.cancel()
            channel_closed_task.cancel()
            await asyncio.gather(
                stdin_task,
                wait_task,
                channel_closed_task,
                return_exceptions=True,
            )
            session.release()
            if process.channel.is_closing():
                for task in output_tasks:
                    task.cancel()
                await asyncio.gather(*output_tasks, return_exceptions=True)
            else:
                await asyncio.gather(*output_tasks)

        if process.channel.is_closing():
            return
        assert returncode is not None
        process.exit(returncode)

    async def _handle_windows_process(
        self, process: asyncssh.SSHServerProcess[bytes], sandbox: Sandbox
    ) -> None:
        # On Windows, asyncio.create_subprocess_exec uses the ProactorEventLoop's
        # IOCP machinery for process-exit notification.  When the IOCP event fires
        # after the subprocess coroutine has already returned (a race that can
        # happen even when communicate() calls wait() internally), it corrupts
        # asyncssh's IOCP state and permanently breaks the SSH session.
        # Running a blocking subprocess in a worker thread sidesteps IOCP
        # entirely: the blocking WaitForSingleObject in the worker thread drains
        # the process exit before the Future resolves, leaving no pending events.
        try:
            loop = asyncio.get_running_loop()
            child, job = await loop.run_in_executor(
                None, sandbox.spawn_windows_session, process.command
            )
        except FileNotFoundError as exc:
            process.stderr.write(f"workspace: cannot spawn shell: {exc}\n".encode())
            process.exit(127)
            return
        except OSError as exc:
            process.stderr.write(f"workspace: cannot own shell process tree: {exc}\n".encode())
            process.exit(1)
            return

        communicate_task = asyncio.create_task(asyncio.to_thread(child.communicate))
        channel_closed_task = asyncio.create_task(process.channel.wait_closed())
        try:
            done, _ = await asyncio.wait(
                (communicate_task, channel_closed_task),
                return_when=asyncio.FIRST_COMPLETED,
            )
            if channel_closed_task in done and not communicate_task.done():
                return
            stdout, stderr = await communicate_task
        finally:
            if not communicate_task.done():
                job.close()
                try:
                    async with asyncio.timeout(_PROCESS_CLOSE_TIMEOUT_S):
                        await asyncio.shield(communicate_task)
                except TimeoutError:
                    for pipe in (child.stdout, child.stderr):
                        if pipe is not None:
                            with contextlib.suppress(OSError):
                                pipe.close()
                    communicate_task.cancel()
                    with contextlib.suppress(TimeoutError):
                        await asyncio.wait_for(
                            asyncio.gather(communicate_task, return_exceptions=True),
                            _PROCESS_CLOSE_TIMEOUT_S,
                        )
            else:
                job.close()
            channel_closed_task.cancel()
            await asyncio.gather(channel_closed_task, return_exceptions=True)

        if process.channel.is_closing():
            return
        if stdout:
            process.stdout.write(stdout)
        if stderr:
            process.stderr.write(stderr)
        assert child.returncode is not None
        process.exit(child.returncode)
