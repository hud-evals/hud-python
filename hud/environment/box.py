"""A placed box as the place a workspace's sessions run, reached through its runtime."""

from __future__ import annotations

import asyncio
import logging
import os
import posixpath
import secrets
import shlex
import shutil
import tarfile
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import IO, TYPE_CHECKING, Any

from hud.utils.process import ProcessGroup, ProcessOutput, create_process_group_exec

from .file_tracker import DEFAULT_EXCLUDE_PATTERNS, DEFAULT_MAX_FILE_SIZE, FileTracker
from .sandbox import SessionProcess, TerminalSize, _open_pty, _pty_streams

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from hud.runtime.core import Box
    from hud.utils.process import ProcessResult

    from .file_tracker import FileTrackingOp

LOGGER = logging.getLogger("hud.environment.box")

# Every session carries this in its environment, so what it started can be found in the box.
_SESSION_MARKER = "HUD_BOX_SESSION"
_STOP_TIMEOUT_S = 10.0
# The box's login shell, given the shell's arguments: bash where the image has it, else sh.
_LOGIN_SHELL = 'if command -v bash >/dev/null 2>&1; then exec bash -l "$@"; fi; exec sh -l "$@"'
# Kills every process with an environment entry matching the pattern given as $1, with the
# box's tr and grep. Only the sessions' own user can read their processes' environments: a
# container's root lacks the ptrace capability that reading another user's would take.
_STOP_SESSION = (
    'for d in /proc/[0-9]*; do { tr "\\0" "\\n" < "$d/environ"; } 2>/dev/null'
    ' | grep -q "$1" && kill -KILL "${d#/proc/}" 2>/dev/null; done; true'
)
# What a mirror's ``FileTracker`` tracks, as ``find`` selects it in the box: regular files
# within its size limit, outside its excluded names.
_TRACKED = (
    ". \\( "
    + " -o ".join(
        f"-type d -name {shlex.quote(pattern[:-1])}"
        if pattern.endswith("/")
        else f"-name {shlex.quote(pattern)}"
        for pattern in DEFAULT_EXCLUDE_PATTERNS
    )
    + f" \\) -prune -o -type f -size -{DEFAULT_MAX_FILE_SIZE + 1}c"
)
# Lists, NUL-terminated, the tracked files modified since the "old" marker (unless $2 is
# "all"), an empty record, then every tracked file; $1 names the tracker's markers. Each
# listing marks its start, and the next starts from the newest earlier mark that the box's
# test tells apart from it: an overlap that catches a write in a mark's own clock tick, or
# its own second where the box's find compares whole seconds.
_LIST_TRACKED = f"""\
for tool in find tar; do
  command -v "$tool" >/dev/null || {{ echo "file tracking needs $tool in the box" >&2; exit 127; }}
done
m="${{TMPDIR:-/tmp}}/hud-files-$1"
touch "$m.next" || exit 1
[ -e "$m.old" ] || touch -r "$m.next" "$m.old" || exit 1
[ "$2" = all ] || find {_TRACKED} -newer "$m.old" -print0
printf '\\0'
find {_TRACKED} -print0
if [ -e "$m.mid" ] && [ "$m.mid" -ot "$m.next" ]; then mv -f "$m.mid" "$m.old" || exit 1; fi
mv -f "$m.next" "$m.mid"
"""


async def _stop_sessions(box: Box, user: str | None, token: str) -> None:
    """Kill what the sessions marked ``token`` run in ``box``; any token, if ``token`` is ``""``."""
    pattern = f"^{_SESSION_MARKER}={token}$" if token else f"^{_SESSION_MARKER}="
    stop = await create_process_group_exec(
        *box.exec_argv(["sh", "-c", _STOP_SESSION, "sh", pattern], user=user),
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.PIPE,
    )
    result = await stop.complete(max_wait=_STOP_TIMEOUT_S)
    if result.returncode != 0 or result.timed_out:
        detail = result.stderr.decode("utf-8", "replace").strip()
        LOGGER.warning("could not stop sessions' commands in the box: %s", detail)


@dataclass(kw_only=True)
class _BoxProcess(ProcessGroup):
    """A session's local client, and the command it runs in the box as ``user``."""

    box: Box
    user: str | None
    token: str

    async def terminate(self) -> None:
        # Ending the local client leaves the command running in the box: stop it there first.
        await _stop_sessions(self.box, self.user, self.token)
        await super().terminate()


class BoxSandbox:
    """A placed box, as the sandbox a ``Workspace`` serves sessions from.

    Each session runs in the box through its runtime, as ``user`` in ``guest_path``, with
    ``env`` set over the box's own environment. The box bounds what a session leaves
    running: a command that finishes keeps what it started in the background, and a
    session stopped mid-command is stopped in the box too.
    """

    bwrap_available = False
    supports_process_connections = False

    def __init__(
        self,
        box: Box,
        *,
        guest_path: str,
        user: str | None = None,
        env: Mapping[str, str] | None = None,
    ) -> None:
        self.box = box
        self.guest_path = guest_path
        self.user = user
        self.env = dict(env or {})
        self._local: Path | None = None

    @property
    def root(self) -> Path:
        """The local stand-in for the box's ``guest_path``."""
        return self.credentials_dir() / "root"

    def prepare(self) -> None:
        """Create the local directory for the workspace's keys. Idempotent."""
        if self._local is None:
            self._local = Path(tempfile.mkdtemp(prefix="hud-box-"))
            (self._local / "root").mkdir()

    async def start(self) -> None:
        self.prepare()

    async def stop(self) -> None:
        """Remove what ``prepare`` created; the box's processes are its runtime's to end."""
        if self._local is not None:
            shutil.rmtree(self._local, ignore_errors=True)
            self._local = None

    async def terminate_sessions(self) -> None:
        """Kill everything the sessions started, background work included."""
        await _stop_sessions(self.box, self.user, "")

    def credentials_dir(self) -> Path:
        """A private local directory for the workspace's keys, never inside the box."""
        self.prepare()
        assert self._local is not None
        return self._local

    async def spawn_session(
        self,
        command: str | None,
        *,
        term: str | None = None,
        terminal_size: TerminalSize = (80, 24, 0, 0),
        connections: Sequence[str] = (),
    ) -> SessionProcess:
        """Start an agent session in the box: its login shell, or ``command`` under one.

        ``term`` asks for a terminal of that type.
        """
        if connections:
            raise ValueError("process-bound connections are not available in a box")
        token = secrets.token_hex(16)
        argv = self.box.exec_argv(
            ["sh", "-c", _LOGIN_SHELL, "sh", *(("-c", command) if command is not None else ())],
            user=self.user,
            cwd=self.guest_path,
            env={**self.env, **({"TERM": term} if term else {}), _SESSION_MARKER: token},
            tty=term is not None,
        )
        terminal = _open_pty(terminal_size) if term is not None else None
        outputs: list[ProcessOutput] = []
        try:
            if terminal is None:
                stdout_read, stdout_write = os.pipe()
                stderr_read, stderr_write = os.pipe()
                outputs = [ProcessOutput(stdout_read), ProcessOutput(stderr_read)]
                try:
                    client = await create_process_group_exec(
                        *argv,
                        stdin=asyncio.subprocess.PIPE,
                        stdout=stdout_write,
                        stderr=stderr_write,
                    )
                    for output in outputs:
                        await output.start()
                finally:
                    os.close(stdout_write)
                    os.close(stderr_write)
            else:
                client = await create_process_group_exec(
                    *argv, stdin=terminal[1], stdout=terminal[1], stderr=terminal[1]
                )
        except BaseException:
            for output in outputs:
                output.finish()
            if terminal is not None:
                os.close(terminal[0])
                os.close(terminal[1])
            raise

        process = _BoxProcess(client.process, box=self.box, user=self.user, token=token)
        if terminal is not None:
            # The client holds the terminal now; this side keeps only the master.
            os.close(terminal[1])
            stdin, output = await _pty_streams(terminal[0])
            return SessionProcess(
                process,
                stdin,
                output.reader,
                None,
                held=True,
                _terminal=terminal[0],
                _outputs=[output],
            )
        stdin = client.process.stdin
        assert stdin is not None
        return SessionProcess(
            process, stdin, outputs[0].reader, outputs[1].reader, held=True, _outputs=outputs
        )


class BoxFileTracker:
    """File tracking for ``path`` in a box: a ``FileTracker`` over a local mirror of it,
    brought up to date through the box before each operation that scans.

    An update copies the files modified since an earlier update, as the box's ``find`` tells
    times apart, and those new to the mirror, and drops those gone from the box. The first
    update and the final flush copy every file, so the flush holds every change, one that
    kept an older modification time included. Copying takes ``find`` and ``tar`` in the
    box, run as ``user``: the mirror holds only what that user can read.
    """

    def __init__(self, box: Box, path: str, *, user: str | None = None) -> None:
        self.box = box
        self.path = path
        self.user = user
        self.root = Path(tempfile.mkdtemp(prefix="hud-box-files-"))
        self._tracker = FileTracker(self.root)
        self._markers = secrets.token_hex(8)
        self._mirrored: set[str] | None = None

    async def call(self, op: FileTrackingOp) -> dict[str, Any]:
        """Run one ``files.*`` operation over the box's ``path``."""
        if op != "snapshot":
            await self._update(everything=self._mirrored is None or op == "flush")
        return await asyncio.to_thread(self._tracker.call, op)

    def close(self) -> None:
        """Remove the mirror."""
        shutil.rmtree(self.root, ignore_errors=True)

    async def _update(self, *, everything: bool) -> None:
        listing = await self._run(
            ["sh", "-c", _LIST_TRACKED, "sh", self._markers, "all" if everything else "changed"]
        )
        if listing.returncode != 0:
            detail = listing.stderr.decode("utf-8", "replace").strip()
            raise RuntimeError(f"could not list the files tracked in the box: {detail}")
        records = [os.fsdecode(record).removeprefix("./") for record in listing.stdout.split(b"\0")]
        cut = records.index("")
        # A name with a newline cannot be asked of tar, so it goes untracked.
        present = {record for record in records[cut + 1 :] if record and "\n" not in record}
        changed = set(records[:cut])
        mirrored = self._mirrored or set()
        wanted = present if everything else (present & changed) | (present - mirrored)
        gone = mirrored - present
        with tempfile.TemporaryFile() as archive:
            if wanted:
                names = b"".join(b"./" + os.fsencode(name) + b"\n" for name in sorted(wanted))
                copy = await self._run(["tar", "-cf", "-", "-T", "-"], stdin=names, stdout=archive)
                # tar still archives the rest when a file vanished after the listing.
                if os.fstat(archive.fileno()).st_size == 0:
                    detail = copy.stderr.decode("utf-8", "replace").strip()
                    raise RuntimeError(f"could not copy the files tracked in the box: {detail}")
                archive.seek(0)
            copied = await asyncio.to_thread(
                self._apply, gone, archive if wanted else None, time.time_ns()
            )
        self._mirrored = (mirrored - gone) | copied

    def _apply(self, gone: set[str], archive: IO[bytes] | None, stamp: int) -> set[str]:
        """Drop ``gone`` from the mirror and unpack ``archive`` into it; the files unpacked.

        Each unpacked file is stamped ``stamp``, so the tracker reads it again even where
        the box's copy kept its size and modification time.
        """
        for name in gone:
            (self.root / name).unlink(missing_ok=True)
        copied: set[str] = set()
        if archive is None:
            return copied
        with tarfile.open(fileobj=archive, mode="r:") as tar:
            for member in tar:
                if not member.isreg():
                    continue
                name = posixpath.normpath(member.name)
                target = self.root / name
                if target.is_dir():
                    shutil.rmtree(target)
                tar.extract(member, self.root, filter="data")
                os.utime(target, ns=(stamp, stamp))
                copied.add(name)
        return copied

    async def _run(
        self,
        argv: Sequence[str],
        *,
        stdin: bytes = b"",
        stdout: IO[bytes] | int = asyncio.subprocess.PIPE,
    ) -> ProcessResult:
        process = await create_process_group_exec(
            *self.box.exec_argv(argv, user=self.user, cwd=self.path),
            stdin=asyncio.subprocess.PIPE,
            stdout=stdout,
            stderr=asyncio.subprocess.PIPE,
        )
        pipe = process.process.stdin
        assert pipe is not None
        pipe.write(stdin)
        pipe.close()
        return await process.complete()
