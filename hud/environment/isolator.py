"""Platform sandbox isolators for Workspace sessions.

``Workspace`` picks one isolator at construction via :func:`select_isolator`.
Linux uses bubblewrap, Darwin uses Seatbelt. Windows is stubbed so a future
AppContainer / WFP backend can plug in without further Workspace branching.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Mapping

    from hud.environment.seatbelt import Seatbelt
    from hud.environment.workspace import Workspace

LOGGER = logging.getLogger("hud.environment.isolator")


@dataclass(slots=True, frozen=True)
class Bubblewrap:
    path: str
    pid_unshare: str | None = None


# Set once the first probe runs (avoid per-instance work).
_bwrap_usable: Bubblewrap | Literal[False] | None = None


def usable_bwrap() -> Bubblewrap | None:
    """A working bubblewrap launch mode for this substrate, if one exists."""
    global _bwrap_usable
    if isinstance(_bwrap_usable, Bubblewrap):
        return _bwrap_usable
    if _bwrap_usable is False:
        return None

    path = shutil.which("bwrap")
    if path is None:
        return None
    probe_binary = shutil.which("true")
    if probe_binary is None:
        return None

    direct = Bubblewrap(path)
    launches = [
        (
            direct,
            [
                path,
                "--unshare-user",
                "--unshare-pid",
                "--ro-bind",
                "/",
                "/",
                "--proc",
                "/proc",
                "--",
                probe_binary,
            ],
        )
    ]
    if unshare := shutil.which("unshare"):
        staged = Bubblewrap(path, pid_unshare=unshare)
        launches.append(
            (
                staged,
                [
                    unshare,
                    "--kill-child=KILL",
                    "--pid",
                    "--mount-proc",
                    path,
                    "--unshare-user",
                    "--ro-bind",
                    "/",
                    "/",
                    "--",
                    probe_binary,
                ],
            )
        )

    failure = "unknown error"
    for launch, argv in launches:
        try:
            probe = subprocess.run(
                argv,
                capture_output=True,
                timeout=15,
                check=False,
            )
            if probe.returncode == 0:
                _bwrap_usable = launch
                return launch
            failure = probe.stderr.decode("utf-8", "replace").strip()[:120]
        except (OSError, subprocess.SubprocessError):
            continue

    _bwrap_usable = False
    LOGGER.warning(
        "bwrap is installed but cannot create an isolated process namespace (%s); "
        "sessions will run WITHOUT isolation.",
        failure,
    )
    return None


@runtime_checkable
class Isolator(Protocol):
    """Platform sandbox backend for a Workspace."""

    @property
    def name(self) -> str:
        """Capability / telemetry label (``bwrap``, ``seatbelt``, ``windows``)."""
        ...

    @property
    def remounts_guest_path(self) -> bool:
        """True when the workspace root is remounted at ``guest_path`` (e.g. ``/workspace``)."""
        ...

    @property
    def uses_namespace_host(self) -> bool:
        """True when SSH sessions join a shared namespace host rather than re-wrapping."""
        ...

    def missing_error(self) -> str:
        """Error text when this backend is the platform default but unavailable."""
        ...

    async def start_sandbox(self, workspace: Workspace) -> int:
        """Start the long-lived sandbox holder; return its pid."""
        ...

    def wrap_session(
        self,
        workspace: Workspace,
        command: str | None,
        *,
        cwd: str | None,
        env: Mapping[str, str] | None,
        tty: bool,
    ) -> list[str]:
        """Argv that runs an SSH session command inside this isolator."""
        ...


@dataclass(frozen=True, slots=True)
class BwrapIsolator:
    backend: Bubblewrap

    @property
    def name(self) -> str:
        return "bwrap"

    @property
    def remounts_guest_path(self) -> bool:
        return True

    @property
    def uses_namespace_host(self) -> bool:
        return True

    def missing_error(self) -> str:
        return (
            "isolation was required but bwrap cannot sandbox here: install "
            "bubblewrap and use a container runtime that allows unprivileged "
            "user namespaces. Refusing to serve sessions that would silently "
            "run unisolated."
        )

    async def start_sandbox(self, workspace: Workspace) -> int:
        return await workspace._start_bwrap_sandbox()

    def wrap_session(
        self,
        workspace: Workspace,
        command: str | None,
        *,
        cwd: str | None,
        env: Mapping[str, str] | None,
        tty: bool,
    ) -> list[str]:
        return workspace._bwrap_session_argv(command, cwd=cwd, env=env, tty=tty)


@dataclass(frozen=True, slots=True)
class SeatbeltIsolator:
    backend: Seatbelt

    @property
    def name(self) -> str:
        return "seatbelt"

    @property
    def remounts_guest_path(self) -> bool:
        return False

    @property
    def uses_namespace_host(self) -> bool:
        return False

    def missing_error(self) -> str:
        return (
            "isolation was required but sandbox-exec (Seatbelt) cannot sandbox "
            "here: ensure /usr/bin/sandbox-exec is available and permitted. "
            "Refusing to serve sessions that would silently run unisolated."
        )

    async def start_sandbox(self, workspace: Workspace) -> int:
        return await workspace._start_seatbelt_sandbox()

    def wrap_session(
        self,
        workspace: Workspace,
        command: str | None,
        *,
        cwd: str | None,
        env: Mapping[str, str] | None,
        tty: bool,
    ) -> list[str]:
        return workspace._seatbelt_session_argv(command, cwd=cwd, env=env, tty=tty)


@dataclass(frozen=True, slots=True)
class WindowsIsolator:
    """Placeholder for a future Windows AppContainer / WFP backend.

    :meth:`probe` returns ``None`` until implemented so Workspace soft-falls
    back the same way as a missing bwrap/Seatbelt. Adding support should mean
    implementing this class and registering it in :func:`select_isolator` —
    not new ``sys.platform`` branches in Workspace.
    """

    @property
    def name(self) -> str:
        return "windows"

    @property
    def remounts_guest_path(self) -> bool:
        return False

    @property
    def uses_namespace_host(self) -> bool:
        return False

    def missing_error(self) -> str:
        return (
            "isolation was required but Windows Workspace sandboxing is not "
            "available yet. Refusing to serve sessions that would silently "
            "run unisolated."
        )

    @classmethod
    def probe(cls) -> WindowsIsolator | None:
        return None

    async def start_sandbox(self, workspace: Workspace) -> int:
        raise NotImplementedError("Windows Workspace isolation is not implemented")

    def wrap_session(
        self,
        workspace: Workspace,
        command: str | None,
        *,
        cwd: str | None,
        env: Mapping[str, str] | None,
        tty: bool,
    ) -> list[str]:
        raise NotImplementedError("Windows Workspace isolation is not implemented")


def missing_isolation_error() -> str:
    """Platform-appropriate error when no isolator could be selected."""
    if sys.platform == "darwin":
        return (
            "isolation was required but sandbox-exec (Seatbelt) cannot sandbox "
            "here: ensure /usr/bin/sandbox-exec is available and permitted. "
            "Refusing to serve sessions that would silently run unisolated."
        )
    if sys.platform == "win32":
        return WindowsIsolator().missing_error()
    return (
        "isolation was required but bwrap cannot sandbox here: install "
        "bubblewrap and use a container runtime that allows unprivileged "
        "user namespaces. Refusing to serve sessions that would silently "
        "run unisolated."
    )


def select_isolator(
    bwrap: Bubblewrap | None,
    seatbelt: Seatbelt | None,
) -> Isolator | None:
    """Pick the first viable isolator. Order: bwrap → Seatbelt → Windows.

    Prefer bubblewrap whenever it works so Linux containers stay on the
    namespace path even if other backends are present.
    """
    if bwrap is not None:
        return BwrapIsolator(bwrap)
    if seatbelt is not None:
        return SeatbeltIsolator(seatbelt)
    return WindowsIsolator.probe()
