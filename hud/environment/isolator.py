"""Platform sandbox isolators for Workspace sessions.

``Workspace`` asks :func:`select_isolator` once at construction. That function
owns probe order: bubblewrap (:mod:`hud.environment.bwrap`) → Seatbelt
(:mod:`hud.environment.seatbelt`) → Windows stub. Adding a backend means
extending selection here, not branching in Workspace.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from hud.environment.bwrap import usable_bwrap
from hud.environment.seatbelt import usable_seatbelt

if TYPE_CHECKING:
    from collections.abc import Mapping

    from hud.environment.bwrap import Bubblewrap
    from hud.environment.seatbelt import Seatbelt
    from hud.environment.workspace import Workspace


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


def select_isolator() -> Isolator | None:
    """Probe substrates and pick the first viable isolator.

    Order: bwrap → Seatbelt → Windows. Prefer bubblewrap whenever it works so
    Linux containers stay on the namespace path even if other backends exist.
    """
    if (backend := usable_bwrap()) is not None:
        return BwrapIsolator(backend)
    if (backend := usable_seatbelt()) is not None:
        return SeatbeltIsolator(backend)
    return WindowsIsolator.probe()
