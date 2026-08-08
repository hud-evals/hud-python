"""Isolator selection and Windows stub."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

import pytest

from hud.environment.bwrap import Bubblewrap
from hud.environment.isolator import (
    BwrapIsolator,
    SeatbeltIsolator,
    WindowsIsolator,
    missing_isolation_error,
    select_isolator,
)
from hud.environment.seatbelt import Seatbelt

if TYPE_CHECKING:
    from pathlib import Path


def test_select_isolator_prefers_bwrap(monkeypatch: pytest.MonkeyPatch) -> None:
    bwrap = Bubblewrap("/usr/bin/bwrap")
    seatbelt = Seatbelt("/usr/bin/sandbox-exec")
    monkeypatch.setattr("hud.environment.isolator.usable_bwrap", lambda: bwrap)
    monkeypatch.setattr("hud.environment.isolator.usable_seatbelt", lambda: seatbelt)
    chosen = select_isolator()
    assert isinstance(chosen, BwrapIsolator)
    assert chosen.backend is bwrap


def test_select_isolator_falls_back_to_seatbelt(monkeypatch: pytest.MonkeyPatch) -> None:
    seatbelt = Seatbelt("/usr/bin/sandbox-exec")
    monkeypatch.setattr("hud.environment.isolator.usable_bwrap", lambda: None)
    monkeypatch.setattr("hud.environment.isolator.usable_seatbelt", lambda: seatbelt)
    chosen = select_isolator()
    assert isinstance(chosen, SeatbeltIsolator)
    assert chosen.backend is seatbelt


def test_select_isolator_windows_probe_is_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("hud.environment.isolator.usable_bwrap", lambda: None)
    monkeypatch.setattr("hud.environment.isolator.usable_seatbelt", lambda: None)
    assert select_isolator() is None
    assert WindowsIsolator.probe() is None


@pytest.mark.asyncio
async def test_windows_isolator_start_and_wrap_raise(tmp_path: Path) -> None:
    from hud.environment.workspace import Workspace

    win = WindowsIsolator()
    assert win.name == "windows"
    assert win.remounts_guest_path is False
    assert win.uses_namespace_host is False
    # Any Workspace instance is fine — methods raise before using it.
    ws = Workspace(tmp_path / "root")
    with pytest.raises(NotImplementedError, match="Windows"):
        await win.start_sandbox(ws)
    with pytest.raises(NotImplementedError, match="Windows"):
        win.wrap_session(ws, "true", cwd=None, env=None, tty=False)


def test_missing_isolation_error_mentions_platform() -> None:
    msg = missing_isolation_error()
    if sys.platform == "darwin":
        assert "Seatbelt" in msg or "sandbox-exec" in msg
    elif sys.platform == "win32":
        assert "Windows" in msg
    else:
        assert "bwrap" in msg
