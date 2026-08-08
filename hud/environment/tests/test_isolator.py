"""Isolator selection and Windows stub."""

from __future__ import annotations

import sys

import pytest

from hud.environment.isolator import (
    Bubblewrap,
    BwrapIsolator,
    SeatbeltIsolator,
    WindowsIsolator,
    missing_isolation_error,
    select_isolator,
)
from hud.environment.seatbelt import Seatbelt


def test_select_isolator_prefers_bwrap() -> None:
    bwrap = Bubblewrap("/usr/bin/bwrap")
    seatbelt = Seatbelt("/usr/bin/sandbox-exec")
    chosen = select_isolator(bwrap, seatbelt)
    assert isinstance(chosen, BwrapIsolator)
    assert chosen.backend is bwrap


def test_select_isolator_falls_back_to_seatbelt() -> None:
    seatbelt = Seatbelt("/usr/bin/sandbox-exec")
    chosen = select_isolator(None, seatbelt)
    assert isinstance(chosen, SeatbeltIsolator)
    assert chosen.backend is seatbelt


def test_select_isolator_windows_probe_is_none() -> None:
    assert select_isolator(None, None) is None
    assert WindowsIsolator.probe() is None


@pytest.mark.asyncio
async def test_windows_isolator_start_and_wrap_raise() -> None:
    win = WindowsIsolator()
    assert win.name == "windows"
    assert win.remounts_guest_path is False
    assert win.uses_namespace_host is False
    with pytest.raises(NotImplementedError, match="Windows"):
        await win.start_sandbox(None)  # type: ignore[arg-type]
    with pytest.raises(NotImplementedError, match="Windows"):
        win.wrap_session(None, "true", cwd=None, env=None, tty=False)  # type: ignore[arg-type]


def test_missing_isolation_error_mentions_platform() -> None:
    msg = missing_isolation_error()
    if sys.platform == "darwin":
        assert "Seatbelt" in msg or "sandbox-exec" in msg
    elif sys.platform == "win32":
        assert "Windows" in msg
    else:
        assert "bwrap" in msg
