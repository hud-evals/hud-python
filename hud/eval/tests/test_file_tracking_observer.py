"""file_tracking_observer: setup must gate polling.

The observer re-baselines past scenario setup and emits the manifest anchor
before it starts streaming diffs. If that setup fails, it must not poll — a
stale baseline would misattribute scenario-setup edits to the agent.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from hud.eval import file_tracking as observer

if TYPE_CHECKING:
    import pytest


class _FakeFt:
    def __init__(self, *, advance_raises: bool = False) -> None:
        self.advance_raises = advance_raises
        self.advance_calls = 0
        self.snapshot_calls = 0
        self.diff_calls = 0

    async def advance(self) -> dict[str, Any]:
        self.advance_calls += 1
        if self.advance_raises:
            raise RuntimeError("advance boom")
        return {"advanced": True}

    async def snapshot(self) -> dict[str, Any]:
        self.snapshot_calls += 1
        return {"files": [], "files_scanned": 0}

    async def diff(self) -> dict[str, Any]:
        self.diff_calls += 1
        return {"files_changed": 1, "patches": [{"path": "a.txt"}]}


class _FakeClient:
    def __init__(self, ft: _FakeFt) -> None:
        self._ft = ft

    def binding(self, name: str) -> object:
        return object()

    async def open(self, name: str) -> _FakeFt:
        return self._ft


class _OpenFailsClient:
    """A bound capability whose open() fails (e.g. a refused tunnel)."""

    def binding(self, name: str) -> object:
        return object()

    async def open(self, name: str) -> object:
        raise ConnectionError("tunnel refused")


def _record_emitters(monkeypatch: pytest.MonkeyPatch) -> tuple[list[Any], list[Any]]:
    diffs: list[Any] = []
    snapshots: list[Any] = []

    def _diff(payload: Any, *, started_at: str, ended_at: str | None = None) -> bool:
        diffs.append(payload)
        return True

    def _snapshot(payload: Any, *, started_at: str, ended_at: str | None = None) -> bool:
        snapshots.append(payload)
        return True

    monkeypatch.setattr(observer, "emit_file_diff", _diff)
    monkeypatch.setattr(observer, "emit_file_snapshot", _snapshot)
    return diffs, snapshots


async def test_setup_failure_skips_polling(monkeypatch: pytest.MonkeyPatch) -> None:
    from hud.settings import settings

    monkeypatch.setattr(settings, "telemetry_enabled", True)
    monkeypatch.setattr(settings, "file_tracking_interval", 0.01)
    diffs, snapshots = _record_emitters(monkeypatch)
    ft = _FakeFt(advance_raises=True)

    async with observer.file_tracking_observer(_FakeClient(ft)):  # type: ignore[arg-type]
        await asyncio.sleep(0.05)

    assert ft.advance_calls == 1
    # advance() raised, so the anchor snapshot and all diff polling are skipped.
    assert ft.snapshot_calls == 0
    assert ft.diff_calls == 0
    assert diffs == []
    assert snapshots == []


async def test_successful_setup_anchors_and_polls(monkeypatch: pytest.MonkeyPatch) -> None:
    from hud.settings import settings

    monkeypatch.setattr(settings, "telemetry_enabled", True)
    monkeypatch.setattr(settings, "file_tracking_interval", 0.01)
    diffs, snapshots = _record_emitters(monkeypatch)
    ft = _FakeFt()

    async with observer.file_tracking_observer(_FakeClient(ft)):  # type: ignore[arg-type]
        await asyncio.sleep(0.05)

    assert ft.advance_calls == 1
    assert len(snapshots) == 1  # manifest anchor emitted once
    assert ft.diff_calls >= 1
    assert diffs  # at least one diff streamed


async def test_open_failure_does_not_break_the_rollout(monkeypatch: pytest.MonkeyPatch) -> None:
    from hud.settings import settings

    monkeypatch.setattr(settings, "telemetry_enabled", True)
    diffs, snapshots = _record_emitters(monkeypatch)

    # A failed open must degrade to a no-op, not raise into the agent loop.
    ran = False
    async with observer.file_tracking_observer(_OpenFailsClient()):  # type: ignore[arg-type]
        ran = True

    assert ran
    assert diffs == []
    assert snapshots == []
