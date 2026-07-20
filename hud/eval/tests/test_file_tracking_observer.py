"""file_tracking_observer: setup must gate polling."""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING, Any

from hud.capabilities import Capability
from hud.environment.file_tracker import FileTracker
from hud.eval import file_tracking as observer
from hud.telemetry.context import set_trace_context

if TYPE_CHECKING:
    import pytest

_CAP = Capability(name="filetracking", protocol="filetracking/1", url="tcp://127.0.0.1:1")


class _FakeFt:
    def __init__(
        self,
        *,
        setup_raises: bool = False,
        flush_result: dict[str, Any] | None = None,
    ) -> None:
        self.setup_raises = setup_raises
        self.flush_result = flush_result
        self.setup_calls = 0
        self.snapshot_calls = 0
        self.diff_calls = 0
        self.flush_calls = 0
        self.close_calls = 0

    async def close(self) -> None:
        self.close_calls += 1

    async def call(self, method: str) -> dict[str, Any]:
        if method == "setup":
            self.setup_calls += 1
            if self.setup_raises:
                raise RuntimeError("setup boom")
            return {"files_changed": 1, "patches": [{"path": "setup.txt"}]}
        if method == "snapshot":
            self.snapshot_calls += 1
            return {"files": [], "files_scanned": 0}
        if method == "diff":
            self.diff_calls += 1
            return {"files_changed": 1, "patches": [{"path": "a.txt"}]}
        if method == "flush":
            self.flush_calls += 1
            if self.flush_result is not None:
                return self.flush_result
            return {
                "diff": {"files_changed": 1, "patches": [{"path": "final.txt"}]},
                "capture": {"files_captured": 1, "files": [{"path": "report.xlsx"}]},
            }
        raise ValueError(method)


class _BoundClient:
    def binding(self, name: str) -> object:
        return _CAP


def _record_emitters(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[list[Any], list[Any], list[Any], list[Any]]:
    diffs: list[Any] = []
    setups: list[Any] = []
    snapshots: list[Any] = []
    captures: list[Any] = []

    def _emit(
        span_name: str,
        payload: Any,
        *,
        started_at: str,
        ended_at: str | None = None,
    ) -> bool:
        if span_name == "filetracking.diff":
            diffs.append(payload)
        elif span_name == "filetracking.setup":
            setups.append(payload)
        elif span_name == "filetracking.snapshot":
            snapshots.append(payload)
        elif span_name == "filetracking.capture":
            captures.append(payload)
        return True

    monkeypatch.setattr(observer, "_emit_file_tracking", _emit)
    return diffs, setups, snapshots, captures


def _connects_to(monkeypatch: pytest.MonkeyPatch, ft: _FakeFt) -> None:
    class _FakeFileTrackingClient:
        @classmethod
        async def connect(cls, cap: Capability) -> _FakeFt:
            assert cap == _CAP
            return ft

    monkeypatch.setattr(observer, "FileTrackingClient", _FakeFileTrackingClient)


async def test_setup_failure_skips_polling(monkeypatch: pytest.MonkeyPatch) -> None:
    from hud.settings import settings

    monkeypatch.setattr(settings, "telemetry_enabled", True)
    monkeypatch.setattr(settings, "file_tracking_interval", 0.01)
    diffs, setups, snapshots, captures = _record_emitters(monkeypatch)
    ft = _FakeFt(setup_raises=True)
    _connects_to(monkeypatch, ft)

    async with observer.file_tracking_observer(_BoundClient()):  # type: ignore[arg-type]
        await asyncio.sleep(0.05)

    assert ft.setup_calls == 1
    assert ft.snapshot_calls == 0
    assert ft.diff_calls == 0
    assert ft.flush_calls == 0
    assert ft.close_calls == 1
    assert diffs == []
    assert snapshots == []
    assert setups == []
    assert captures == []


async def test_successful_setup_anchors_and_polls(monkeypatch: pytest.MonkeyPatch) -> None:
    from hud.settings import settings

    monkeypatch.setattr(settings, "telemetry_enabled", True)
    monkeypatch.setattr(settings, "file_tracking_interval", 0.01)
    diffs, setups, snapshots, captures = _record_emitters(monkeypatch)
    ft = _FakeFt()
    _connects_to(monkeypatch, ft)

    async with observer.file_tracking_observer(_BoundClient()):  # type: ignore[arg-type]
        await asyncio.sleep(0.05)

    assert ft.setup_calls == 1
    assert setups == [{"files_changed": 1, "patches": [{"path": "setup.txt"}]}]
    assert len(snapshots) == 1  # manifest anchor emitted once
    assert ft.diff_calls >= 1
    assert ft.flush_calls == 1
    assert ft.close_calls == 1
    assert diffs  # at least one diff streamed
    assert {"files_changed": 1, "patches": [{"path": "final.txt"}]} in diffs
    assert captures == [{"files_captured": 1, "files": [{"path": "report.xlsx"}]}]


async def test_flush_emits_skipped_capture_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    from hud.settings import settings

    capture = {
        "files_changed": 1,
        "files_eligible": 1,
        "files_captured": 0,
        "files_skipped": 1,
        "truncated": True,
        "files": [],
    }
    monkeypatch.setattr(settings, "telemetry_enabled", True)
    monkeypatch.setattr(settings, "file_tracking_interval", 60.0)
    diffs, setups, snapshots, captures = _record_emitters(monkeypatch)
    ft = _FakeFt(flush_result={"diff": {"files_changed": 0, "patches": []}, "capture": capture})
    _connects_to(monkeypatch, ft)

    async with observer.file_tracking_observer(_BoundClient()):  # type: ignore[arg-type]
        pass

    assert ft.flush_calls == 1
    assert len(snapshots) == 1
    assert len(setups) == 1
    assert diffs == []
    assert captures == [capture]


async def test_connect_failure_does_not_break_the_rollout(monkeypatch: pytest.MonkeyPatch) -> None:
    from hud.settings import settings

    monkeypatch.setattr(settings, "telemetry_enabled", True)
    diffs, setups, snapshots, captures = _record_emitters(monkeypatch)

    class _FailingFileTrackingClient:
        @classmethod
        async def connect(cls, cap: Capability) -> object:
            assert cap == _CAP
            raise ConnectionError("tunnel refused")

    monkeypatch.setattr(observer, "FileTrackingClient", _FailingFileTrackingClient)

    # A failed connection must degrade to a no-op, not raise into the agent loop.
    ran = False
    async with observer.file_tracking_observer(_BoundClient()):  # type: ignore[arg-type]
        ran = True

    assert ran
    assert diffs == []
    assert snapshots == []
    assert setups == []
    assert captures == []


async def test_file_tracking_client_reads_large_snapshot_frame() -> None:
    large_path = "a" * (80 * 1024)

    async def handle(
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        await reader.readline()
        writer.write(
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "result": {
                        "files_scanned": 1,
                        "files": [{"path": large_path, "sha256": "0" * 64, "size": 1}],
                    },
                },
                separators=(",", ":"),
            ).encode("utf-8")
            + b"\n"
        )
        await writer.drain()
        writer.close()
        await writer.wait_closed()

    server = await asyncio.start_server(handle, "127.0.0.1", 0)
    host, port = server.sockets[0].getsockname()[:2]
    cap = Capability(name="filetracking", protocol="filetracking/1", url=f"tcp://{host}:{port}")
    try:
        ft = await observer.FileTrackingClient.connect(cap)
        snapshot = await ft.call("snapshot")
        assert snapshot["files"][0]["path"] == large_path
    finally:
        server.close()
        await server.wait_closed()


def test_file_tracking_frame_limit_covers_flush_caps() -> None:
    encoded_capture_cap = 4 * ((FileTracker._MAX_CAPTURE_TOTAL_BYTES + 2) // 3)
    escaped_diff_cap = 2 * FileTracker._MAX_DIFF_BYTES
    assert escaped_diff_cap + encoded_capture_cap <= observer._FRAME_LIMIT_BYTES


def test_emit_file_tracking_noops_without_a_trace_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[dict[str, Any]] = []
    monkeypatch.setattr(observer, "queue_span", captured.append)

    emitted = observer._emit_file_tracking(
        "filetracking.diff",
        {"files_changed": 1},
        started_at="2026-06-18T22:00:00Z",
    )

    assert emitted is False
    assert captured == []


def test_emit_file_tracking_builds_schema_tagged_span(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list[dict[str, Any]] = []
    monkeypatch.setattr(observer, "queue_span", captured.append)

    with set_trace_context("run-abc-123"):
        emitted = observer._emit_file_tracking(
            "filetracking.capture",
            {"files_captured": 1, "files": [{"path": "report.xlsx"}]},
            started_at="2026-06-18T22:00:00Z",
        )

    assert emitted is True
    span = captured[0]
    assert span["name"] == "filetracking.capture"
    assert span["attributes"]["hud.schema"] == "hud.filetracking.v1"
    assert span["attributes"]["hud.task_run_id"] == "run-abc-123"
    assert span["attributes"]["hud.payload"]["files"][0]["path"] == "report.xlsx"
    assert len(span["trace_id"]) == 32
