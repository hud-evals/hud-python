"""Tests for telemetry exporter with mock backend."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import patch

import pytest

from hud.settings import settings
from hud.telemetry.exporter import _do_upload, flush, queue_span


@pytest.fixture(autouse=True)
def drain_exporter():
    """Drain the background worker before and after each test."""
    assert flush(timeout=1.0)
    yield
    assert flush(timeout=1.0)


class _RecordingUpload:
    """Captures (task_run_id, spans, api_key) for each upload."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, list[dict[str, Any]], str]] = []

    def __call__(
        self,
        task_run_id: str,
        spans: list[dict[str, Any]],
        telemetry_url: str,
        api_key: str,
    ) -> None:
        self.calls.append((task_run_id, spans, api_key))


def _configure(
    monkeypatch: pytest.MonkeyPatch,
    *,
    api_key: str | None = "test-key",
    enabled: bool = True,
) -> None:
    """Pin the settings fields the exporter reads, on the real object.

    Field patches (not a whole-object MagicMock) keep every other setting
    real — including the root conftest's isolation pins, which a mock object
    would replace with truthy auto-attributes.
    """
    monkeypatch.setattr(settings, "api_key", api_key)
    monkeypatch.setattr(settings, "telemetry_enabled", enabled)
    monkeypatch.setattr(settings, "hud_telemetry_url", "https://api.hud.ai")


class TestDoUpload:
    def test_upload_posts_to_trace_endpoint(self):
        with patch("hud.telemetry.exporter.make_request_sync") as mock_request:
            _do_upload(
                task_run_id="test-task-123",
                spans=[{"name": "test.span"}],
                telemetry_url="https://api.hud.ai",
                api_key="test-key",
            )

        mock_request.assert_called_once()
        kwargs = mock_request.call_args.kwargs
        assert kwargs["method"] == "POST"
        assert "test-task-123" in kwargs["url"]
        assert kwargs["api_key"] == "test-key"
        assert kwargs["json"] == {"telemetry": [{"name": "test.span"}]}

    def test_upload_swallows_request_errors(self):
        with patch("hud.telemetry.exporter.make_request_sync", side_effect=Exception("boom")):
            _do_upload("test-task-123", [{"name": "test.span"}], "https://api.hud.ai", "test-key")


class TestQueueSpan:
    @pytest.mark.parametrize(
        ("api_key", "enabled", "attributes"),
        [
            (None, True, {"hud.task_run_id": "123"}),
            ("test-key", False, {"hud.task_run_id": "123"}),
            ("test-key", True, {}),
        ],
    )
    def test_span_is_dropped(self, monkeypatch, api_key, enabled, attributes):
        upload = _RecordingUpload()
        _configure(monkeypatch, api_key=api_key, enabled=enabled)
        with patch("hud.telemetry.exporter._do_upload", side_effect=upload):
            queue_span({"name": "test", "attributes": attributes})
            assert flush(timeout=1.0)

        assert upload.calls == []

    def test_spans_upload_in_one_batch_per_trace(self, monkeypatch):
        upload = _RecordingUpload()
        _configure(monkeypatch)
        with patch("hud.telemetry.exporter._do_upload", side_effect=upload):
            queue_span({"name": "span-1", "attributes": {"hud.task_run_id": "task-1"}})
            queue_span({"name": "span-2", "attributes": {"hud.task_run_id": "task-1"}})
            queue_span({"name": "span-3", "attributes": {"hud.task_run_id": "task-2"}})
            assert flush(timeout=1.0)

        by_task = {task_run_id: spans for task_run_id, spans, _ in upload.calls}
        assert [span["name"] for span in by_task["task-1"]] == ["span-1", "span-2"]
        assert [span["name"] for span in by_task["task-2"]] == ["span-3"]

    def test_upload_uses_settings_api_key(self, monkeypatch):
        upload = _RecordingUpload()
        _configure(monkeypatch)
        with patch("hud.telemetry.exporter._do_upload", side_effect=upload):
            queue_span({"name": "test", "attributes": {"hud.task_run_id": "task-1"}})
            assert flush(timeout=1.0)

        assert [api_key for _, _, api_key in upload.calls] == ["test-key"]


class TestLocalExport:
    def test_span_written_to_local_dir(self, monkeypatch, tmp_path):
        _configure(monkeypatch, api_key=None, enabled=False)
        monkeypatch.setattr(settings, "telemetry_local_dir", str(tmp_path))

        queue_span({"name": "s", "trace_id": "t-1", "attributes": {"hud.task_run_id": "task-1"}})

        lines = (tmp_path / "t-1.jsonl").read_text(encoding="utf-8").splitlines()
        assert json.loads(lines[0])["name"] == "s"

    def test_non_str_local_dir_fails_loudly(self, monkeypatch):
        _configure(monkeypatch, api_key=None, enabled=False)
        monkeypatch.setattr(settings, "telemetry_local_dir", object())

        with pytest.raises(TypeError, match="telemetry_local_dir"):
            queue_span({"name": "s", "attributes": {"hud.task_run_id": "task-1"}})


class TestFlush:
    def test_flush_is_noop_when_idle(self):
        assert flush(timeout=1.0)

    def test_flush_drains_queued_spans(self, monkeypatch):
        upload = _RecordingUpload()
        _configure(monkeypatch)
        with patch("hud.telemetry.exporter._do_upload", side_effect=upload):
            queue_span({"name": "final-span", "attributes": {"hud.task_run_id": "task-1"}})
            assert flush(timeout=1.0)

        assert [span["name"] for _, spans, _ in upload.calls for span in spans] == ["final-span"]
