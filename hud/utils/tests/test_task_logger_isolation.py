"""Tests for task log handler context isolation."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from hud.otel import context as otel_context
from hud.utils.task_logger import TaskLogHandler, TaskLogger


def _create_handler(tmp_path: Path, task_run_id: str) -> TaskLogHandler:
    logger = TaskLogger(task_run_id=task_run_id, log_dir=str(tmp_path))
    handler = TaskLogHandler(logger)
    handler.setLevel(logging.DEBUG)
    return handler


def _read_log(handler: TaskLogHandler) -> str:
    handler.task_logger._file_handler.flush()
    with handler.task_logger.log_path.open("r", encoding="utf-8") as file:
        return file.read()


def _make_record(message: str, *, name: str = "hud.test", **extra: object) -> logging.LogRecord:
    record = logging.LogRecord(name, logging.INFO, __file__, 0, message, args=(), exc_info=None)
    for key, value in extra.items():
        setattr(record, key, value)
    return record


def test_handler_accepts_matching_tag(tmp_path: Path) -> None:
    handler = _create_handler(tmp_path, "task-a")
    try:
        handler.emit(_make_record("first", hud_task_run_id="task-a"))
        contents = _read_log(handler)
        assert "first" in contents

        handler.emit(_make_record("second", hud_task_run_id="task-b"))
        contents = _read_log(handler)
        assert "second" not in contents
    finally:
        handler.task_logger.cleanup()


def test_handler_uses_context_logger_when_tag_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    handler = _create_handler(tmp_path, "task-b")

    monkeypatch.setattr(otel_context, "get_current_task_logger", lambda: handler.task_logger)
    monkeypatch.setattr(otel_context, "get_current_task_run_id", lambda: handler.task_run_id)

    try:
        handler.emit(_make_record("context message"))
        contents = _read_log(handler)
        assert "context message" in contents
    finally:
        handler.task_logger.cleanup()


def test_handler_rejects_mismatched_context(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    handler = _create_handler(tmp_path, "task-c")

    monkeypatch.setattr(otel_context, "get_current_task_logger", lambda: None)
    monkeypatch.setattr(otel_context, "get_current_task_run_id", lambda: "other-task")

    try:
        handler.emit(_make_record("ignored"))
        contents = _read_log(handler)
        assert contents == ""
    finally:
        handler.task_logger.cleanup()


def test_task_run_stamp_filter_sets_attribute() -> None:
    filter_ = otel_context._TaskRunStampFilter("task-d")
    record = _make_record("filtered")

    assert getattr(record, "hud_task_run_id", None) is None
    assert filter_.filter(record) is True
    assert record.hud_task_run_id == "task-d"

    # Existing tag should not be overridden
    record_with_tag = _make_record("existing", hud_task_run_id="task-e")
    assert filter_.filter(record_with_tag) is True
    assert record_with_tag.hud_task_run_id == "task-e"
