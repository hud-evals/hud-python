from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from hud.types import Task
from hud.utils.tasks import load_tasks


def test_load_tasks_from_list():
    """Test loading tasks from a list of dictionaries."""
    task_dicts = [
        {"id": "1", "prompt": "Test task 1", "mcp_config": {}},
        {"id": "2", "prompt": "Test task 2", "mcp_config": {}},
    ]

    tasks = load_tasks(task_dicts)

    assert len(tasks) == 2
    assert all(isinstance(t, Task) for t in tasks)
    assert tasks[0].prompt == "Test task 1"  # type: ignore
    assert tasks[1].prompt == "Test task 2"  # type: ignore


def test_load_tasks_from_list_raw():
    """Test loading tasks from a list in raw mode."""
    task_dicts = [
        {"id": "1", "prompt": "Test task 1", "mcp_config": {}},
        {"id": "2", "prompt": "Test task 2", "mcp_config": {}},
    ]

    tasks = load_tasks(task_dicts, raw=True)

    assert len(tasks) == 2
    assert all(isinstance(t, dict) for t in tasks)
    assert tasks[0]["prompt"] == "Test task 1"  # type: ignore


def test_load_tasks_from_json_file():
    """Test loading tasks from a JSON file."""
    task_dicts = [
        {"id": "1", "prompt": "Test task 1", "mcp_config": {}},
        {"id": "2", "prompt": "Test task 2", "mcp_config": {}},
    ]

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
        json.dump(task_dicts, f)
        temp_path = f.name

    try:
        tasks = load_tasks(temp_path)

        assert len(tasks) == 2
        assert all(isinstance(t, Task) for t in tasks)
        assert tasks[0].prompt == "Test task 1"  # type: ignore
    finally:
        Path(temp_path).unlink()


def test_load_tasks_from_json_file_raw():
    """Test loading tasks from a JSON file in raw mode."""
    task_dicts = [
        {"id": "1", "prompt": "Test task 1", "mcp_config": {}},
        {"id": "2", "prompt": "Test task 2", "mcp_config": {}},
    ]

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
        json.dump(task_dicts, f)
        temp_path = f.name

    try:
        tasks = load_tasks(temp_path, raw=True)

        assert len(tasks) == 2
        assert all(isinstance(t, dict) for t in tasks)
    finally:
        Path(temp_path).unlink()


def test_load_tasks_from_jsonl_file():
    """Test loading tasks from a JSONL file."""
    task_dicts = [
        {"id": "1", "prompt": "Test task 1", "mcp_config": {}},
        {"id": "2", "prompt": "Test task 2", "mcp_config": {}},
    ]

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
    ) as f:
        for task_dict in task_dicts:
            f.write(json.dumps(task_dict) + "\n")
        temp_path = f.name

    try:
        tasks = load_tasks(temp_path)

        assert len(tasks) == 2
        assert all(isinstance(t, Task) for t in tasks)
        assert tasks[0].prompt == "Test task 1"  # type: ignore
    finally:
        Path(temp_path).unlink()


def test_load_tasks_from_jsonl_file_with_empty_lines():
    """Test loading tasks from a JSONL file with empty lines."""
    task_dicts = [
        {"id": "1", "prompt": "Test task 1", "mcp_config": {}},
        {"id": "2", "prompt": "Test task 2", "mcp_config": {}},
    ]

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
    ) as f:
        f.write(json.dumps(task_dicts[0]) + "\n")
        f.write("\n")  # Empty line
        f.write(json.dumps(task_dicts[1]) + "\n")
        temp_path = f.name

    try:
        tasks = load_tasks(temp_path)

        assert len(tasks) == 2
        assert all(isinstance(t, Task) for t in tasks)
    finally:
        Path(temp_path).unlink()


def test_load_tasks_from_jsonl_file_with_list():
    """Test loading tasks from a JSONL file where a line contains a list."""
    task_dict = {"id": "1", "prompt": "Test task 1", "mcp_config": {}}

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
    ) as f:
        f.write(json.dumps([task_dict, task_dict]) + "\n")
        temp_path = f.name

    try:
        tasks = load_tasks(temp_path)

        assert len(tasks) == 2
        assert all(isinstance(t, Task) for t in tasks)
    finally:
        Path(temp_path).unlink()


def test_load_tasks_json_not_array_error():
    """Test that loading from JSON file with non-array raises error."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
        json.dump({"not": "an array"}, f)
        temp_path = f.name

    try:
        with pytest.raises(ValueError, match="JSON file must contain an array"):
            load_tasks(temp_path)
    finally:
        Path(temp_path).unlink()


def test_load_tasks_invalid_jsonl_format():
    """Test that loading from JSONL with invalid format raises error."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
    ) as f:
        f.write(json.dumps("invalid") + "\n")
        temp_path = f.name

    try:
        with pytest.raises(ValueError, match="Invalid JSONL format"):
            load_tasks(temp_path)
    finally:
        Path(temp_path).unlink()


def test_load_tasks_invalid_input_type():
    """Test that invalid input type raises TypeError."""
    with pytest.raises(TypeError, match="tasks_input must be str or list"):
        load_tasks(123)  # type: ignore


def test_load_tasks_nonexistent_file():
    """Test that loading from nonexistent file raises error."""
    with pytest.raises(ValueError, match="neither a file path nor a HuggingFace dataset"):
        load_tasks("nonexistent_file_without_slash")
