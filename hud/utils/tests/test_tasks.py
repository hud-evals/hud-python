from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from hud.types import Task
from hud.utils.tasks import load_tasks, save_tasks


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


def test_save_tasks_basic():
    """Test basic save_tasks functionality."""
    tasks = [
        {"id": "1", "prompt": "test", "mcp_config": {"key": "value"}},
        {"id": "2", "prompt": "test2", "mcp_config": {"key2": "value2"}},
    ]

    with patch("hud.utils.tasks.Dataset") as mock_dataset_class:
        mock_dataset = MagicMock()
        mock_dataset_class.from_list.return_value = mock_dataset

        save_tasks(tasks, "test/repo")

        mock_dataset_class.from_list.assert_called_once()
        call_args = mock_dataset_class.from_list.call_args[0][0]
        assert len(call_args) == 2
        # Check that mcp_config was JSON serialized
        assert isinstance(call_args[0]["mcp_config"], str)
        mock_dataset.push_to_hub.assert_called_once_with("test/repo")


def test_save_tasks_with_specific_fields():
    """Test save_tasks with specific fields."""
    tasks = [
        {"id": "1", "prompt": "test", "mcp_config": {"key": "value"}, "extra": "data"},
    ]

    with patch("hud.utils.tasks.Dataset") as mock_dataset_class:
        mock_dataset = MagicMock()
        mock_dataset_class.from_list.return_value = mock_dataset

        save_tasks(tasks, "test/repo", fields=["id", "prompt"])

        call_args = mock_dataset_class.from_list.call_args[0][0]
        assert "id" in call_args[0]
        assert "prompt" in call_args[0]
        assert "extra" not in call_args[0]


def test_save_tasks_with_list_field():
    """Test save_tasks serializes list fields."""
    tasks = [
        {"id": "1", "tags": ["tag1", "tag2"], "count": 5},
    ]

    with patch("hud.utils.tasks.Dataset") as mock_dataset_class:
        mock_dataset = MagicMock()
        mock_dataset_class.from_list.return_value = mock_dataset

        save_tasks(tasks, "test/repo")

        call_args = mock_dataset_class.from_list.call_args[0][0]
        # List should be JSON serialized
        assert isinstance(call_args[0]["tags"], str)
        assert '"tag1"' in call_args[0]["tags"]


def test_save_tasks_with_primitive_types():
    """Test save_tasks handles various primitive types."""
    tasks = [
        {
            "string": "text",
            "integer": 42,
            "float": 3.14,
            "boolean": True,
            "none": None,
        },
    ]

    with patch("hud.utils.tasks.Dataset") as mock_dataset_class:
        mock_dataset = MagicMock()
        mock_dataset_class.from_list.return_value = mock_dataset

        save_tasks(tasks, "test/repo")

        call_args = mock_dataset_class.from_list.call_args[0][0]
        assert call_args[0]["string"] == "text"
        assert call_args[0]["integer"] == 42
        assert call_args[0]["float"] == 3.14
        assert call_args[0]["boolean"] is True
        assert call_args[0]["none"] == ""  # None becomes empty string


def test_save_tasks_with_other_type():
    """Test save_tasks converts other types to string."""

    class CustomObj:
        def __str__(self):
            return "custom_value"

    tasks = [
        {"id": "1", "custom": CustomObj()},
    ]

    with patch("hud.utils.tasks.Dataset") as mock_dataset_class:
        mock_dataset = MagicMock()
        mock_dataset_class.from_list.return_value = mock_dataset

        save_tasks(tasks, "test/repo")

        call_args = mock_dataset_class.from_list.call_args[0][0]
        assert call_args[0]["custom"] == "custom_value"


def test_save_tasks_rejects_task_objects():
    """Test save_tasks raises error for Task objects."""
    task = Task(prompt="test", mcp_config={})

    with pytest.raises(ValueError, match="expects dictionaries, not Task objects"):
        save_tasks([task], "test/repo")  # type: ignore


def test_save_tasks_rejects_task_objects_in_list():
    """Test save_tasks raises error when Task object is in the list."""
    tasks = [
        {"id": "1", "prompt": "test", "mcp_config": {}},
        Task(prompt="test2", mcp_config={}),  # Task object
    ]

    with pytest.raises(ValueError, match="Item 1 is a Task object"):
        save_tasks(tasks, "test/repo")  # type: ignore


def test_save_tasks_with_kwargs():
    """Test save_tasks passes kwargs to push_to_hub."""
    tasks = [{"id": "1", "prompt": "test"}]

    with patch("hud.utils.tasks.Dataset") as mock_dataset_class:
        mock_dataset = MagicMock()
        mock_dataset_class.from_list.return_value = mock_dataset

        save_tasks(tasks, "test/repo", private=True, commit_message="Test commit")

        mock_dataset.push_to_hub.assert_called_once_with(
            "test/repo", private=True, commit_message="Test commit"
        )


def test_save_tasks_field_not_in_dict():
    """Test save_tasks handles missing fields gracefully."""
    tasks = [
        {"id": "1", "prompt": "test"},
    ]

    with patch("hud.utils.tasks.Dataset") as mock_dataset_class:
        mock_dataset = MagicMock()
        mock_dataset_class.from_list.return_value = mock_dataset

        # Request fields that don't exist
        save_tasks(tasks, "test/repo", fields=["id", "missing_field"])

        call_args = mock_dataset_class.from_list.call_args[0][0]
        assert "id" in call_args[0]
        assert "missing_field" not in call_args[0]


def test_save_tasks_empty_list():
    """Test save_tasks with empty list."""
    with patch("hud.utils.tasks.Dataset") as mock_dataset_class:
        mock_dataset = MagicMock()
        mock_dataset_class.from_list.return_value = mock_dataset

        save_tasks([], "test/repo")

        mock_dataset_class.from_list.assert_called_once_with([])
        mock_dataset.push_to_hub.assert_called_once()
