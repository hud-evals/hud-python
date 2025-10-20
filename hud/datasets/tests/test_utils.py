from __future__ import annotations

from unittest.mock import MagicMock, mock_open, patch

import pytest

from hud.datasets.utils import fetch_system_prompt_from_dataset, save_tasks
from hud.types import Task


@pytest.mark.asyncio
async def test_fetch_system_prompt_success():
    """Test successful fetch of system prompt."""
    with patch("huggingface_hub.hf_hub_download") as mock_download:
        mock_download.return_value = "/tmp/system_prompt.txt"
        with patch("builtins.open", mock_open(read_data="Test system prompt")):
            result = await fetch_system_prompt_from_dataset("test/dataset")
            assert result == "Test system prompt"
            mock_download.assert_called_once()


@pytest.mark.asyncio
async def test_fetch_system_prompt_empty_file():
    """Test fetch when file is empty."""
    with patch("huggingface_hub.hf_hub_download") as mock_download:
        mock_download.return_value = "/tmp/system_prompt.txt"
        with patch("builtins.open", mock_open(read_data="  \n  ")):
            result = await fetch_system_prompt_from_dataset("test/dataset")
            assert result is None


@pytest.mark.asyncio
async def test_fetch_system_prompt_file_not_found():
    """Test fetch when file doesn't exist."""
    with patch("huggingface_hub.hf_hub_download") as mock_download:
        from huggingface_hub.errors import EntryNotFoundError

        mock_download.side_effect = EntryNotFoundError("File not found")
        result = await fetch_system_prompt_from_dataset("test/dataset")
        assert result is None


@pytest.mark.asyncio
async def test_fetch_system_prompt_import_error():
    """Test fetch when huggingface_hub is not installed."""
    # Mock the import itself to raise ImportError
    import sys

    with patch.dict(sys.modules, {"huggingface_hub": None}):
        result = await fetch_system_prompt_from_dataset("test/dataset")
        assert result is None


@pytest.mark.asyncio
async def test_fetch_system_prompt_general_exception():
    """Test fetch with general exception."""
    with patch("huggingface_hub.hf_hub_download") as mock_download:
        mock_download.side_effect = Exception("Network error")
        result = await fetch_system_prompt_from_dataset("test/dataset")
        assert result is None


def test_save_tasks_basic():
    """Test basic save_tasks functionality."""
    tasks = [
        {"id": "1", "prompt": "test", "mcp_config": {"key": "value"}},
        {"id": "2", "prompt": "test2", "mcp_config": {"key2": "value2"}},
    ]

    with patch("hud.datasets.utils.Dataset") as mock_dataset_class:
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

    with patch("hud.datasets.utils.Dataset") as mock_dataset_class:
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

    with patch("hud.datasets.utils.Dataset") as mock_dataset_class:
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

    with patch("hud.datasets.utils.Dataset") as mock_dataset_class:
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

    with patch("hud.datasets.utils.Dataset") as mock_dataset_class:
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

    with patch("hud.datasets.utils.Dataset") as mock_dataset_class:
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

    with patch("hud.datasets.utils.Dataset") as mock_dataset_class:
        mock_dataset = MagicMock()
        mock_dataset_class.from_list.return_value = mock_dataset

        # Request fields that don't exist
        save_tasks(tasks, "test/repo", fields=["id", "missing_field"])

        call_args = mock_dataset_class.from_list.call_args[0][0]
        assert "id" in call_args[0]
        assert "missing_field" not in call_args[0]


def test_save_tasks_empty_list():
    """Test save_tasks with empty list."""
    with patch("hud.datasets.utils.Dataset") as mock_dataset_class:
        mock_dataset = MagicMock()
        mock_dataset_class.from_list.return_value = mock_dataset

        save_tasks([], "test/repo")

        mock_dataset_class.from_list.assert_called_once_with([])
        mock_dataset.push_to_hub.assert_called_once()
