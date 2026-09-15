from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from hud.cli.eval import find_tasks_file


@patch("pathlib.Path.cwd")
def test_find_tasks_file_with_arg(mock_cwd):
    assert find_tasks_file("some/path.json") == "some/path.json"
    mock_cwd.assert_not_called()


@patch("pathlib.Path.cwd")
def test_find_tasks_file_no_files(mock_cwd):
    mock_path = MagicMock(spec=Path)
    mock_path.glob.return_value = []
    mock_cwd.return_value = mock_path

    with pytest.raises(FileNotFoundError, match="No task JSON or JSONL files found"):
        find_tasks_file(None)


@patch("hud.cli.eval.hud_console")
@patch("pathlib.Path.cwd")
def test_find_tasks_file_single_file(mock_cwd, mock_console):
    mock_path = MagicMock(spec=Path)
    mock_file = MagicMock(spec=Path)
    mock_file.name = "test.json"

    def glob_side_effect(pattern):
        if pattern == "*.json":
            return [mock_file]
        return []

    mock_path.glob.side_effect = glob_side_effect
    mock_cwd.return_value = mock_path

    result = find_tasks_file(None)
    assert result == "test.json"
    mock_console.select.assert_not_called()


@patch("hud.cli.eval.hud_console")
@patch("pathlib.Path.cwd")
def test_find_tasks_file_multiple_files(mock_cwd, mock_console):
    mock_path = MagicMock(spec=Path)
    mock_file1 = MagicMock(spec=Path)
    mock_file1.name = "test1.json"
    mock_file2 = MagicMock(spec=Path)
    mock_file2.name = "test2.jsonl"

    def glob_side_effect(pattern):
        if pattern == "*.json":
            return [mock_file1]
        if pattern == "*.jsonl":
            return [mock_file2]
        return []

    mock_path.glob.side_effect = glob_side_effect
    mock_cwd.return_value = mock_path
    mock_console.select.return_value = "test2.jsonl"

    result = find_tasks_file(None)

    assert result == "test2.jsonl"
    mock_console.select.assert_called_once()
    call_args = mock_console.select.call_args
    assert call_args[0][0] == "Select a tasks file"
    assert "test1.json" in call_args[1]["choices"]
    assert "test2.jsonl" in call_args[1]["choices"]
