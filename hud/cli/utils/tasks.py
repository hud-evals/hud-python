from __future__ import annotations

from pathlib import Path

from hud.utils.hud_console import hud_console


def find_tasks_file(tasks_file: str | None, msg: str = "Select a tasks file") -> str:
    """Find tasks file."""
    if tasks_file:
        return tasks_file

    # Get current directory and find all .json and .jsonl files
    current_dir = Path.cwd()
    all_files = list(current_dir.glob("*.json")) + list(current_dir.glob("*.jsonl"))
    all_files = [
        str(file).replace(str(current_dir), "").lstrip("/").lstrip("\\") for file in all_files
    ]
    all_files = [file for file in all_files if file[0] != "."]  # Remove all config files

    if not all_files:
        # No task files found - raise a clear exception
        raise FileNotFoundError("No task JSON or JSONL files found in current directory")

    if len(all_files) == 1:
        return str(all_files[0])
    else:
        # Prompt user to select a file
        return hud_console.select(msg, choices=all_files)
