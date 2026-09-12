from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from hud.cli.utils.output import CliError, ExitCode
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


def parse_task_args(args: str) -> dict[str, Any]:
    try:
        parsed = json.loads(args or "{}")
    except json.JSONDecodeError as exc:
        raise CliError(
            error="usage",
            message=f"--args must be valid JSON: {exc}",
            input={"args": args},
            suggestion='Pass a JSON object, e.g. --args \'{"key": "value"}\'.',
            exit_code=ExitCode.USAGE,
        ) from exc
    if not isinstance(parsed, dict):
        raise CliError(
            error="usage",
            message="--args must be a JSON object",
            input={"args": args},
            exit_code=ExitCode.USAGE,
        )
    return parsed
