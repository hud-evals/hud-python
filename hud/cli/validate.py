"""Validate task files or datasets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import typer
from pydantic import ValidationError

from hud.datasets import load_tasks
from hud.eval.utils import validate_v4_task
from hud.types import Task
from hud.utils.hud_console import hud_console


def validate_command(source: str) -> None:
    """Validate tasks from a file or HuggingFace dataset."""
    try:
        raw_tasks, type_errors = _load_raw_tasks(source)
    except Exception as e:
        hud_console.error(f"Failed to load tasks: {e}")
        raise typer.Exit(1) from e

    errors: list[str] = []
    errors.extend(type_errors)
    for idx, task in enumerate(raw_tasks):
        label = task.get("id") or f"index {idx}"
        try:
            if _looks_like_v4(task):
                validate_v4_task(task)
            Task(**_as_dict(task))
        except ValidationError as e:
            errors.append(f"{label}: {e}")
        except Exception as e:
            errors.append(f"{label}: {e}")

    if errors:
        hud_console.error(f"Found {len(errors)} invalid task(s).")
        for err in errors:
            hud_console.error(f"- {err}")
        raise typer.Exit(1)

    hud_console.success(f"Validated {len(raw_tasks)} task(s).")


def _as_dict(task: Any) -> dict[str, Any]:
    if isinstance(task, dict):
        return task
    try:
        return dict(task)
    except Exception:
        return {}


def _looks_like_v4(task: dict[str, Any]) -> bool:
    return any(
        key in task
        for key in ("prompt", "mcp_config", "evaluate_tool", "setup_tool", "integration_test_tool")
    )


def _load_raw_tasks(source: str) -> tuple[list[dict[str, Any]], list[str]]:
    path = Path(source)
    if path.exists() and path.suffix.lower() in {".json", ".jsonl"}:
        return _load_raw_from_file(path)
    return cast("list[dict[str, Any]]", load_tasks(source, raw=True)), []


def _load_raw_from_file(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    errors: list[str] = []
    items: list[dict[str, Any]] = []

    if path.suffix.lower() == ".jsonl":
        with open(path, encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as e:
                    errors.append(f"line {line_no}: invalid JSON ({e.msg})")
                    continue
                if isinstance(value, dict):
                    items.append(value)
                    continue
                if isinstance(value, list):
                    for idx, entry in enumerate(value):
                        if isinstance(entry, dict):
                            items.append(entry)
                        else:
                            entry_type = type(entry).__name__
                            errors.append(
                                f"line {line_no} item {idx}: expected object, got {entry_type}"
                            )
                    continue
                errors.append(
                    f"line {line_no}: expected object or list, got {type(value).__name__}"
                )
        return items, errors

    with open(path, encoding="utf-8") as f:
        value = json.load(f)

    if isinstance(value, dict):
        return [value], errors
    if isinstance(value, list):
        for idx, entry in enumerate(value):
            if isinstance(entry, dict):
                items.append(entry)
            else:
                errors.append(f"index {idx}: expected object, got {type(entry).__name__}")
        return items, errors

    raise ValueError(f"JSON file must contain an object or array, got {type(value).__name__}")
