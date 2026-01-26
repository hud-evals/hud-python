"""Validate task files or datasets."""

from __future__ import annotations

from typing import Any

import typer
from pydantic import ValidationError

from hud.types import Task
from hud.utils.tasks import load_tasks
from hud.utils.hud_console import hud_console


def validate_command(source: str) -> None:
    """Validate tasks from a file or HuggingFace dataset."""
    try:
        raw_tasks = load_tasks(source, raw=True)
    except Exception as e:
        hud_console.error(f"Failed to load tasks: {e}")
        raise typer.Exit(1) from e

    errors: list[str] = []
    for idx, task in enumerate(raw_tasks):
        label = task.get("id") or f"index {idx}"
        try:
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
