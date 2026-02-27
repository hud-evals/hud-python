"""Task loading utilities for HUD.

Unified interface for loading evaluation tasks from:
- HUD API (v5 format)
- Local JSON/JSONL files (v4 LegacyTask format, auto-converted)
"""

from __future__ import annotations

import json
import logging
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any, overload

import httpx

from hud.settings import settings

if TYPE_CHECKING:
    from hud.eval.task import Task

logger = logging.getLogger(__name__)

__all__ = ["load_dataset", "load_tasks", "save_tasks"]


def _load_raw_from_file(path: Path) -> list[dict[str, Any]]:
    """Load raw task dicts from a local JSON or JSONL file."""
    raw_items: list[dict[str, Any]] = []

    if path.suffix == ".jsonl":
        # JSONL: one task per line
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                # Handle case where line contains a list
                if isinstance(item, list):
                    raw_items.extend(i for i in item if isinstance(i, dict))
                elif isinstance(item, dict):
                    raw_items.append(item)
                else:
                    raise ValueError(
                        f"Invalid JSONL format: expected dict or list, got {type(item)}"
                    )
    else:
        # JSON: array of tasks
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            raw_items = [item for item in data if isinstance(item, dict)]
        elif isinstance(data, dict):
            raw_items = [data]
        else:
            raise ValueError(f"JSON file must contain an array or object, got {type(data)}")

    return raw_items


def _load_from_file(path: Path) -> list[Task]:
    """Load tasks from a local JSON or JSONL file."""
    from hud.eval.task import Task

    raw_items = _load_raw_from_file(path)
    # Default args to {} for runnable tasks (None = template)
    return [Task(**{**item, "args": item.get("args") or {}}) for item in raw_items]


def _load_raw_from_api(dataset_name: str) -> tuple[list[dict[str, Any]], str | None]:
    """Load raw task dicts from HUD API.

    Returns (tasks, taskset_id) tuple.
    """
    from hud.datasets.utils import _normalize_task_dict

    headers = {}
    if settings.api_key:
        headers["Authorization"] = f"Bearer {settings.api_key}"

    with httpx.Client() as client:
        response = client.get(
            f"{settings.hud_api_url}/tasks/evalset/{dataset_name}",
            headers=headers,
            params={"all": "true"},
        )
        response.raise_for_status()
        data = response.json()

        taskset_id = data.get("evalset_id")
        tasks_dict = data.get("tasks", {})

        tasks = [
            _normalize_task_dict(task_data)
            for task_data in tasks_dict.values()
            if isinstance(task_data, dict)
        ]
        return tasks, taskset_id


def _load_from_api(dataset_name: str) -> tuple[list[Task], str | None]:
    """Load tasks from HUD API.

    Returns (tasks, taskset_id) tuple.
    """
    from hud.eval.task import Task

    raw_items, taskset_id = _load_raw_from_api(dataset_name)
    tasks = [Task(**{**item, "args": item.get("args") or {}}) for item in raw_items]
    return tasks, taskset_id


@overload
def load_tasks(source: str, *, raw: bool = False) -> list[Task]: ...


@overload
def load_tasks(source: str, *, raw: bool = True) -> list[dict[str, Any]]: ...


def load_tasks(source: str, *, raw: bool = False) -> list[Task] | list[dict[str, Any]]:
    """Load tasks from a source.

    Supports multiple sources with auto-detection:
    - Local file path (JSON or JSONL)
    - HUD API dataset slug (e.g., "hud-evals/SheetBench-50")

    Automatically detects and converts v4 LegacyTask format to v5 Task.

    Args:
        source: Task source. Can be:
            - Path to a local JSON/JSONL file
            - HUD API dataset slug (e.g., "hud-evals/SheetBench-50")
        raw: If True, return raw dicts without validation or env var substitution.
            Useful for preserving template strings like "${HUD_API_KEY}".

    Returns:
        - If raw=False (default): list[Task] ready to use with hud.eval()
        - If raw=True: list[dict] with raw task data

    Raises:
        ValueError: If task loading fails
    """
    # Check if it's a local file
    path = Path(source)
    if path.exists() and path.suffix in {".json", ".jsonl"}:
        logger.info("Loading tasks from file: %s", source)
        items = _load_raw_from_file(path) if raw else _load_from_file(path)
        logger.info("Loaded %d tasks from %s", len(items), source)
        return items

    # Try HUD API
    logger.info("Trying HUD API: %s", source)
    if raw:
        items, _ = _load_raw_from_api(source)
    else:
        items, _ = _load_from_api(source)
    logger.info("Loaded %d tasks from HUD API: %s", len(items), source)
    return items


def save_tasks(
    name: str,
    tasks: list[Task],
) -> str:
    """Save tasks to the HUD API.

    Creates or updates a taskset with the given tasks.

    Args:
        name: Taskset name/slug (e.g., "my-evals/benchmark-v1").
            If no org prefix, uses user's default org.
        tasks: List of Task objects (v5 format) to save.

    Returns:
        The taskset ID of the created/updated taskset.

    Example:
        ```python
        from hud.datasets import save_tasks, load_tasks
        from hud.eval.task import Task
        from hud.environment import Environment

        # Create tasks
        env = Environment("my-env")
        tasks = [
            Task(env=env, scenario="checkout", args={"user": "alice"}),
            Task(env=env, scenario="checkout", args={"user": "bob"}),
        ]

        # Save to HUD API
        taskset_id = save_tasks("my-evals/benchmark-v1", tasks)

        # Later, load them back
        loaded = load_tasks("my-evals/benchmark-v1")
        ```

    Raises:
        TypeError: If any task is not a v5 Task object (must have 'scenario')
        ValueError: If API key is not set or save fails
    """
    if not settings.api_key:
        raise ValueError("HUD_API_KEY is required to save tasks")

    # Validate all tasks are v5 format (must have 'scenario')
    for i, task in enumerate(tasks):
        if not hasattr(task, "scenario"):
            raise TypeError(
                f"Task at index {i} is missing 'scenario' - only v5 Task objects can be saved. "
                "Use Task.from_v4(legacy_task) to convert from LegacyTask."
            )

    # Convert tasks to dicts (Task is a Pydantic model).
    # id is internal/platform-assigned; uploads should identify via slug.
    task_dicts: list[dict[str, Any]] = []
    for task in tasks:
        task_data = task.model_dump(mode="json", exclude_none=True)
        task_data.pop("id", None)
        task_dicts.append(task_data)

    # Build request payload
    payload: dict[str, Any] = {
        "name": name,
        "tasks": task_dicts,
    }

    headers = {"Authorization": f"Bearer {settings.api_key}"}

    try:
        with httpx.Client(timeout=60) as client:
            response = client.post(
                f"{settings.hud_api_url}/tasks/upload",
                json=payload,
                headers=headers,
            )
            response.raise_for_status()
            data = response.json()
            taskset_id = data.get("evalset_id") or data.get("id") or name
            logger.info("Saved %d tasks to taskset: %s", len(tasks), taskset_id)
            return taskset_id
    except httpx.HTTPStatusError as e:
        raise ValueError(f"Failed to save tasks: {e.response.text}") from e
    except Exception as e:
        raise ValueError(f"Failed to save tasks: {e}") from e


# Deprecated alias for backwards compatibility
def load_dataset(source: str, *, raw: bool = False) -> list[Task] | list[dict[str, Any]]:
    """Deprecated: Use load_tasks() instead.

    .. deprecated:: 0.6.0
        load_dataset() is deprecated. Use load_tasks() instead.
    """
    warnings.warn(
        "load_dataset() is deprecated. Use load_tasks() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return load_tasks(source, raw=raw)
