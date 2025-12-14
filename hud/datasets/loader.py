"""Dataset loading utilities for HUD.

Unified interface for loading evaluation datasets from:
- HUD API (v5 format)
- Local JSON/JSONL files (v4 LegacyTask format, auto-converted)
- HuggingFace datasets (v4 LegacyTask format, auto-converted)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, overload

if TYPE_CHECKING:
    from hud.eval.task import Task

logger = logging.getLogger(__name__)

__all__ = ["load_dataset"]


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
    return [Task(**item) for item in raw_items]


def _load_raw_from_huggingface(dataset_name: str) -> list[dict[str, Any]]:
    """Load raw task dicts from HuggingFace dataset."""
    try:
        from datasets import load_dataset as hf_load_dataset
    except ImportError as e:
        raise ImportError(
            "Please install 'datasets' to load from HuggingFace: uv pip install datasets"
        ) from e

    # Parse dataset name and optional split
    if ":" in dataset_name:
        name, split = dataset_name.split(":", 1)
    else:
        name = dataset_name
        split = "train"  # Default split

    logger.info("Loading from HuggingFace dataset: %s (split=%s)", name, split)
    dataset = hf_load_dataset(name, split=split)

    raw_items: list[dict[str, Any]] = []
    for item in dataset:
        if not isinstance(item, dict):
            raise ValueError(f"Invalid HuggingFace dataset: expected dict, got {type(item)}")
        raw_items.append(dict(item))

    return raw_items


def _load_from_huggingface(dataset_name: str) -> list[Task]:
    """Load tasks from HuggingFace dataset."""
    raw_items = _load_raw_from_huggingface(dataset_name)
    from hud.eval.task import Task

    return [Task(**item) for item in raw_items]


def _load_raw_from_api(dataset_name: str) -> list[dict[str, Any]]:
    """Load raw task dicts from HUD API."""
    import httpx

    from hud.settings import settings

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

        # Extract tasks dict from response
        tasks_dict = data.get("tasks", {})

        raw_items: list[dict[str, Any]] = []
        for task_id, task_data in tasks_dict.items():
            if task_data.get("id") is None:
                task_data["id"] = task_id
            raw_items.append(task_data)

        return raw_items


def _load_from_api(dataset_name: str) -> list[Task]:
    """Load tasks from HUD API."""
    from hud.eval.task import Task

    raw_items = _load_raw_from_api(dataset_name)
    return [Task(**item) for item in raw_items]


@overload
def load_dataset(source: str, *, raw: bool = False) -> list[Task]: ...


@overload
def load_dataset(source: str, *, raw: bool = True) -> list[dict[str, Any]]: ...


def load_dataset(
    source: str, *, raw: bool = False
) -> list[Task] | list[dict[str, Any]]:
    """Load tasks from a dataset source.

    Supports multiple sources with auto-detection:
    - Local file path (JSON or JSONL)
    - HUD API dataset slug (e.g., "hud-evals/SheetBench-50")
    - HuggingFace dataset (e.g., "username/dataset" or "username/dataset:split")

    Automatically detects and converts v4 LegacyTask format to v5 Task.

    Args:
        source: Dataset source. Can be:
            - Path to a local JSON/JSONL file
            - HUD API dataset slug (e.g., "hud-evals/SheetBench-50")
            - HuggingFace dataset name (e.g., "hud-evals/tasks" or "hud-evals/tasks:train")
        raw: If True, return raw dicts without validation or env var substitution.
            Useful for preserving template strings like "${HUD_API_KEY}".

    Returns:
        - If raw=False (default): list[Task] ready to use with hud.eval()
        - If raw=True: list[dict] with raw task data

    Example:
        ```python
        import hud
        from hud.datasets import load_dataset

        # Load from HUD API
        tasks = load_dataset("hud-evals/SheetBench-50")

        # Load from local file (v4 format auto-converted)
        tasks = load_dataset("./my-tasks.json")

        # Load from HuggingFace
        tasks = load_dataset("hud-evals/benchmark:test")

        # Load raw dicts (preserves env var placeholders)
        raw_tasks = load_dataset("./tasks.json", raw=True)

        # Run evaluation
        async with hud.eval(tasks) as ctx:
            await agent.run(ctx)
        ```

    Raises:
        ValueError: If dataset loading fails
    """
    # Check if it's a local file
    path = Path(source)
    if path.exists() and path.suffix in {".json", ".jsonl"}:
        logger.info("Loading tasks from file: %s", source)
        items = _load_raw_from_file(path) if raw else _load_from_file(path)
        logger.info("Loaded %d tasks from %s", len(items), source)
        return items

    # Try HUD API first
    try:
        logger.info("Trying HUD API: %s", source)
        items = _load_raw_from_api(source) if raw else _load_from_api(source)
        logger.info("Loaded %d tasks from HUD API: %s", len(items), source)
        return items
    except Exception as hud_error:
        logger.debug("HUD API load failed (%s), trying HuggingFace", hud_error)

    # Try HuggingFace as fallback
    try:
        logger.info("Trying HuggingFace dataset: %s", source)
        items = _load_raw_from_huggingface(source) if raw else _load_from_huggingface(source)
        logger.info("Loaded %d tasks from HuggingFace: %s", len(items), source)
        return items
    except ImportError:
        raise ValueError(
            f"Failed to load dataset '{source}'. "
            "Install 'datasets' package for HuggingFace support."
        ) from None
    except Exception as hf_error:
        raise ValueError(f"Failed to load dataset '{source}': {hf_error}") from hf_error
