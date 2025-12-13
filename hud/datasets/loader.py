"""Dataset loading utilities for HUD.

Unified interface for loading evaluation datasets from:
- HUD API (v5 format)
- Local JSON/JSONL files (v4 LegacyTask format, auto-converted)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from hud.eval.task import Task

logger = logging.getLogger(__name__)

__all__ = ["load_dataset"]


def _is_legacy_task_format(item: dict[str, Any]) -> bool:
    """Check if a dict is in v4 LegacyTask format.

    LegacyTask has: prompt, mcp_config (required), setup_tool, evaluate_tool (optional)
    v5 Task has: env, scenario, args
    """
    # If it has prompt + mcp_config, it's legacy format
    # If it has setup_tool or evaluate_tool, it's legacy
    return (
        ("prompt" in item and "mcp_config" in item)
        or "setup_tool" in item
        or "evaluate_tool" in item
    )


def _task_from_dict(item: dict[str, Any]) -> Task:
    """Convert a dict to Task, auto-detecting v4 vs v5 format."""
    from hud.eval.task import Task
    from hud.types import MCPToolCall

    if _is_legacy_task_format(item):
        # v4 LegacyTask format - convert via Task.from_v4()
        return Task.from_v4(item)
    else:
        # v5 format - env is EnvConfig dict with name, include, exclude
        # Convert validation dicts to MCPToolCall objects
        validation = None
        if item.get("validation"):
            validation = [MCPToolCall(**v) for v in item["validation"]]

        return Task(
            id=item.get("id"),
            env=item.get("env"),  # EnvConfig dict: {"name": "browser", "include": [...], ...}
            scenario=item.get("scenario"),
            args=item.get("args", {}),
            validation=validation,
        )


def _load_from_file(path: Path) -> list[Task]:
    """Load tasks from a local JSON or JSONL file."""
    tasks: list[Task] = []

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
                    tasks.extend(_task_from_dict(i) for i in item)
                elif isinstance(item, dict):
                    tasks.append(_task_from_dict(item))
                else:
                    raise ValueError(
                        f"Invalid JSONL format: expected dict or list, got {type(item)}"
                    )
    else:
        # JSON: array of tasks
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            tasks = [_task_from_dict(item) for item in data]
        elif isinstance(data, dict):
            tasks = [_task_from_dict(data)]
        else:
            raise ValueError(f"JSON file must contain an array or object, got {type(data)}")

    return tasks


def _load_from_api(dataset_name: str) -> list[Task]:
    """Load tasks from HUD API."""
    import httpx

    from hud.settings import settings

    headers = {}
    if settings.api_key:
        headers["Authorization"] = f"Bearer {settings.api_key}"

    with httpx.Client() as client:
        response = client.get(
            f"{settings.hud_api_url}/evals/{dataset_name}",
            headers=headers,
            params={"all": "true"},
        )
        response.raise_for_status()
        data = response.json()

        # Extract tasks dict from response
        tasks_dict = data.get("tasks", {})

        tasks: list[Task] = []
        for task_id, task_data in tasks_dict.items():
            if task_data.get("id") is None:
                task_data["id"] = task_id
            tasks.append(_task_from_dict(task_data))

        return tasks


def load_dataset(source: str) -> list[Task]:
    """Load tasks from a dataset source.

    Supports multiple sources with auto-detection:
    - Local file path (JSON or JSONL)
    - HUD API dataset slug (e.g., "hud-evals/SheetBench-50")

    Automatically detects and converts v4 LegacyTask format to v5 Task.

    Args:
        source: Dataset source. Can be:
            - Path to a local JSON/JSONL file
            - HUD API dataset slug (e.g., "hud-evals/SheetBench-50")

    Returns:
        List of Task objects ready to use with hud.eval()

    Example:
        ```python
        import hud
        from hud.datasets import load_dataset

        # Load from HUD API
        tasks = load_dataset("hud-evals/SheetBench-50")

        # Load from local file (v4 format auto-converted)
        tasks = load_dataset("./my-tasks.json")

        # Run evaluation
        async with hud.eval(tasks, variants={"model": ["gpt-4o"]}, group=4) as ctx:
            await agent.run(ctx)
        ```

    Raises:
        ValueError: If dataset loading fails
    """
    # Check if it's a local file
    path = Path(source)
    if path.exists() and path.suffix in {".json", ".jsonl"}:
        logger.info("Loading tasks from file: %s", source)
        tasks = _load_from_file(path)
        logger.info("Loaded %d tasks from %s", len(tasks), source)
        return tasks

    # Otherwise, try HUD API
    logger.info("Loading dataset from HUD API: %s", source)
    try:
        tasks = _load_from_api(source)
        logger.info("Loaded %d tasks from %s", len(tasks), source)
        return tasks
    except Exception as e:
        raise ValueError(f"Failed to load dataset '{source}' from HUD API: {e}") from e
