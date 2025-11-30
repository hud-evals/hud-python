from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from datasets import Dataset

from hud.types import Task
from hud.utils.hud_console import HUDConsole

hud_console = HUDConsole()


def load_tasks(tasks_input: str | list[dict], *, raw: bool = False) -> list[Task] | list[dict]:
    """Load tasks from various sources.

    Args:
        tasks_input: Either:
            - Path to a JSON file (array of tasks)
            - Path to a JSONL file (one task per line)
            - HuggingFace dataset name (format: "username/dataset" or "username/dataset:split")
            - List of task dictionaries
        raw: If True, return raw dicts without validation or env substitution

    Returns:
        - If raw=False (default): list[Task]
        - If raw=True: list[dict]
    """
    tasks: list[Task] | list[dict] = []

    if isinstance(tasks_input, list):
        # Direct list of task dicts
        hud_console.info(f"Loading {len(tasks_input)} tasks from provided list")
        if raw:
            return [item for item in tasks_input if isinstance(item, dict)]
        for item in tasks_input:
            task = Task(**item)
            tasks.append(task)

    elif isinstance(tasks_input, str):
        # Check if it's a file path
        if Path(tasks_input).exists():
            file_path = Path(tasks_input)

            with open(file_path, encoding="utf-8") as f:
                # Handle JSON files (array of tasks)
                if file_path.suffix.lower() == ".json":
                    data = json.load(f)
                    if not isinstance(data, list):
                        raise ValueError(
                            f"JSON file must contain an array of tasks, got {type(data)}"
                        )
                    if raw:
                        return [item for item in data if isinstance(item, dict)]
                    for item in data:
                        task = Task(**item)
                        tasks.append(task)

                # Handle JSONL files (one task per line)
                else:
                    raw_items: list[dict] = []
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        item = json.loads(line)
                        if isinstance(item, list):
                            raw_items.extend([it for it in item if isinstance(it, dict)])
                        elif isinstance(item, dict):
                            raw_items.append(item)
                        else:
                            raise ValueError(
                                f"Invalid JSONL format: expected dict or list of dicts, got {type(item)}"  # noqa: E501
                            )
                    if raw:
                        return raw_items
                    for it in raw_items:
                        task = Task(**it)
                        tasks.append(task)

        # Check if it's a HuggingFace dataset
        elif "/" in tasks_input:
            hud_console.info(f"Loading tasks from HuggingFace dataset: {tasks_input}")
            try:
                from datasets import load_dataset

                # Parse dataset name and optional split
                if ":" in tasks_input:
                    dataset_name, split = tasks_input.split(":", 1)
                else:
                    dataset_name = tasks_input
                    split = "train"  # Default split

                dataset = load_dataset(dataset_name, split=split)

                # Convert dataset rows to Task objects
                raw_rows: list[dict] = []
                for item in dataset:
                    if not isinstance(item, dict):
                        raise ValueError(
                            f"Invalid HuggingFace dataset: expected dict, got {type(item)}"
                        )
                    if not item["mcp_config"] or not item["prompt"]:
                        raise ValueError(
                            f"Invalid HuggingFace dataset: expected mcp_config and prompt, got {item}"  # noqa: E501
                        )
                    raw_rows.append(item)
                if raw:
                    return raw_rows
                for row in raw_rows:
                    task = Task(**row)
                    tasks.append(task)

            except ImportError as e:
                raise ImportError(
                    "Please install 'datasets' to load from HuggingFace: uv pip install datasets"
                ) from e
            except Exception as e:
                raise ValueError(f"Failed to load HuggingFace dataset '{tasks_input}': {e}") from e

        else:
            raise ValueError(
                f"Invalid tasks input: '{tasks_input}' is neither a file path nor a HuggingFace dataset"  # noqa: E501
            )

    else:
        raise TypeError(f"tasks_input must be str or list, got {type(tasks_input)}")

    return tasks


def save_tasks(
    tasks: list[dict[str, Any]],
    repo_id: str,
    fields: list[str] | None = None,
    **kwargs: Any,
) -> None:
    """
    Save data to a HuggingFace dataset with JSON string serialization.

    Complex fields (dicts, lists) are serialized as JSON strings to keep schemas clean
    and avoid null-value pollution when uploaded to the Hub.

    Args:
        tasks: List of dictionaries to save.
        repo_id: HuggingFace repository ID (e.g., "hud-evals/my-tasks").
        fields: Optional subset of fields to persist. Defaults to all keys per task.
        **kwargs: Extra kwargs forwarded to `Dataset.push_to_hub`.
    """

    if tasks and isinstance(tasks[0], Task):
        raise ValueError(
            "save_tasks expects dictionaries, not Task objects. "
            "Task objects have resolved environment variables which would expose secrets. "
            "Please pass raw dictionaries with template strings like '${HUD_API_KEY}' preserved."
        )

    data: list[dict[str, Any]] = []
    for index, task_dict in enumerate(tasks):
        if isinstance(task_dict, Task):
            raise ValueError(
                f"Item {index} is a Task object, not a dictionary. "
                "This would expose resolved environment variables. "
                "Please convert to dictionary format with template strings preserved."
            )

        row: dict[str, Any] = {}
        fields_to_process = fields if fields is not None else list(task_dict.keys())

        for field in fields_to_process:
            if field not in task_dict:
                continue

            value = task_dict[field]
            if isinstance(value, (dict | list)):
                row[field] = json.dumps(value)
            elif isinstance(value, (str | int | float | bool | type(None))):
                row[field] = value if value is not None else ""
            else:
                row[field] = str(value)

        data.append(row)

    dataset = Dataset.from_list(data)
    dataset.push_to_hub(repo_id, **kwargs)
