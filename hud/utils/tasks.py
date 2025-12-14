from __future__ import annotations

import json
from typing import Any

from hud.types import LegacyTask


def save_tasks(
    tasks: list[dict[str, Any]],
    repo_id: str,
    fields: list[str] | None = None,
    **kwargs: Any,
) -> None:
    """Save data to a HuggingFace dataset with JSON string serialization.

    Complex fields (dicts, lists) are serialized as JSON strings to keep schemas clean
    and avoid null-value pollution when uploaded to the Hub.

    Args:
        tasks: List of dictionaries to save.
        repo_id: HuggingFace repository ID (e.g., "hud-evals/my-tasks").
        fields: Optional subset of fields to persist. Defaults to all keys per task.
        **kwargs: Extra kwargs forwarded to `Dataset.push_to_hub`.
    """
    if tasks and isinstance(tasks[0], LegacyTask):
        raise ValueError(
            "save_tasks expects dictionaries, not LegacyTask objects. "
            "LegacyTask objects have resolved environment variables which would expose secrets. "
            "Please pass raw dictionaries with template strings like '${HUD_API_KEY}' preserved."
        )

    data: list[dict[str, Any]] = []
    for index, task_dict in enumerate(tasks):
        if isinstance(task_dict, LegacyTask):
            raise ValueError(
                f"Item {index} is a LegacyTask object, not a dictionary. "
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

    from datasets import Dataset

    ds = Dataset.from_list(data)
    ds.push_to_hub(repo_id, **kwargs)
