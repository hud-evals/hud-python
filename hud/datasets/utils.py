"""Dataset utilities for loading, saving, and fetching datasets."""

from __future__ import annotations

import json
import logging
from typing import Any

from datasets import Dataset

from hud.types import Task

logger = logging.getLogger("hud.datasets")

def save_tasks(
    tasks: list[dict[str, Any]], repo_id: str, fields: list[str] | None = None, **kwargs: Any
) -> None:
    """
    Save data to HuggingFace dataset with JSON string serialization.

    Complex fields (dicts, lists) are serialized as JSON strings to maintain clean schema
    and avoid null value pollution in HuggingFace datasets.

    Args:
        tasks: List of dictionaries to save
        repo_id: HuggingFace repository ID (e.g., "hud-evals/my-tasks")
        fields: Optional list of fields to save. If None, saves all fields from each dict.
        **kwargs: Additional arguments passed to dataset.push_to_hub()
    """
    # Safety check: Ensure we're not saving Task objects (which have resolved env vars)
    if tasks and isinstance(tasks[0], Task):
        raise ValueError(
            "save_tasks expects dictionaries, not Task objects. "
            "Task objects have resolved environment variables which would expose secrets. "
            "Please pass raw dictionaries with template strings like '${HUD_API_KEY}' preserved."
        )

    # Convert to rows with JSON string fields
    data = []
    for i, tc_dict in enumerate(tasks):
        # Additional safety check for each item
        if isinstance(tc_dict, Task):
            raise ValueError(
                f"Item {i} is a Task object, not a dictionary. "
                "This would expose resolved environment variables. "
                "Please convert to dictionary format with template strings preserved."
            )

        row = {}

        # Determine which fields to process
        fields_to_process = fields if fields is not None else list(tc_dict.keys())

        for field in fields_to_process:
            if field in tc_dict:
                value = tc_dict[field]
                # Serialize complex types as JSON strings
                if isinstance(value, (dict | list)):
                    row[field] = json.dumps(value)
                elif isinstance(value, (str | int | float | bool | type(None))):
                    row[field] = value if value is not None else ""
                else:
                    # For other types, convert to string
                    row[field] = str(value)

        data.append(row)

    # Create and push dataset
    dataset = Dataset.from_list(data)
    dataset.push_to_hub(repo_id, **kwargs)
