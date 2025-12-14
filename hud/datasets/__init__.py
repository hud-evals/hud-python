"""HUD datasets module.

Provides unified task loading, saving, and execution for HUD evaluations.

Key functions:
- load_tasks(): Load tasks from JSON, JSONL, HuggingFace, or HUD API
- save_tasks(): Save tasks to the HUD API
- run_dataset(): Run an agent on a dataset of tasks
- submit_rollouts(): Submit tasks for remote execution

Supports both v4 (LegacyTask) and v5 (Task) formats with automatic conversion.
"""

from __future__ import annotations

from hud.eval.display import display_results

from .loader import load_dataset, load_tasks, save_tasks
from .runner import run_dataset, run_single_task
from .utils import (
    BatchRequest,
    SingleTaskRequest,
    submit_rollouts,
)

__all__ = [
    "BatchRequest",
    "SingleTaskRequest",
    "display_results",
    "load_dataset",  # Deprecated alias
    "load_tasks",
    "run_dataset",
    "run_single_task",
    "save_tasks",
    "submit_rollouts",
]
