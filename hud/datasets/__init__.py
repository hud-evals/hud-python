"""HUD datasets module.

Provides unified dataset loading and execution for HUD evaluations.

Key functions:
- load_dataset(): Load tasks from JSON, JSONL, HuggingFace, or HUD API
- run_dataset(): Run an agent on a dataset of tasks
- submit_rollouts(): Submit tasks for remote execution

Supports both v4 (LegacyTask) and v5 (Task) formats with automatic conversion.
"""

from __future__ import annotations

from hud.eval.display import display_results
from hud.utils.tasks import save_tasks

from .loader import load_dataset
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
    "load_dataset",
    "run_dataset",
    "run_single_task",
    "save_tasks",
    "submit_rollouts",
]
