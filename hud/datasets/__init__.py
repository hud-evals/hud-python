"""HUD datasets module.

Provides data models, utilities, and execution functions for working with HUD datasets.
"""

# Data models
# Execution functions
from __future__ import annotations

from hud.types import Task
from hud.utils.tasks import save_tasks

from .runner import run_dataset, run_single_task, run_tasks
from .utils import (
    BatchRequest,
    SingleTaskRequest,
    calculate_group_stats,
    display_results,
    submit_rollouts,
)

__all__ = [
    "BatchRequest",
    "SingleTaskRequest",
    "Task",
    "calculate_group_stats",
    "display_results",
    "run_dataset",
    "run_single_task",
    "run_tasks",
    "save_tasks",
    "submit_rollouts",
]
