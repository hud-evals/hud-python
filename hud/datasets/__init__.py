"""HUD datasets module.

Provides data models, utilities, and execution functions for working with HUD datasets.
"""

# Data models
# Execution functions
from __future__ import annotations

from .utils import (
    SingleTaskRequest,
    BatchRequest,
    submit_rollouts,
    calculate_group_stats,
    display_results,
)
from .runner import run_dataset, run_single_task, run_tasks

from hud.types import Task
from hud.utils.tasks import save_tasks

__all__ = [
    # Backwards compatibility
    "Task",
    "save_tasks",
    "run_dataset",  # deprecated, use run_tasks
    # Execution
    "run_single_task",
    "run_tasks",
    # Request schemas
    "SingleTaskRequest",
    "BatchRequest",
    "submit_rollouts",
    # Utils
    "calculate_group_stats",
    "display_results",
]