"""HUD datasets module.

Provides data models, utilities, and execution functions for working with HUD datasets.
"""

# Data models
# Execution functions
from __future__ import annotations

from hud.types import LegacyTask
from hud.utils.tasks import save_tasks

from .loader import load_dataset
from .runner import run_dataset, run_tasks
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
    "LegacyTask",
    "calculate_group_stats",
    "display_results",
    "load_dataset",
    "run_dataset",
    "run_tasks",
    "save_tasks",
    "submit_rollouts",
]
