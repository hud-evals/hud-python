"""HUD datasets module.

Provides data models, utilities, and execution functions for working with HUD datasets.
"""

# Data models
# Execution functions
from __future__ import annotations

from hud.types import Task

from .parallel import (
    calculate_optimal_workers,
    run_dataset_parallel,
    run_dataset_parallel_manual,
)
from .runner import run_dataset

# Utilities
from .utils import fetch_system_prompt_from_dataset, save_tasks

__all__ = [
    # Core data model
    "Task",
    "calculate_optimal_workers",
    # Utilities
    "fetch_system_prompt_from_dataset",
    # Execution
    "run_dataset",
    "run_dataset_parallel",
    "run_dataset_parallel_manual",
    "save_tasks",
]
