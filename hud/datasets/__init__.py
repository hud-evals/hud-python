"""HUD datasets module.

Provides data models, utilities, and execution functions for working with HUD datasets.
"""

# Data models
# Execution functions
from __future__ import annotations

from .runner import display_results, run_dataset, run_tasks

from hud.types import Task
from hud.utils.tasks import save_tasks

__all__ = [
    # Backwards compatibility
    "Task",
    "save_tasks",
    "run_dataset",  # deprecated, use run_tasks
    # Execution
    "run_tasks",
    "display_results",
]