"""HUD datasets module.

Provides data models, utilities, and execution functions for working with HUD datasets.
"""

# Data models
# Execution functions
from __future__ import annotations

from .runner import run_dataset

from hud.types import Task
from hud.utils.tasks import save_tasks

__all__ = [
    # Backwards compatibility
    "Task",
    "save_tasks",
    # Execution
    "run_dataset",
]