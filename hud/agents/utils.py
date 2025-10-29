from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING

from hud.otel.context import (
    _update_task_status_async,
    get_current_task_run_id,
)

if TYPE_CHECKING:
    from hud.datasets import Task


async def log_task_config_to_current_trace(task: Task) -> None:
    with contextlib.suppress(Exception):
        task_run_id = get_current_task_run_id()
        if not task_run_id:
            return

        raw_config = task.model_dump()

        await _update_task_status_async(
            task_run_id,
            "running",
            task_id=task.id,
            extra_metadata={"task_config": raw_config},
        )


async def log_agent_metadata_to_status(
    model_name: str | None = None, checkpoint_name: str | None = None
) -> None:
    """Attach agent metadata (model/checkpoint) to current trace status metadata."""
    with contextlib.suppress(Exception):
        task_run_id = get_current_task_run_id()
        if not task_run_id or (not model_name and not checkpoint_name):
            return

        agent_meta = {}
        if model_name is not None:
            agent_meta["model_name"] = model_name
        if checkpoint_name is not None:
            agent_meta["checkpoint_name"] = checkpoint_name

        await _update_task_status_async(
            task_run_id,
            "running",
            extra_metadata={"agent": agent_meta},
        )
