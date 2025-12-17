"""Utility functions and schemas for the datasets module."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import httpx
from pydantic import BaseModel, Field, field_validator, model_validator

from hud.settings import settings
from hud.types import AgentType, TaskInput
from hud.utils.hud_console import HUDConsole

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)
hud_console = HUDConsole()

__all__ = [
    "BatchRequest",
    "SingleTaskRequest",
    "cancel_all_jobs",
    "cancel_job",
    "cancel_task",
    "submit_rollouts",
]


class SingleTaskRequest(BaseModel):
    """Request to run a single task remotely - mirrors run_single_task() args."""

    task: dict[str, Any] = Field(
        description="Task definition (v4 LegacyTask or v5 Task format).",
    )
    agent_type: AgentType = Field(description="Agent type to execute the task.")
    agent_params: dict[str, Any] = Field(
        default_factory=dict,
        description="Agent constructor parameters passed to agent.create(). "
        "Should include fields from BaseCreateParams (auto_trace, auto_respond, verbose) "
        "plus agent-specific config fields (e.g., checkpoint_name for ClaudeConfig).",
    )
    max_steps: int = Field(default=10, description="Maximum steps allowed for the agent.")
    job_id: str = Field(description="HUD job identifier for telemetry association.")
    task_id: str | None = Field(default=None, description="Task identifier.")
    trace_name: str | None = Field(default=None, description="Trace name.")
    group_id: str | None = Field(default=None, description="Optional HUD group identifier.")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata to inject into the trace context.",
    )
    trace_id: str | None = Field(default=None, description="Pre-assigned trace ID.")

    @model_validator(mode="after")
    def _validate_task(self) -> SingleTaskRequest:
        """Validate task is either v4 LegacyTask or v5 Task format."""
        from hud.eval.utils import is_v4_format, validate_v4_task

        # v4 format: looks like v4 (prompt + mcp_config)?
        if is_v4_format(self.task):
            # Validate completeness (requires evaluate_tool too)
            validate_v4_task(self.task)
            return self

        # v5 format: env required
        if "env" in self.task:
            return self

        # Neither v4 nor v5
        raise ValueError("Task must have 'env' (v5) or 'prompt'+'mcp_config'+'evaluate_tool' (v4)")

    @field_validator("job_id")
    @classmethod
    def _validate_job_id(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("job_id must be a non-empty string.")
        return value


class BatchRequest(BaseModel):
    """Request to run multiple tasks remotely."""

    requests: list[SingleTaskRequest] = Field(
        description="List of single task requests to submit.",
        min_length=1,
        max_length=1000,
    )


def _normalize_tasks(tasks: Sequence[TaskInput]) -> list[dict[str, Any]]:
    """Convert tasks to list of dicts for remote API submission."""
    result = []
    for t in tasks:
        if isinstance(t, dict):
            result.append(t)
        elif hasattr(t, "model_dump"):
            result.append(t.model_dump(mode="json"))
        else:
            raise TypeError(f"Cannot convert {type(t).__name__} to dict")
    return result


async def submit_rollouts(
    tasks: Sequence[TaskInput],
    job_id: str,
    agent_type: AgentType,
    agent_params: dict[str, Any] | None = None,
    max_steps: int = 10,
    group_size: int = 1,
    batch_size: int = 50,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Submit rollouts to the HUD platform API for remote execution (fire-and-forget).

    Args:
        tasks: List of tasks (v5 Task, v4 LegacyTask, or dicts)
        job_id: HUD job ID for telemetry grouping
        agent_type: Agent type to use for execution
        agent_params: Parameters passed to agent.create()
        max_steps: Maximum steps per rollout
        group_size: Number of rollouts per task (for variance estimation)
        batch_size: Number of rollouts per API batch request
        metadata: Additional metadata for each rollout
    """
    from hud.eval.utils import is_v4_format

    if not settings.api_key:
        raise ValueError("HUD_API_KEY is required for remote execution")

    # Convert to dicts once for uniform processing
    task_dicts = _normalize_tasks(tasks)

    # Validate v4 tasks have remote-compatible mcp_config (URL-based, not command-based)
    for i, td in enumerate(task_dicts):
        if not is_v4_format(td):
            continue  # v5 tasks use env config, no mcp_config to check
        mcp_config = td.get("mcp_config") or {}
        for server_name, server_cfg in mcp_config.items():
            is_local = (
                isinstance(server_cfg, dict)
                and "command" in server_cfg
                and not server_cfg.get("url")
            )
            if is_local:
                raise ValueError(
                    f"Remote execution requires URL-based mcp_config. "
                    f"Task {td.get('id') or i} uses local Docker config for '{server_name}'. "
                    "Convert to remote with: hud convert <tasks_file>"
                )

    # Build single task requests
    requests: list[SingleTaskRequest] = []
    for task_idx, td in enumerate(task_dicts):
        base_task_id = td.get("id") or f"task_{task_idx}"
        trace_name = td.get("prompt") or td.get("scenario") or base_task_id

        for rollout_idx in range(group_size):
            task_id = f"{base_task_id}_r{rollout_idx}" if group_size > 1 else base_task_id
            requests.append(
                SingleTaskRequest(
                    task=td,
                    agent_type=agent_type,
                    agent_params=agent_params or {},
                    max_steps=max_steps,
                    job_id=job_id,
                    task_id=task_id,
                    trace_name=trace_name,
                    group_id=base_task_id if group_size > 1 else None,
                    metadata=metadata or {},
                )
            )

    # Submit in batches
    api_url = f"{settings.hud_api_url.rstrip('/')}/v1/rollouts/run_list"
    headers = {"Authorization": f"Bearer {settings.api_key}"}

    total_accepted = 0
    total_rejected = 0

    async with httpx.AsyncClient(timeout=120) as client:
        for i in range(0, len(requests), batch_size):
            batch = requests[i : i + batch_size]
            batch_request = BatchRequest(requests=batch)

            try:
                response = await client.post(
                    api_url,
                    json=batch_request.model_dump(mode="json"),
                    headers=headers,
                )
                response.raise_for_status()
                result = response.json()

                total_accepted += result.get("accepted", 0)
                total_rejected += result.get("rejected", 0)

                for item in result.get("results", []):
                    if isinstance(item, dict) and item.get("status") == "rejected":
                        hud_console.warning(f"Task rejected: {item.get('error', 'Unknown reason')}")

                batch_num = (i // batch_size) + 1
                total_batches = (len(requests) + batch_size - 1) // batch_size
                hud_console.info(
                    f"Batch {batch_num}/{total_batches}: "
                    f"{result.get('accepted', 0)}/{len(batch)} accepted"
                )

            except httpx.HTTPStatusError as exc:
                if 400 <= exc.response.status_code < 500:
                    raise ValueError(f"Submission failed: {exc.response.text}") from exc
                hud_console.error(f"Batch submission failed: {exc.response.status_code}")
                total_rejected += len(batch)

            except Exception as exc:
                hud_console.error(f"Batch submission failed: {exc}")
                total_rejected += len(batch)

    # Log final summary
    if total_rejected > 0:
        hud_console.warning(
            f"Submitted {total_accepted}/{len(requests)} requests ({total_rejected} rejected)"
        )
    else:
        hud_console.info(f"Submitted {total_accepted}/{len(requests)} requests")


async def cancel_job(job_id: str) -> dict[str, Any]:
    """Cancel all tasks for a specific job.

    Args:
        job_id: The job ID to cancel

    Returns:
        Response with cancellation results including total_found, cancelled counts
    """
    api_url = f"{settings.hud_api_url.rstrip('/')}/v1/rollouts/cancel_job"
    headers = {"Authorization": f"Bearer {settings.api_key}"}

    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(
            api_url,
            json={"job_id": job_id},
            headers=headers,
        )
        response.raise_for_status()
        return response.json()


async def cancel_task(job_id: str, task_id: str) -> dict[str, Any]:
    """Cancel a specific task within a job.

    Args:
        job_id: The job ID
        task_id: The specific task ID to cancel

    Returns:
        Response with cancellation result
    """
    api_url = f"{settings.hud_api_url.rstrip('/')}/v1/rollouts/cancel"
    headers = {"Authorization": f"Bearer {settings.api_key}"}

    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(
            api_url,
            json={"job_id": job_id, "task_id": task_id},
            headers=headers,
        )
        response.raise_for_status()
        return response.json()


async def cancel_all_jobs() -> dict[str, Any]:
    """Cancel ALL active jobs for the authenticated user.

    This is a "panic button" to stop all running rollouts.

    Returns:
        Response with jobs_cancelled, total_tasks_cancelled, and job_details
    """
    api_url = f"{settings.hud_api_url.rstrip('/')}/v1/rollouts/cancel_user_jobs"
    headers = {"Authorization": f"Bearer {settings.api_key}"}

    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.post(
            api_url,
            json={},
            headers=headers,
        )
        response.raise_for_status()
        return response.json()
