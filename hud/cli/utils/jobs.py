"""Platform job/rollout cancellation helpers (used by ``hud cancel``)."""

from __future__ import annotations

from typing import Any

from hud.utils.platform import PlatformClient


async def cancel_job(job_id: str) -> dict[str, Any]:
    """Cancel all tasks for a specific job.

    Returns the response with cancellation results (``total_found``, ``cancelled``).
    """
    return await PlatformClient.from_settings().apost(
        "/v1/rollouts/cancel_job",
        json={"job_id": job_id},
    )


async def cancel_task(job_id: str, trace_id: str) -> dict[str, Any]:
    """Cancel a specific task run within a job."""
    return await PlatformClient.from_settings().apost(
        "/v1/rollouts/cancel",
        json={"job_id": job_id, "trace_id": trace_id},
    )


async def cancel_all_jobs() -> dict[str, Any]:
    """Cancel ALL active jobs for the authenticated user (panic button).

    Returns the response with ``jobs_cancelled``, ``total_tasks_cancelled``, and
    ``job_details``.
    """
    return await PlatformClient.from_settings().apost("/v1/rollouts/cancel_user_jobs", json={})


__all__ = ["cancel_all_jobs", "cancel_job", "cancel_task"]
