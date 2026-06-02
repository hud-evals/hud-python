"""Platform job/rollout cancellation helpers (used by ``hud cancel``)."""

from __future__ import annotations

from typing import Any

import httpx

from hud.settings import settings


def _headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {settings.api_key}"}


async def cancel_job(job_id: str) -> dict[str, Any]:
    """Cancel all tasks for a specific job.

    Returns the response with cancellation results (``total_found``, ``cancelled``).
    """
    api_url = f"{settings.hud_api_url.rstrip('/')}/v1/rollouts/cancel_job"
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(api_url, json={"job_id": job_id}, headers=_headers())
        response.raise_for_status()
        return response.json()


async def cancel_task(job_id: str, trace_id: str) -> dict[str, Any]:
    """Cancel a specific task run within a job."""
    api_url = f"{settings.hud_api_url.rstrip('/')}/v1/rollouts/cancel"
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(
            api_url,
            json={"job_id": job_id, "trace_id": trace_id},
            headers=_headers(),
        )
        response.raise_for_status()
        return response.json()


async def cancel_all_jobs() -> dict[str, Any]:
    """Cancel ALL active jobs for the authenticated user (panic button).

    Returns the response with ``jobs_cancelled``, ``total_tasks_cancelled``, and
    ``job_details``.
    """
    api_url = f"{settings.hud_api_url.rstrip('/')}/v1/rollouts/cancel_user_jobs"
    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.post(api_url, json={}, headers=_headers())
        response.raise_for_status()
        return response.json()


__all__ = ["cancel_all_jobs", "cancel_job", "cancel_task"]
