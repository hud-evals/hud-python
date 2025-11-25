"""Remote evaluation execution."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Iterable, cast

import httpx
from pydantic import BaseModel, Field

import hud
from hud.settings import settings
from hud.telemetry.job import _print_job_url
from hud.types import AgentType, Task
from hud.utils.hud_console import HUDConsole

logger = logging.getLogger(__name__)
hud_console = HUDConsole()

RUN_ENDPOINT = "/v1/rollouts/run"
RUN_LIST_ENDPOINT = "/v1/rollouts/run_list"


class AgentConfig(BaseModel):
    """Settings for constructing an agent instance."""

    type: AgentType
    model: str | None = None
    allowed_tools: list[str] | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


class RolloutRequest(BaseModel):
    """Payload describing a single agent rollout."""

    job_id: str
    group_id: str | None = None
    task: dict[str, Any]
    agent: AgentConfig
    max_steps: int | None = None
    verbose: bool = False
    trace_name: str | None = None
    task_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class RolloutBatchRequest(BaseModel):
    """Request to submit multiple rollouts at once."""

    rollouts: list[RolloutRequest]


@dataclass
class SubmissionResult:
    """Result of submitting rollouts."""

    accepted: int = 0
    rejected: int = 0
    job_id: str | None = None
    job_name: str | None = None
    total_rollouts: int = 0


def chunked(sequence: list[Any], size: int) -> Iterable[list[Any]]:
    """Chunk a sequence into smaller lists."""
    return (sequence[i : i + size] for i in range(0, len(sequence), size))


async def submit_batch_payloads(
    payloads: list[RolloutRequest],
    endpoint: str,
    batch_size: int,
    client: httpx.AsyncClient,
    api_key: str,
) -> tuple[int, int]:
    """Submit rollouts in batches."""
    accepted = rejected = 0
    batches = list(chunked(payloads, batch_size))
    headers = {"Authorization": f"Bearer {api_key}"}

    with hud_console.progress(f"Submitting {len(payloads)} rollouts in {len(batches)} batches...") as progress:
        for idx, batch in enumerate(batches, 1):
            try:
                batch_request = RolloutBatchRequest(rollouts=batch)
                response = await client.post(
                    endpoint, json=batch_request.model_dump(mode="json"), headers=headers
                )
                
                if response.status_code >= 400:
                    hud_console.error(f"Batch {idx} API Error {response.status_code}: {response.text}")
                    rejected += len(batch)
                else:
                    response.raise_for_status()
                    result = response.json()
                    accepted += result.get("accepted", 0)
                    rejected += result.get("rejected", 0)
                    progress.update(f"Batch {idx}/{len(batches)}: {result.get('accepted', 0)} accepted")

            except Exception as exc:
                hud_console.error(f"Batch {idx} failed: {exc}")
                rejected += len(batch)
            
            if idx < len(batches):
                await asyncio.sleep(0.2)

    return (accepted, rejected)


async def run_remote_eval(
    source: str,
    *,
    agent_type: AgentType,
    model: str | None = None,
    allowed_tools: list[str] | None = None,
    max_steps: int = 10,
    verbose: bool = False,
    group_size: int = 1,
    task_id: str | None = None,
    full: bool = False,
    batch_size: int = 10,
) -> None:
    """Run evaluation remotely on HUD infrastructure."""
    
    # Verify API Key
    if not settings.api_key:
        hud_console.error("HUD_API_KEY is required for remote execution.")
        hud_console.info("Set it in your environment or run: hud set HUD_API_KEY=your-key-here")
        raise Exception("Missing HUD API Key")

    # Load tasks
    try:
        from hud.utils.tasks import load_tasks
    except ImportError as e:
        hud_console.error("Dataset dependencies are not installed.")
        raise e

    hud_console.info(f"📊 Loading tasks from: {source}…")
    tasks = cast(list[Task], load_tasks(source))

    if not tasks:
        hud_console.error(f"No tasks found in: {source}")
        return

    # Filter by task_id if provided
    if task_id:
        found = next((t for t in tasks if str(getattr(t, "id", "")) == str(task_id)), None)
        if not found:
            hud_console.error(f"Task with ID '{task_id}' not found in source.")
            return
        tasks = [found]
        hud_console.info(f"Found task with ID '{task_id}', running single task.")
    elif not full:
        # Default behavior: use the first task if not full
        tasks = [tasks[0]]
        hud_console.info("Running first task only (use --full to run all tasks).")

    # Determine dataset name for job
    dataset_name = source.split("/")[-1]
    
    # Create Job
    job_name = f"Remote Eval: {dataset_name} ({agent_type})"
    metadata = {
        "source": source,
        "agent_type": agent_type,
        "agent_model": model,
        "group_size": group_size,
        "max_steps": max_steps,
        "mode": "remote",
    }
    
    try:
        job = hud.create_job(job_name, metadata=metadata, dataset_link=source)
        job.update_status_sync("created")
        _print_job_url(job.id, job.name)
    except Exception as e:
        hud_console.error(f"Failed to create job: {e}")
        return

    # Prepare RolloutRequests
    rollouts: list[RolloutRequest] = []
    
    # Construct Agent Config
    agent_config = AgentConfig(
        type=agent_type,
        model=model,
        allowed_tools=allowed_tools,
    )

    for task in tasks:
        # Ensure task is dict
        task_dict = task.model_dump() if hasattr(task, "model_dump") else dict(task)
        base_task_id = str(task_dict.get("id", "task"))
        
        for i in range(group_size):
            # Generate unique task_id per rollout to avoid celery job deduplication conflicts
            # The backend dedups by celery_task_id which is {job_id}_{task_id}
            # group_id groups rollouts of same task together for analysis
            unique_task_id = f"{base_task_id}_r{i}" if group_size > 1 else base_task_id

            req = RolloutRequest(
                job_id=job.id,
                group_id=base_task_id,
                task=task_dict,
                task_id=unique_task_id,
                agent=agent_config,
                max_steps=max_steps,
                verbose=verbose,
                metadata={"rollout_index": i},
            )
            rollouts.append(req)

    # Submit
    api_url = settings.hud_api_url.rstrip("/")
    endpoint = f"{api_url}{RUN_LIST_ENDPOINT}"
    
    hud_console.info(f"🚀 Submitting {len(rollouts)} rollouts to {api_url}...")
    
    async with httpx.AsyncClient(timeout=120) as client:
        accepted, rejected = await submit_batch_payloads(
            rollouts,
            endpoint,
            batch_size=batch_size,
            client=client,
            api_key=settings.api_key,
        )

    hud_console.success(f"Submission complete! Accepted: {accepted}, Rejected: {rejected}")
    hud_console.info(f"Monitor progress at: https://hud.so/jobs/{job.id}")

