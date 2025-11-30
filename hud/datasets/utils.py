"""Utility functions and schemas for the datasets module."""

from __future__ import annotations

import logging
from statistics import mean, stdev
from typing import Any

import httpx
import numpy as np
from pydantic import BaseModel, Field, field_validator, model_validator

from hud.settings import settings
from hud.types import AgentType, Task, Trace
from hud.utils.hud_console import HUDConsole

logger = logging.getLogger(__name__)
hud_console = HUDConsole()


class SingleTaskRequest(BaseModel):
    """Request to run a single task remotely - mirrors run_single_task() args."""

    task: dict[str, Any] = Field(
        description="Task definition compatible with hud.types.Task.",
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
    task_id: str = Field(description="Task identifier.")
    trace_name: str = Field(description="Trace name.")
    group_id: str | None = Field(default=None, description="Optional HUD group identifier.")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata to inject into the trace context.",
    )

    @model_validator(mode="after")
    def _validate_task(self) -> SingleTaskRequest:
        try:
            Task(**self.task)
        except Exception as exc:
            raise ValueError(f"Invalid task payload: {exc}") from exc
        return self

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


async def submit_rollouts(
    tasks: list[Task],
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
        tasks: List of Task objects to execute
        job_id: HUD job ID for telemetry grouping
        agent_type: Agent type to use for execution
        agent_params: Parameters passed to agent.create(). Should include fields
            from BaseCreateParams (auto_trace, auto_respond, verbose) plus
            agent-specific config fields (e.g., checkpoint_name for ClaudeConfig).
        max_steps: Maximum steps per rollout
        group_size: Number of rollouts per task (for variance estimation)
        batch_size: Number of rollouts per API batch request
        metadata: Additional metadata for each rollout
    """
    if not settings.api_key:
        raise ValueError("HUD_API_KEY is required for remote execution")

    # Validate tasks have remote-compatible mcp_config (URL-based, not command-based)
    local_task_servers: list[tuple[int, str, str]] = []  # (task_idx, task_id, server_name)
    affected_task_indices: set[int] = set()
    for i, task in enumerate(tasks):
        if task.mcp_config:
            for server_name, server_cfg in task.mcp_config.items():
                if (
                    isinstance(server_cfg, dict)
                    and "command" in server_cfg
                    and not server_cfg.get("url")
                ):
                    local_task_servers.append((i, task.id or f"task_{i}", server_name))
                    affected_task_indices.add(i)

    if local_task_servers:
        task_details = ", ".join(f"{tid} ({srv})" for _, tid, srv in local_task_servers[:3])
        if len(local_task_servers) > 3:
            task_details += f", ... and {len(local_task_servers) - 3} more"
        raise ValueError(
            f"Remote execution requires URL-based mcp_config, but "
            f"{len(affected_task_indices)} task(s) use local Docker configs "
            f"(command-based): {task_details}. "
            "Convert to remote with: hud convert <tasks_file>"
        )

    # Build single task requests
    requests: list[SingleTaskRequest] = []
    for task_idx, task in enumerate(tasks):
        base_task_id = task.id or f"task_{task_idx}"
        for rollout_idx in range(group_size):
            task_id = f"{base_task_id}_r{rollout_idx}" if group_size > 1 else base_task_id
            requests.append(
                SingleTaskRequest(
                    task=task.model_dump(mode="json"),
                    agent_type=agent_type,
                    agent_params=agent_params or {},
                    max_steps=max_steps,
                    job_id=job_id,
                    task_id=task_id,
                    trace_name=task.prompt or task_id,
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


def calculate_group_stats(
    tasks: list[Task],
    traces: list[Trace | None],
    group_size: int,
    group_ids: dict[int, str],
) -> list[dict[str, Any]]:
    """Calculate statistics for each task group.

    Args:
        tasks: List of Task objects
        traces: List of Trace results (may contain None for failed tasks)
        group_size: Number of runs per task
        group_ids: Mapping from task index to group ID

    Returns:
        List of statistics dicts, one per task, containing:
        - task_id, prompt, group_id, group_size
        - rewards: list of individual rewards
        - mean_reward, std_reward, min_reward, max_reward
        - success_rate, error_rate
        - traces: list of Trace objects for this group
    """
    stats = []

    for task_idx, task in enumerate(tasks):
        # Get traces for this task
        start = task_idx * group_size
        task_traces = [t for t in traces[start : start + group_size] if t is not None]

        if not task_traces:
            stats.append(
                {
                    "task_id": task.id or f"task_{task_idx}",
                    "prompt": task.prompt or "",
                    "group_id": group_ids[task_idx],
                    "group_size": group_size,
                    "rewards": [],
                    "mean_reward": 0.0,
                    "std_reward": 0.0,
                    "success_rate": 0.0,
                    "error_rate": 1.0,
                }
            )
            continue

        rewards = np.array([t.reward for t in task_traces])
        errors = [t for t in task_traces if t.isError]

        task_stats = {
            "task_id": task.id or f"task_{task_idx}",
            "prompt": task.prompt or "",
            "group_id": group_ids[task_idx],
            "group_size": group_size,
            "rewards": rewards.tolist(),
            "mean_reward": float(np.mean(rewards)),
            "std_reward": float(np.std(rewards)) if len(rewards) > 1 else 0.0,
            "min_reward": float(np.min(rewards)),
            "max_reward": float(np.max(rewards)),
            "success_rate": float(np.sum(rewards > 0) / len(rewards)),
            "error_rate": len(errors) / len(task_traces),
            "traces": task_traces,
        }
        stats.append(task_stats)

    return stats


def display_results(
    results: list[Any],
    *,
    tasks: list[Task],
    elapsed: float | None = None,
    show_details: bool = True,
) -> None:
    """Display evaluation results in a formatted table.

    Args:
        results: List of Trace objects or grouped statistics dicts
        tasks: List of Task objects corresponding to results
        elapsed: Optional elapsed time in seconds
        show_details: Whether to show per-task details table
    """
    from rich.console import Console
    from rich.table import Table

    from hud.utils.hud_console import HUDConsole

    hud_console = HUDConsole()
    console = Console()

    if not results:
        hud_console.warning("No results to display")
        return

    # Detect if this is grouped results (list of dicts with 'mean_reward') or traces
    is_grouped = isinstance(results[0], dict) and "mean_reward" in results[0]

    if is_grouped:
        # Grouped evaluation stats
        all_means = [s["mean_reward"] for s in results]
        overall_mean = mean(all_means) if all_means else 0.0
        overall_std = stdev(all_means) if len(all_means) > 1 else 0.0
        group_size = results[0].get("group_size", 1)
        total_episodes = sum(len(s.get("rewards", [])) for s in results)

        hud_console.success("\n📊 Evaluation Complete")
        hud_console.info(f"Tasks: {len(results)} x {group_size} runs = {total_episodes} episodes")
        if elapsed:
            hud_console.info(f"Time: {elapsed:.1f}s ({total_episodes / elapsed:.1f} episodes/s)")
        hud_console.info(f"Mean reward: {overall_mean:.3f} ± {overall_std:.3f}")

        if show_details and len(results) <= 50:
            table = Table(title="\nPer-Task Performance")
            table.add_column("#", style="dim", justify="right")
            table.add_column("Task ID", style="cyan", no_wrap=True)
            table.add_column("Prompt", style="dim", max_width=40)
            table.add_column("Mean±Std", justify="right", style="green")
            table.add_column("Min/Max", justify="right")
            table.add_column("Success%", justify="right", style="yellow")

            for i, (stat, task) in enumerate(zip(results, tasks, strict=False)):
                task_id = (task.id or "")[:20]
                prompt = (task.prompt or "")[:40]
                if len(task.prompt or "") > 40:
                    prompt += "..."
                table.add_row(
                    str(i + 1),
                    task_id,
                    prompt,
                    f"{stat.get('mean_reward', 0):.3f}±{stat.get('std_reward', 0):.3f}",
                    f"{stat.get('min_reward', 0):.2f}/{stat.get('max_reward', 0):.2f}",
                    f"{stat.get('success_rate', 0) * 100:.0f}%",
                )
            console.print(table)

        high_var = [s for s in results if s.get("std_reward", 0) > 0.3]
        if high_var:
            hud_console.warning(f"\n⚠️  {len(high_var)} tasks show high variance (std > 0.3)")

    else:
        # Single-run traces
        valid_results = [r for r in results if r is not None]
        rewards = [getattr(r, "reward", 0) for r in valid_results]

        if not rewards:
            hud_console.warning("No valid results")
            return

        mean_reward = sum(rewards) / len(rewards)
        successful = sum(1 for r in rewards if r > 0.7)
        success_rate = successful / len(results)

        hud_console.success("\n📊 Evaluation Complete")
        hud_console.info(f"Tasks: {len(results)}")
        if elapsed:
            hud_console.info(f"Time: {elapsed:.1f}s ({len(results) / elapsed:.1f} tasks/s)")
        hud_console.info(f"Mean reward: {mean_reward:.3f}")
        hud_console.info(f"Success rate: {success_rate * 100:.1f}% ({successful}/{len(results)})")

        if show_details and len(results) <= 50:
            table = Table(title="\nPer-Task Results")
            table.add_column("#", style="dim", justify="right")
            table.add_column("Task ID", style="cyan", no_wrap=True)
            table.add_column("Prompt", style="dim", max_width=40)
            table.add_column("Reward", justify="right", style="green")
            table.add_column("Status", justify="center")

            for i, r in enumerate(results):
                task = tasks[i]
                task_id = (task.id or "")[:20]
                prompt = (task.prompt or "")[:40]
                if len(task.prompt or "") > 40:
                    prompt += "..."

                if r is None:
                    table.add_row(str(i + 1), task_id, prompt, "—", "[red]Error[/red]")
                else:
                    reward = getattr(r, "reward", 0)
                    status = "[green]✓[/green]" if reward > 0.7 else "[yellow]✗[/yellow]"
                    table.add_row(str(i + 1), task_id, prompt, f"{reward:.3f}", status)
