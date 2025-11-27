"""Core task runner for evaluating agents on datasets."""

from __future__ import annotations

import asyncio
import logging
import uuid
import warnings
from statistics import mean, stdev
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from datasets import Dataset, load_dataset

from hud.types import Task, Trace

if TYPE_CHECKING:
    from hud.agents import MCPAgent

logger = logging.getLogger("hud.datasets")


async def run_tasks(
    tasks: list[Task],
    agent_class: type["MCPAgent"],
    agent_config: dict[str, Any] | None = None,
    *,
    name: str = "Evaluation",
    max_concurrent: int = 30,
    metadata: dict[str, Any] | None = None,
    max_steps: int = 10,
    auto_respond: bool = False,
    group_size: int = 1,
) -> list[Any]:
    """Run a list of tasks with automatic job and telemetry tracking.

    This is the core evaluation function. Use this when you have a list of tasks
    to run, whether loaded from a dataset, filtered, or constructed programmatically.

    Args:
        tasks: List of Task objects
        agent_class: Agent class to instantiate
        agent_config: Configuration kwargs for agent initialization
        name: Name for the job
        max_concurrent: Maximum concurrent tasks
        metadata: Optional job metadata
        max_steps: Maximum steps per task
        auto_respond: Whether to use auto-response agent
        group_size: Number of times to run each task (for variance estimation)

    Returns:
        If group_size == 1: List of Trace results in task order.
        If group_size > 1: List of statistics dicts for each task group.

    Example:
        # Run specific tasks
        all_tasks = load_tasks("hud-evals/SheetBench-50")
        selected = [t for t in all_tasks if t.id in ["task_1", "task_5"]]
        results = await run_tasks(selected, ClaudeAgent, {"model": "claude-sonnet-4-5"})

        # Run with variance estimation
        stats = await run_tasks(tasks, ClaudeAgent, group_size=3)
    """
    import hud

    # Create job metadata
    job_metadata = metadata or {}
    job_metadata["agent_config"] = agent_config or {}
    if group_size > 1:
        job_metadata["group_size"] = group_size
        job_metadata["total_episodes"] = len(tasks) * group_size

    async with hud.async_job(name, metadata=job_metadata) as job_obj:
        return await _run_tasks(
            tasks, agent_class, agent_config, max_concurrent, max_steps, auto_respond, group_size, job_obj
        )


async def run_dataset(
    name: str,
    dataset: str | Dataset | list[dict[str, Any]],
    agent_class: type["MCPAgent"],
    agent_config: dict[str, Any] | None = None,
    max_concurrent: int = 30,
    metadata: dict[str, Any] | None = None,
    max_steps: int = 10,
    split: str = "train",
    auto_respond: bool = False,
    group_size: int = 1,
) -> list[Any]:
    """Load and run all tasks from a dataset.

    .. deprecated::
        Use `run_tasks()` for new code. This function remains for backwards
        compatibility but `run_tasks()` offers more flexibility (filtering,
        custom task lists, etc.).

    Args:
        name: Name for the job
        dataset: HuggingFace dataset identifier, Dataset object, or list of dicts
        agent_class: Agent class to instantiate
        agent_config: Configuration kwargs for agent initialization
        max_concurrent: Maximum concurrent tasks
        metadata: Optional job metadata
        max_steps: Maximum steps per task
        split: Dataset split to use when loading from string
        auto_respond: Whether to use auto-response agent
        group_size: Number of times to run each task (for variance estimation)

    Returns:
        If group_size == 1: List of results from agent.run() in dataset order.
        If group_size > 1: List of statistics dicts for each task group.
    """
    warnings.warn(
        "run_dataset() is deprecated. Use run_tasks() instead for more flexibility.",
        DeprecationWarning,
        stacklevel=2,
    )

    # Load dataset and convert to Task objects
    task_dicts: list[dict[str, Any]]
    dataset_link: str | None = None

    if isinstance(dataset, str):
        logger.info("Loading dataset %s from HuggingFace...", dataset)
        dataset_link = dataset
        loaded = cast("Dataset", load_dataset(dataset, split=split))
        task_dicts = cast("list[dict[str, Any]]", list(loaded))
    elif isinstance(dataset, Dataset):
        task_dicts = cast("list[dict[str, Any]]", list(dataset))
        # Try to extract dataset link
        try:
            general_info = next(iter(dataset.info.__dict__["download_checksums"].keys())).split("/")
            dataset_link = f"{general_info[3]}/{general_info[4].split('@')[0]}"
        except Exception:
            pass
    else:
        task_dicts = dataset

    # Convert dicts to Task objects
    tasks = [Task(**d) for d in task_dicts]

    # Add dataset link to metadata
    job_metadata = metadata or {}
    if dataset_link:
        job_metadata["dataset_link"] = dataset_link

    return await run_tasks(
        tasks=tasks,
        agent_class=agent_class,
        agent_config=agent_config,
        name=name,
        max_concurrent=max_concurrent,
        metadata=job_metadata,
        max_steps=max_steps,
        auto_respond=auto_respond,
        group_size=group_size,
    )


async def _run_tasks(
    tasks: list[Task],
    agent_class: type["MCPAgent"],
    agent_config: dict[str, Any] | None,
    max_concurrent: int,
    max_steps: int,
    auto_respond: bool,
    group_size: int,
    job_obj: Any,
) -> list[Any]:

    import hud

    sem = asyncio.Semaphore(max_concurrent)
    config = agent_config or {}

    # Generate group IDs for each task (used for telemetry grouping)
    group_ids = {i: str(uuid.uuid4()) for i in range(len(tasks))}

    # Expand tasks: each task runs group_size times
    expanded: list[tuple[int, int, Task]] = []  # (flat_idx, task_idx, task)
    for task_idx, task in enumerate(tasks):
        for run_idx in range(group_size):
            expanded.append((len(expanded), task_idx, task))

    traces: list[Trace | None] = [None] * len(expanded)

    async def worker(flat_idx: int, task_idx: int, run_idx: int, task: Task) -> None:
        async with sem:
            try:
                base_task_id = str(task.id) if task.id is not None else f"task_{task_idx}"
                trace_name = task.prompt or base_task_id

                if group_size == 1:
                    async with hud.async_trace(trace_name, job_id=job_obj.id, task_id=base_task_id):
                        agent = _create_agent(agent_class, config, auto_respond)
                        traces[flat_idx] = await agent.run(task, max_steps=max_steps)
                else:
                    task_id_with_run = f"{base_task_id}_{run_idx}"
                    async with hud.async_trace(trace_name, job_id=job_obj.id, task_id=task_id_with_run, group_id=group_ids[task_idx]):
                        agent = _create_agent(agent_class, config, auto_respond)
                        traces[flat_idx] = await agent.run(task, max_steps=max_steps)
            except Exception as e:
                if group_size == 1:
                    logger.exception("Task %s failed: %s", task_idx, e)
                    traces[flat_idx] = None
                else:
                    logger.warning("Episode %s failed: %s", flat_idx, e)
                    traces[flat_idx] = Trace(isError=True, content=str(e), reward=0.0, done=True)

    await asyncio.gather(
        *[
            worker(flat_idx, task_idx, flat_idx % group_size, task)
            for flat_idx, task_idx, task in expanded
        ],
        return_exceptions=True,
    )

    # Return format depends on group_size
    if group_size == 1:
        return list(traces)
    else:
        return _calculate_group_stats(tasks, traces, group_size, group_ids)


def _create_agent(
    agent_class: type["MCPAgent"],
    config: dict[str, Any],
    auto_respond: bool,
) -> "MCPAgent":
    """Create an agent instance from config."""
    payload = dict(config)
    base_keys = {"mcp_client", "auto_trace", "verbose", "auto_respond"}
    base_kwargs = {k: payload.pop(k) for k in list(payload.keys()) if k in base_keys}

    # Override auto_respond if specified
    if auto_respond:
        base_kwargs["auto_respond"] = True

    return agent_class.create(**base_kwargs, **payload)


def _calculate_group_stats(
    tasks: list[Task],
    traces: list[Trace | None],
    group_size: int,
    group_ids: dict[int, str],
) -> list[dict[str, Any]]:
    """Calculate statistics for each task group."""
    stats = []

    for task_idx, task in enumerate(tasks):
        # Get traces for this task
        start = task_idx * group_size
        task_traces = [t for t in traces[start : start + group_size] if t is not None]

        if not task_traces:
            stats.append({
                "task_id": task.id or f"task_{task_idx}",
                "prompt": task.prompt or "",
                "group_id": group_ids[task_idx],
                "group_size": group_size,
                "rewards": [],
                "mean_reward": 0.0,
                "std_reward": 0.0,
                "success_rate": 0.0,
                "error_rate": 1.0,
            })
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
    tasks: list["Task"],
    elapsed: float | None = None,
    show_details: bool = True,
) -> None:
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
        hud_console.info(f"Tasks: {len(results)} × {group_size} runs = {total_episodes} episodes")
        if elapsed:
            hud_console.info(f"Time: {elapsed:.1f}s ({total_episodes/elapsed:.1f} episodes/s)")
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
            hud_console.info(f"Time: {elapsed:.1f}s ({len(results)/elapsed:.1f} tasks/s)")
        hud_console.info(f"Mean reward: {mean_reward:.3f}")
        hud_console.info(f"Success rate: {success_rate*100:.1f}% ({successful}/{len(results)})")

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
            console.print(table)
