"""Utilities for grouped evaluation of tasks, following the RL pattern."""

from __future__ import annotations

import asyncio
from statistics import mean, stdev
from typing import Any

import numpy as np

import hud
from hud.datasets import Task
from hud.types import Trace
from hud.utils.hud_console import HUDConsole

hud_console = HUDConsole()


async def run_tasks_grouped(
    tasks: list[Any],
    agent_class: type | Any,
    agent_config: dict[str, Any] | None = None,
    group_size: int = 1,
    max_parallel_episodes: int = 48,
    max_steps: int = 10,
    verbose: bool = False,
    job_id: str | None = None,
) -> list[dict[str, Any]]:
    """
    Run tasks with grouping, following the RL Actor pattern.

    Args:
        tasks: List of tasks to run
        agent_class: Agent class or instance to use
        agent_config: Configuration for agent instantiation
        group_size: Number of times to run each task
        max_parallel_episodes: Maximum parallel episodes to run
        max_steps: Maximum steps per episode
        verbose: Whether to show progress
        job_id: Optional job ID for tracking

    Returns:
        List of statistics for each task group
    """
    agent_config = agent_config or {}

    # Duplicate tasks according to group_size, exactly like RL
    grouped_tasks = []
    task_mapping = []  # Track which group each result belongs to

    for i, task in enumerate(tasks):
        for _ in range(group_size):
            grouped_tasks.append(task)
            task_mapping.append(i)

    hud_console.info(
        f"Running {len(tasks)} tasks with group_size={group_size} ({len(grouped_tasks)} total runs)"
    )

    # Run all episodes, respecting max_parallel_episodes
    all_traces = []

    # Generate group_ids for each unique task
    import uuid

    task_group_ids = {}
    for i in range(len(tasks)):
        task_group_ids[i] = str(uuid.uuid4())

    for batch_start in range(0, len(grouped_tasks), max_parallel_episodes):
        batch_end = min(batch_start + max_parallel_episodes, len(grouped_tasks))
        batch = grouped_tasks[batch_start:batch_end]

        # Run batch in parallel
        async def run_single_episode(task_data: dict[str, Any] | Task, idx: int) -> Trace:
            """Run a single episode."""
            try:
                # Create task if needed
                task = Task(**task_data) if isinstance(task_data, dict) else task_data

                # Create fresh agent instance
                if isinstance(agent_class, type):
                    agent = agent_class(**agent_config)
                else:
                    # Agent is already instantiated
                    agent = agent_class

                # Run the task
                trace_name = f"Eval | {task.id if hasattr(task, 'id') else 'Task'} | Group {task_mapping[idx]}"  # noqa: E501
                # Use the group_id for this specific task
                task_group_id = task_group_ids[task_mapping[idx]]
                with hud.trace(trace_name, job_id=job_id, group_id=task_group_id):
                    result = await agent.run(task, max_steps=max_steps)
                    return result

            except Exception as e:
                hud_console.warning_log(f"Episode failed: {e}")
                return Trace(isError=True, content=str(e), reward=0.0, done=True)

        # Run batch
        batch_results = await asyncio.gather(
            *[run_single_episode(t, batch_start + i) for i, t in enumerate(batch)],
            return_exceptions=True,
        )

        # Normalize exceptions to error traces
        for res in batch_results:
            if isinstance(res, Exception):
                hud_console.warning_log(f"Episode error: {res}")
                all_traces.append(Trace(isError=True, content=str(res), reward=0.0, done=True))
            else:
                all_traces.append(res)

        if verbose:
            hud_console.info(f"Completed batch: {len(all_traces)}/{len(grouped_tasks)} episodes")

    # Group results back by original task and calculate statistics
    return calculate_group_statistics(tasks, all_traces, task_mapping, group_size, task_group_ids)


def calculate_group_statistics(
    original_tasks: list[Any],
    traces: list[Trace],
    task_mapping: list[int],
    group_size: int,
    task_group_ids: dict[int, str] | None = None,
) -> list[dict[str, Any]]:
    """
    Calculate statistics for each group, similar to preprocess_advantages.

    Args:
        original_tasks: Original task list
        traces: All traces from grouped runs
        task_mapping: Mapping of trace index to task index
        group_size: Number of runs per task
        task_group_ids: Dict mapping task index to group ID

    Returns:
        List of statistics for each task
    """
    stats = []

    # Process each original task
    for task_idx, task in enumerate(original_tasks):
        # Get all traces for this task
        task_traces = [
            traces[i] for i, mapping_idx in enumerate(task_mapping) if mapping_idx == task_idx
        ]

        # Extract rewards
        rewards = np.array([t.reward for t in task_traces])
        errors = [t for t in task_traces if t.isError]

        # Calculate statistics
        task_stats = {
            "task_id": task.id
            if isinstance(task, Task) and hasattr(task, "id")
            else f"task_{task_idx}",
            "prompt": task.prompt if isinstance(task, Task) else task.get("prompt", ""),
            "group_size": group_size,
            "group_id": task_group_ids.get(task_idx) if task_group_ids else None,
            "rewards": rewards.tolist(),
            "mean_reward": float(np.mean(rewards)),
            "std_reward": float(np.std(rewards)) if len(rewards) > 1 else 0.0,
            "min_reward": float(np.min(rewards)),
            "max_reward": float(np.max(rewards)),
            "success_rate": float(np.sum(rewards > 0) / len(rewards)) if len(rewards) > 0 else 0.0,
            "error_rate": len(errors) / len(task_traces) if len(task_traces) > 0 else 0.0,
            "traces": task_traces,  # Keep full traces for detailed analysis
        }

        # Add variance info like RL does
        if task_stats["std_reward"] > 1e-6:
            task_stats["normalized_rewards"] = [
                (r - task_stats["mean_reward"]) / task_stats["std_reward"] for r in rewards
            ]
        else:
            task_stats["normalized_rewards"] = [0.0] * len(rewards)

        stats.append(task_stats)

    return stats


def display_group_statistics(stats: list[dict[str, Any]], show_details: bool = True) -> None:
    """Display statistics from grouped evaluation."""
    from rich.console import Console
    from rich.table import Table

    console = Console()

    # Overall statistics
    all_means = [s["mean_reward"] for s in stats]
    overall_mean = mean(all_means) if all_means else 0.0
    overall_std = stdev(all_means) if len(all_means) > 1 else 0.0

    hud_console.success("\n📊 Evaluation Summary")
    hud_console.info(f"Tasks evaluated: {len(stats)}")
    hud_console.info(f"Episodes per task: {stats[0]['group_size'] if stats else 0}")
    hud_console.info(f"Total episodes: {sum(len(s['rewards']) for s in stats)}")
    hud_console.info(f"Overall mean reward: {overall_mean:.3f} ± {overall_std:.3f}")

    # Detailed table
    if show_details and len(stats) <= 50:  # Only show for reasonable dataset sizes
        table = Table(title="\nPer-Task Performance Distribution")
        table.add_column("Task", style="cyan", no_wrap=True)
        table.add_column("Mean±Std", justify="right", style="green")
        table.add_column("Min/Max", justify="right")
        table.add_column("Success%", justify="right", style="yellow")
        table.add_column("Rewards", style="dim")

        for stat in stats:
            task_name = stat["prompt"][:30] + "..." if len(stat["prompt"]) > 30 else stat["prompt"]
            rewards_str = " ".join([f"{r:.2f}" for r in stat["rewards"][:5]])
            if len(stat["rewards"]) > 5:
                rewards_str += " ..."

            table.add_row(
                task_name,
                f"{stat['mean_reward']:.3f}±{stat['std_reward']:.3f}",
                f"{stat['min_reward']:.2f}/{stat['max_reward']:.2f}",
                f"{stat['success_rate'] * 100:.0f}%",
                rewards_str,
            )

        console.print(table)

    # High variance tasks
    high_variance_tasks = [s for s in stats if s["std_reward"] > 0.3 and s["group_size"] > 1]
    if high_variance_tasks:
        hud_console.warning(f"\n{len(high_variance_tasks)} tasks show high variance (std > 0.3)")
        for task in high_variance_tasks[:3]:
            hud_console.info(
                f"  • {task['task_id']}: μ={task['mean_reward']:.3f}, σ={task['std_reward']:.3f}"  # noqa: RUF001
            )
