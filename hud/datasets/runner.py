"""Core task runner for evaluating agents on datasets.

Requires the [agents] extra: pip install hud-python[agents]
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import hud

if TYPE_CHECKING:
    from hud.agents import MCPAgent
    from hud.eval.context import EvalContext
    from hud.eval.task import Task

logger = logging.getLogger("hud.datasets")


async def run_dataset(
    tasks: str | list[Task],
    agent: MCPAgent,
    *,
    max_steps: int = 10,
    max_concurrent: int = 30,
    group_size: int = 1,
) -> list[EvalContext]:
    """Run an agent on a dataset of tasks.

    This is the primary entry point for running evaluations programmatically.

    Args:
        tasks: Either a source string (file path, API slug) or list of Task objects.
            If a string, tasks are loaded via load_dataset().
        agent: The agent instance to run.
        max_steps: Maximum steps per task.
        max_concurrent: Maximum concurrent tasks (for parallel execution).
        group_size: Number of times to run each task (for variance estimation).

    Returns:
        List of EvalContext results from each task execution. Access `.reward` on each.

    Example:
        ```python
        from hud.agents import ClaudeAgent
        from hud.datasets import load_dataset, run_dataset

        # Load tasks
        tasks = load_dataset("my-tasks.json")

        # Create agent
        agent = ClaudeAgent.create(checkpoint_name="claude-sonnet-4-20250514")

        # Run evaluation
        results = await run_dataset(tasks, agent, max_steps=50)
        for ctx in results:
            print(f"Reward: {ctx.reward}")
        ```
    """
    from hud.datasets.loader import load_dataset

    # Load tasks if string provided
    task_list = load_dataset(tasks) if isinstance(tasks, str) else tasks

    if not task_list:
        raise ValueError("No tasks to run")

    # Use hud.eval() for both single and parallel execution
    async with hud.eval(
        task_list,
        group=group_size,
        max_concurrent=max_concurrent,
    ) as ctx:
        result = await agent.run(ctx, max_steps=max_steps)
        ctx.reward = result.reward

    # For parallel execution, results are collected via ctx.results
    if hasattr(ctx, "results") and ctx.results:
        return ctx.results

    return [ctx]
