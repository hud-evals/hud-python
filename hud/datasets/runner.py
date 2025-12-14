"""Core task runner for evaluating agents on datasets.

Requires the [agents] extra: pip install hud-python[agents]
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import hud
from hud.types import AgentType, Trace

if TYPE_CHECKING:
    from hud.agents import MCPAgent
    from hud.eval.context import EvalContext
    from hud.eval.task import Task

logger = logging.getLogger("hud.datasets")


async def run_tasks(
    tasks: list[Task],
    *,
    agent_type: str,
    agent_params: dict[str, Any] | None = None,
    max_steps: int = 10,
    max_concurrent: int = 30,
    group_size: int = 1,
) -> list[EvalContext]:
    """Run tasks with an agent created from type and parameters.

    This is a convenience wrapper around run_dataset that creates the agent
    from a type string and parameters dictionary.

    Args:
        tasks: List of Task objects to run.
        agent_type: Type of agent to create (e.g., "claude", "openai", "gemini").
        agent_params: Parameters to pass to agent.create().
        max_steps: Maximum steps per task.
        max_concurrent: Maximum concurrent tasks.
        group_size: Number of times to run each task.

    Returns:
        List of EvalContext results from each task execution.
    """
    # Use AgentType enum to get the agent class (same pattern as CLI)
    agent_type_enum = AgentType(agent_type)
    agent_cls = agent_type_enum.cls
    agent = agent_cls.create(**(agent_params or {}))

    return await run_dataset(
        tasks,
        agent,
        max_steps=max_steps,
        max_concurrent=max_concurrent,
        group_size=group_size,
    )


async def run_dataset(
    tasks: str | list[Task] | list[dict[str, Any]] | Task | dict[str, Any],
    agent: MCPAgent,
    *,
    max_steps: int = 10,
    max_concurrent: int = 30,
    group_size: int = 1,
) -> list[EvalContext]:
    """Run an agent on a dataset of tasks.

    This is the primary entry point for running evaluations programmatically.

    Args:
        tasks: Tasks to run. Can be:
            - A source string (file path, API slug) - loaded via load_dataset()
            - A single Task object or dict (v4 or v5 format)
            - A list of Task objects or dicts (v4 or v5 format)
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
    from hud.datasets.loader import _task_from_dict, load_dataset
    from hud.eval.task import Task

    # Normalize tasks to list[Task]
    if isinstance(tasks, str):
        task_list = load_dataset(tasks)
    elif isinstance(tasks, Task):
        task_list = [tasks]
    elif isinstance(tasks, dict):
        task_list = [_task_from_dict(tasks)]
    elif isinstance(tasks, list):
        task_list = []
        for t in tasks:
            if isinstance(t, Task):
                task_list.append(t)
            elif isinstance(t, dict):
                task_list.append(_task_from_dict(t))
            else:
                raise TypeError(f"Expected Task or dict, got {type(t)}")
    else:
        raise TypeError(f"Expected str, Task, dict, or list, got {type(tasks)}")

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


async def run_single_task(
    task: Task | dict[str, Any],
    *,
    agent_type: AgentType,
    agent_params: dict[str, Any] | None = None,
    max_steps: int = 10,
    job_id: str | None = None,
    task_id: str | None = None,
    group_id: str | None = None,
    trace_name: str | None = None,
    metadata: dict[str, Any] | None = None,
    trace_id: str | None = None,
) -> Trace:
    """Run a single task with full control over eval context parameters.

    This is the low-level entry point for running individual tasks with explicit
    trace/job/group IDs. Useful for remote execution workers.

    Args:
        task: Task to run. Can be a Task object or dict (v4 or v5 format).
        agent_type: AgentType enum specifying the agent to use.
        agent_params: Parameters passed to agent.create(). Should include
            pre-configured model_client for inference gateway usage.
        max_steps: Maximum steps allowed for the agent.
        job_id: HUD job identifier for telemetry association.
        task_id: Task identifier (used in trace name if trace_name not provided).
        group_id: Optional group identifier for parallel runs.
        trace_name: Name for the trace (defaults to task_id or task.id).
        metadata: Additional metadata for the trace context.
        trace_id: Pre-assigned trace ID (if provided by backend).

    Returns:
        Trace result from the agent run.

    Example:
        ```python
        from hud.datasets import run_single_task
        from hud.types import AgentType
        from openai import AsyncOpenAI

        # Configure agent with inference gateway
        agent_params = {
            "checkpoint_name": "gpt-4o",
            "validate_api_key": False,
            "model_client": AsyncOpenAI(
                api_key=hud_api_key,
                base_url=settings.hud_gateway_url,
            ),
        }

        result = await run_single_task(
            task={"env": {"name": "browser"}, "scenario": "find_page"},
            agent_type=AgentType.OPENAI,
            agent_params=agent_params,
            max_steps=20,
            job_id="job-123",
            task_id="task-456",
        )
        ```
    """
    from hud.datasets.loader import _task_from_dict
    from hud.eval.task import Task as TaskCls

    # Normalize task to Task object
    if isinstance(task, dict):
        task_obj = _task_from_dict(task)
    elif isinstance(task, TaskCls):
        task_obj = task
    else:
        raise TypeError(f"Expected Task or dict, got {type(task)}")

    # Create agent
    agent_cls = agent_type.cls
    agent = agent_cls.create(**(agent_params or {}))

    # Determine trace name
    effective_trace_name = trace_name or task_id or task_obj.id or "single_task"

    # Run with explicit eval context parameters
    async with hud.eval(
        task_obj,
        name=effective_trace_name,
        job_id=job_id,
        group_id=group_id,
        trace_id=trace_id,
    ) as ctx:
        # Store metadata if provided
        if metadata:
            for key, value in metadata.items():
                setattr(ctx, f"_meta_{key}", value)

        result = await agent.run(ctx, max_steps=max_steps)
        ctx.reward = result.reward

    return result
