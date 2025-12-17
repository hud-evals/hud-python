"""Core task runner for evaluating agents on datasets.

Requires the [agents] extra: pip install hud-python[agents]
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import hud
from hud.types import AgentType, LegacyTask, TaskInput, Trace

if TYPE_CHECKING:
    from collections.abc import Sequence

    from hud.eval.context import EvalContext
    from hud.eval.task import Task

logger = logging.getLogger("hud.datasets")


async def run_dataset(
    tasks: str | TaskInput | Sequence[TaskInput],
    agent_type: str | AgentType,
    *,
    agent_params: dict[str, Any] | None = None,
    max_steps: int = 10,
    max_concurrent: int = 30,
    group_size: int = 1,
    quiet: bool = True,
) -> list[EvalContext]:
    """Run an agent on a dataset of tasks.

    This is the primary entry point for running evaluations programmatically.
    The agent is created fresh for each task context to ensure correct tool initialization.

    Args:
        tasks: Tasks to run. Can be:
            - A source string (file path, API slug) - loaded via load_tasks()
            - A single TaskInput (Task, LegacyTask, or dict)
            - A list of TaskInput objects
        agent_type: Type of agent to create (e.g., "claude", "openai", AgentType.CLAUDE).
        agent_params: Parameters to pass to agent.create().
        max_steps: Maximum steps per task.
        max_concurrent: Maximum concurrent tasks (for parallel execution).
        group_size: Number of times to run each task (for variance estimation).
        quiet: Whether to suppress printing eval links and opening browser (default True).

    Returns:
        List of EvalContext results from each task execution. Access `.reward` on each.

    Example:
        ```python
        from hud.datasets import load_tasks, run_dataset

        # Load tasks and run
        tasks = load_tasks("my-tasks.json")
        results = await run_dataset(
            tasks,
            agent_type="claude",
            agent_params={"checkpoint_name": "claude-sonnet-4-20250514"},
            max_steps=50,
        )

        for ctx in results:
            print(f"Reward: {ctx.reward}")
        ```
    """
    from hud.datasets.loader import load_tasks
    from hud.eval.task import Task

    # Normalize tasks to list[Task]
    task_list: list[Task]
    if isinstance(tasks, str):
        task_list = load_tasks(tasks)
    elif isinstance(tasks, Task):
        task_list = [tasks]
    elif isinstance(tasks, LegacyTask | dict):
        # Single LegacyTask or dict - convert to Task
        task_list = [Task.from_v4(tasks)]
    else:
        # Sequence of TaskInput - convert each to Task
        task_list = [t if isinstance(t, Task) else Task.from_v4(t) for t in tasks]

    if not task_list:
        raise ValueError("No tasks to run")

    # Resolve agent class
    agent_type_enum = agent_type if isinstance(agent_type, AgentType) else AgentType(agent_type)
    agent_cls = agent_type_enum.cls

    # Use hud.eval() for both single and parallel execution
    async with hud.eval(
        task_list,
        group=group_size,
        max_concurrent=max_concurrent,
        quiet=quiet,
    ) as ctx:
        # Create agent fresh for each context (ensures correct tool initialization)
        agent = agent_cls.create(**(agent_params or {}))
        result = await agent.run(ctx, max_steps=max_steps)
        ctx.reward = result.reward

    # For parallel execution, results are collected via ctx.results
    if hasattr(ctx, "results") and ctx.results:
        return ctx.results

    return [ctx]


async def run_single_task(
    task: Task,
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
    api_key: str | None = None,
    trace: bool = True,
    quiet: bool = False,
) -> Trace:
    """Run a single task with full control over eval context parameters.

    This is the low-level entry point for running individual tasks with explicit
    trace/job/group IDs. Used by remote execution workers.

    Args:
        task: Task object to run. Use Task.from_v4() or load_tasks() to create.
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
        api_key: API key override for telemetry and backend calls.
        trace: Whether to send trace data to backend (default True).
        quiet: Whether to suppress printing eval link (default False).

    Returns:
        Trace result from the agent run.

    Example:
        ```python
        from hud.datasets import run_single_task
        from hud.eval.task import Task
        from hud.types import AgentType
        from openai import AsyncOpenAI

        # Create task (from v4 dict or directly)
        task = Task.from_v4({"prompt": "...", "mcp_config": {...}, "evaluate_tool": {...}})

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
            task=task,
            agent_type=AgentType.OPENAI,
            agent_params=agent_params,
            max_steps=20,
            job_id="job-123",
            task_id="task-456",
        )
        ```
    """
    # Determine trace name
    effective_trace_name = trace_name or task_id or task.id or "single_task"

    # Run with explicit eval context parameters
    async with hud.eval(
        task,
        name=effective_trace_name,
        job_id=job_id,
        group_id=group_id,
        trace_id=trace_id,
        api_key=api_key,
        trace=trace,
        quiet=quiet,
    ) as ctx:
        # Build agent params - use system_prompt from ctx (set from task.agent_config)
        final_agent_params = dict(agent_params or {})
        if ctx.system_prompt and "system_prompt" not in final_agent_params:
            final_agent_params["system_prompt"] = ctx.system_prompt

        # Create agent inside ctx so it has access to context-derived values
        agent_cls = agent_type.cls
        agent = agent_cls.create(**final_agent_params)

        # Store metadata if provided
        if metadata:
            ctx.metadata.update(metadata)

        result = await agent.run(ctx, max_steps=max_steps)
        ctx.reward = result.reward

    return result
