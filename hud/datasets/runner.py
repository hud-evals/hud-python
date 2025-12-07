"""Core task runner for evaluating agents on datasets."""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
import warnings
from typing import TYPE_CHECKING, Any, cast

from datasets import Dataset, load_dataset

from hud import async_job, async_trace
from hud.datasets.utils import calculate_group_stats, submit_rollouts
from hud.types import AgentType, Task, Trace

if TYPE_CHECKING:
    from hud.agents import MCPAgent

logger = logging.getLogger("hud.datasets")


async def run_single_task(
    task: Task,
    agent_type: AgentType,
    agent_params: dict[str, Any] | None = None,
    max_steps: int = 10,
    job_id: str | None = None,
    task_id: str | None = None,
    group_id: str | None = None,
    trace_id: str | None = None,
    trace_name: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> Trace:
    """Execute a single task with tracing.

    This is the core execution primitive for running a single task.

    Args:
        task: Task to execute
        agent_type: Agent type to use
        agent_params: Parameters passed to agent.create(). Should include fields
            from BaseCreateParams (auto_trace, auto_respond, verbose) plus
            agent-specific config fields (e.g., use_computer_beta for ClaudeConfig).
        max_steps: Maximum steps for agent execution
        job_id: Job ID for telemetry grouping
        task_id: Task ID for telemetry
        group_id: Group ID for variance estimation runs
        trace_id: Trace ID for telemetry (auto-generated if not provided)
        trace_name: Name for the trace (defaults to task prompt)
        metadata: Additional trace metadata

    Returns:
        Trace result from agent execution
    """
    name = trace_name or task.prompt or task_id or "task"

    async with async_trace(
        name,
        job_id=job_id,
        task_id=task_id,
        group_id=group_id,
        trace_id=trace_id,
        attrs=metadata or {},
    ):
        agent = agent_type.cls.create(**(agent_params or {}))
        return await agent.run(task, max_steps=max_steps)


async def run_tasks(
    tasks: list[Task],
    agent_type: AgentType,
    agent_params: dict[str, Any] | None = None,
    *,
    name: str = "Evaluation",
    max_concurrent: int = 30,
    metadata: dict[str, Any] | None = None,
    max_steps: int = 10,
    group_size: int = 1,
    remote: bool = False,
) -> list[Any]:
    """Run a list of tasks with automatic job and telemetry tracking.

    This is the core evaluation function. Use this when you have a list of tasks
    to run, whether loaded from a dataset, filtered, or constructed programmatically.

    Args:
        tasks: List of Task objects
        agent_type: AgentType specifying which agent to use
        agent_params: Parameters passed to agent.create(). Should include fields
            from BaseCreateParams (auto_trace, auto_respond, verbose) plus
            agent-specific config fields (e.g., checkpoint_name for ClaudeConfig).
        name: Name for the job
        max_concurrent: Maximum concurrent tasks
        metadata: Optional job metadata
        max_steps: Maximum steps per task
        group_size: Number of times to run each task (for variance estimation)
        remote: If True, submit tasks to HUD platform for remote execution

    Returns:
        If remote: Empty list (fire-and-forget submission)
        If group_size == 1: List of Trace results in task order.
        If group_size > 1: List of statistics dicts for each task group.

    Example:
        # Run specific tasks locally
        all_tasks = load_tasks("hud-evals/SheetBench-50")
        selected = [t for t in all_tasks if t.id in ["task_1", "task_5"]]
        results = await run_tasks(selected, AgentType.CLAUDE, {"checkpoint_name": "..."})

        # Run with variance estimation
        stats = await run_tasks(tasks, AgentType.CLAUDE, group_size=3)

        # Submit for remote execution
        await run_tasks(tasks, AgentType.CLAUDE, remote=True)
    """
    import hud
    from hud.utils.hud_console import HUDConsole

    job_metadata = metadata or {}
    job_metadata["agent_params"] = json.dumps(agent_params or {})
    job_metadata["agent_type"] = agent_type.value
    if group_size > 1:
        job_metadata["group_size"] = group_size
        job_metadata["total_episodes"] = len(tasks) * group_size

    if remote:
        hud_console = HUDConsole()

        job = hud.create_job(name, metadata=job_metadata)
        job.update_status_sync("created")

        await submit_rollouts(
            tasks=tasks,
            job_id=job.id,
            agent_type=agent_type,
            agent_params=agent_params,
            max_steps=max_steps,
            group_size=group_size,
            metadata=metadata,
        )
        hud_console.success(f"Submitted {len(tasks) * group_size} rollouts for remote execution")
        hud_console.info(f"Monitor progress at: https://hud.ai/jobs/{job.id}")
        return []

    # Local execution
    agent_class = agent_type.cls

    async with async_job(name, metadata=job_metadata) as job_obj:
        return await _run_tasks(
            tasks, agent_class, agent_params, max_concurrent, max_steps, group_size, job_obj
        )


async def run_dataset(
    name: str,
    dataset: str | Dataset | list[dict[str, Any]],
    agent_class: type[MCPAgent],
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
        except Exception:  # noqa: S110
            pass
    else:
        task_dicts = dataset

    # Convert dicts to Task objects
    tasks = [Task(**d) for d in task_dicts]

    # Add dataset link to metadata
    job_metadata = metadata or {}
    job_metadata["agent_config"] = agent_config or {}
    if dataset_link:
        job_metadata["dataset_link"] = dataset_link
    if group_size > 1:
        job_metadata["group_size"] = group_size
        job_metadata["total_episodes"] = len(tasks) * group_size

    async with async_job(name, metadata=job_metadata) as job_obj:
        return await _run_tasks(
            tasks, agent_class, agent_config, max_concurrent, max_steps, group_size, job_obj
        )


async def _run_tasks(
    tasks: list[Task],
    agent_class: type[MCPAgent],
    agent_params: dict[str, Any] | None,
    max_concurrent: int,
    max_steps: int,
    group_size: int,
    job_obj: Any,
) -> list[Any]:
    sem = asyncio.Semaphore(max_concurrent)
    params = agent_params or {}

    # Generate group IDs for each task (used for telemetry grouping)
    group_ids = {i: str(uuid.uuid4()) for i in range(len(tasks))}

    # Expand tasks: each task runs group_size times
    expanded: list[tuple[int, int, Task]] = []  # (flat_idx, task_idx, task)
    for task_idx, task in enumerate(tasks):
        for _ in range(group_size):
            expanded.append((len(expanded), task_idx, task))

    traces: list[Trace | None] = [None] * len(expanded)

    async def worker(flat_idx: int, task_idx: int, run_idx: int, task: Task) -> None:
        async with sem:
            try:
                base_task_id = str(task.id) if task.id is not None else f"task_{task_idx}"
                trace_name = task.prompt or base_task_id

                if group_size == 1:
                    async with async_trace(trace_name, job_id=job_obj.id, task_id=base_task_id):
                        agent = agent_class.create(**params)
                        traces[flat_idx] = await agent.run(task, max_steps=max_steps)
                else:
                    task_id_with_run = f"{base_task_id}_{run_idx}"
                    async with async_trace(
                        trace_name,
                        job_id=job_obj.id,
                        task_id=task_id_with_run,
                        group_id=group_ids[task_idx],
                    ):
                        agent = agent_class.create(**params)
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
        return calculate_group_stats(tasks, traces, group_size, group_ids)
