"""Process-based parallel dataset runner."""

from __future__ import annotations

import asyncio
import logging
import multiprocessing
import os
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from datasets import Dataset

    from hud.agents import MCPAgent

logger = logging.getLogger(__name__)


# Worker function that runs in a separate process
def _process_worker(
    task_batch: list[tuple[int, dict[str, Any]]],
    agent_class_module: str,
    agent_class_name: str,
    agent_config: dict[str, Any] | None,
    job_id: str,
    job_name: str,
    max_steps: int,
    auto_respond: bool,
    worker_id: int,
    total_workers: int,
    max_concurrent_per_worker: int,
) -> list[tuple[int, Any]]:
    """
    Worker function that runs in a separate process.

    This function:
    1. Reinitializes telemetry in the new process
    2. Creates its own event loop
    3. Processes a batch of tasks asynchronously
    4. Returns results with their original indices
    5. Handles interruption signals gracefully

    Args:
        task_batch: List of (index, task_dict) tuples
        agent_class_module: Module path for the agent class
        agent_class_name: Name of the agent class
        agent_config: Configuration for agent initialization
        job_id: Job ID for telemetry tracking
        job_name: Job name for logging
        max_steps: Maximum steps per task
        auto_respond: Whether to use ResponseAgent
        worker_id: ID of this worker process
        total_workers: Total number of worker processes
        max_concurrent_per_worker: Maximum concurrent tasks within each worker

    Returns:
        List of (index, result) tuples
    """
    # Import inside worker to avoid pickling issues
    import signal
    import sys

    import hud
    from hud.agents.misc.response_agent import ResponseAgent
    from hud.datasets.task import Task
    from hud.otel import configure_telemetry

    # Ensure stdout is not buffered for immediate output
    try:
        sys.stdout.reconfigure(line_buffering=True)  # type: ignore
        sys.stderr.reconfigure(line_buffering=True)  # type: ignore
    except AttributeError:
        pass

    # Set up signal handler for clean interruption
    def signal_handler(signum: int, frame: Any) -> None:
        logger.warning("Worker %s: Received interrupt signal", worker_id)
        # Raise KeyboardInterrupt to actually interrupt the worker
        raise KeyboardInterrupt(f"Worker {worker_id} interrupted by user")

    signal.signal(signal.SIGINT, signal_handler)

    # Reinitialize telemetry in this process
    configure_telemetry()

    # Dynamically import the agent class
    try:
        import importlib

        module = importlib.import_module(agent_class_module)
        agent_class = getattr(module, agent_class_name)
    except (ImportError, AttributeError) as e:
        logger.error("Worker %s: Failed to import agent class: %s", worker_id, e)
        return [(idx, {"error": str(e), "isError": True}) for idx, _ in task_batch]

    # Create new event loop for this process
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    async def process_batch() -> list[tuple[int, Any]]:
        """Process all tasks in the batch asynchronously."""
        results = []

        # Use semaphore to limit concurrency within the process
        sem = asyncio.Semaphore(max_concurrent_per_worker)

        async def process_single_task(index: int, task_dict: dict[str, Any]) -> tuple[int, Any]:
            """Process a single task with telemetry tracking."""
            async with sem:
                try:
                    # Create trace for this task (linked to the job) - match original format
                    task_name = task_dict.get("prompt") or f"Task {index}"

                    # Use the job_id to group all tasks under the same job
                    with hud.trace(task_name, job_id=job_id, task_id=task_dict.get("id")):
                        # Convert dict to Task
                        task = Task(**task_dict)

                        # Create agent instance
                        agent = agent_class(**(agent_config or {}))

                        if auto_respond:
                            agent.response_agent = ResponseAgent()

                        # Run the task
                        result = await agent.run(task, max_steps=max_steps)

                        # Extract and print evaluation score for visibility
                        reward = getattr(result, "reward", "N/A")
                        logger.info(
                            "[Worker %s] Task %s: ✓ Completed (reward: %s)",
                            worker_id,
                            index,
                            reward,
                        )

                        logger.info(
                            "[Worker %s] Completed task %s (reward: %s)",
                            worker_id,
                            index,
                            reward,
                        )

                        return (index, result)

                except Exception as e:
                    error_msg = f"Worker {worker_id}: Task {index} failed: {e}"
                    logger.error(
                        "[Worker %s] Task %s: ✗ Failed (%s)", worker_id, index, str(e)[:100]
                    )
                    logger.error("%s\n%s", error_msg, traceback.format_exc())

                    return (
                        index,
                        {
                            "error": str(e),
                            "traceback": traceback.format_exc(),
                            "isError": True,
                            "reward": 0.0,
                            "done": False,
                            "content": f"Task failed: {e}",
                        },
                    )

        # Process all tasks in parallel within this process
        tasks = [process_single_task(idx, task_dict) for idx, task_dict in task_batch]

        try:
            results = await asyncio.gather(*tasks, return_exceptions=False)
            return results
        except asyncio.CancelledError:
            logger.info("Worker %s: Tasks cancelled due to interruption", worker_id)
            # Return error results for all tasks
            return [
                (
                    idx,
                    {
                        "error": "Task cancelled (Ctrl+C)",
                        "isError": True,
                        "reward": 0.0,
                        "done": False,
                        "content": "Task cancelled",
                    },
                )
                for idx, _ in task_batch
            ]

    try:
        # Run the async batch processing
        results = loop.run_until_complete(process_batch())

        # Ensure telemetry is fully sent before process exits
        # Spans are buffered in BatchSpanProcessor and need explicit flush

        from opentelemetry import trace as otel_trace

        provider = otel_trace.get_tracer_provider()
        if provider and hasattr(provider, "force_flush"):
            # Flush of buffered spans
            success = provider.force_flush(timeout_millis=2000)  # type: ignore[arg-type]
            if not success:
                logger.warning("Worker %s: Telemetry flush timed out", worker_id)

        return results
    except KeyboardInterrupt:
        logger.info("Worker %s: Interrupted by user, stopping gracefully", worker_id)
        # Return partial results for tasks that completed
        partial_results = []
        for idx, _ in task_batch:
            partial_results.append(
                (
                    idx,
                    {
                        "error": "Worker interrupted by user (Ctrl+C)",
                        "isError": True,
                        "reward": 0.0,
                        "done": False,
                        "content": "Task interrupted",
                    },
                )
            )
        return partial_results
    except Exception as e:
        logger.error("[Worker %s] Batch processing failed: %s", worker_id, e)
        logger.error("Worker %s batch processing failed: %s", worker_id, e)
        return [(idx, {"error": str(e), "isError": True}) for idx, _ in task_batch]
    finally:
        try:
            from opentelemetry import trace as otel_trace

            provider = otel_trace.get_tracer_provider()
            if provider and hasattr(provider, "force_flush"):
                # Flush buffered spans with reasonable timeout
                success = provider.force_flush(timeout_millis=2000)  # type: ignore[arg-type]
                if not success:
                    logger.warning("Worker %s: Telemetry flush timed out", worker_id)
        except Exception as e:
            logger.warning("Worker %s: Failed to flush telemetry: %s", worker_id, e)

        # Clean up the event loop
        try:
            loop.close()
        except Exception as e:
            logger.warning("Worker %s: Failed to close event loop: %s", worker_id, e)


async def run_dataset_parallel_manual(
    name: str,
    dataset: str | Dataset | list[dict[str, Any]],
    agent_class: type[MCPAgent],
    agent_config: dict[str, Any] | None = None,
    max_workers: int | None = None,
    max_concurrent_per_worker: int = 25,
    max_concurrent: int | None = None,
    metadata: dict[str, Any] | None = None,
    max_steps: int = 10,
    split: str = "train",
    auto_respond: bool = False,
    custom_system_prompt: str | None = None,
) -> list[Any]:
    """
    Run all tasks in a dataset using process-based parallelism with manual configuration.

    This function distributes tasks evenly across multiple processes to achieve true parallelism,
    bypassing Python's GIL limitations. Each process runs its own event loop with concurrent
    task execution controlled by max_concurrent_per_worker or max_concurrent.

    Args:
        name: Name for the job (shown in telemetry)
        dataset: HuggingFace dataset identifier, Dataset object, or list of task dicts
        agent_class: Agent class to use (must be importable in worker processes)
        agent_config: Configuration for agent initialization
        max_workers: Number of processes (defaults to CPU count)
        max_concurrent_per_worker: Max concurrent tasks within each worker
        max_concurrent: Optional total concurrent limit across all workers (overrides per-worker)
        metadata: Optional metadata for the job
        max_steps: Maximum steps per task
        split: Dataset split when loading from string
        auto_respond: Whether to use ResponseAgent
        custom_system_prompt: Override system prompt for all tasks

    Returns:
        List of results in the same order as the input dataset

    Example:
        >>> from hud.agents import ClaudeAgent
        >>> from hud.datasets import run_dataset_parallel_manual
        >>> # Run with 8 workers, 10 concurrent per worker (80 total concurrent)
        >>> results = await run_dataset_parallel_manual(
        ...     "Large Scale Eval",
        ...     "hud-evals/benchmark-400",
        ...     ClaudeAgent,
        ...     max_workers=8,
        ...     max_concurrent_per_worker=10,
        ... )
        >>> # OR limit total concurrent to prevent rate limits
        >>> results = await run_dataset_parallel_manual(
        ...     "Rate Limited Eval",
        ...     dataset,
        ...     ClaudeAgent,
        ...     max_workers=8,
        ...     max_concurrent=20,  # Only 20 total concurrent
        ... )
    """
    from datasets import Dataset
    from datasets import load_dataset as hf_load_dataset

    import hud

    # Determine optimal worker count
    if max_workers is None:
        max_workers = min(os.cpu_count() or 4, 16)  # Cap at 16 to be reasonable

    # If max_concurrent is specified, calculate per-worker concurrency
    if max_concurrent is not None:
        # Distribute concurrent limit across workers
        # Each worker should get a fair share of the total concurrent limit
        max_concurrent_per_worker = max(1, max_concurrent // max_workers)
        logger.info(
            "Limiting to %s total concurrent tasks %s per worker)",
            max_concurrent,
            max_concurrent_per_worker,
        )

    logger.info(
        "Starting parallel dataset run with %s workers (%s concurrent per worker)",
        max_workers,
        max_concurrent_per_worker,
    )

    # Load dataset if needed
    dataset_link = None
    task_dicts: list[dict[str, Any]]

    if isinstance(dataset, str):
        logger.info("Loading dataset %s from HuggingFace...", dataset)
        dataset_link = dataset
        loaded_dataset = hf_load_dataset(dataset, split=split)
        task_dicts = list(loaded_dataset)  # type: ignore
    elif isinstance(dataset, Dataset):
        task_dicts = list(dataset)  # type: ignore
    elif isinstance(dataset, list):
        task_dicts = dataset
    else:
        raise ValueError(f"Dataset must be string, Dataset, or list, got {type(dataset)}")

    # Apply custom system prompt if provided
    if custom_system_prompt:
        for task_dict in task_dicts:
            if "system_prompt" not in task_dict:
                task_dict["system_prompt"] = custom_system_prompt
            else:
                task_dict["system_prompt"] += "\n" + custom_system_prompt

    # Prepare job metadata
    job_metadata = metadata or {}
    job_metadata.update(
        {
            "agent_class": agent_class.__name__,
            "agent_config": agent_config,
            "parallel_mode": "process_pool",
            "max_workers": max_workers,
            "max_concurrent_per_worker": max_concurrent_per_worker,
            "total_tasks": len(task_dicts),
        }
    )

    # Extract dataset verification info if available (match original)
    if isinstance(dataset, Dataset) and not dataset_link:
        try:
            general_info = next(iter(dataset.info.__dict__["download_checksums"].keys())).split("/")
            project = general_info[3]
            dataset_name = general_info[4].split("@")[0]
            dataset_link = f"{project}/{dataset_name}"
        except Exception:
            logger.warning("Failed to extract dataset verification info")

    # task_dicts = task_dicts[:10]

    # Create job context
    with hud.job(name, metadata=job_metadata, dataset_link=dataset_link) as job_obj:
        # Prepare agent class info for pickling
        agent_module = agent_class.__module__
        agent_name = agent_class.__name__

        # Divide tasks evenly among workers
        num_tasks = len(task_dicts)
        tasks_per_worker = (num_tasks + max_workers - 1) // max_workers  # Ceiling division

        task_batches: list[list[tuple[int, dict[str, Any]]]] = []
        for i in range(0, num_tasks, tasks_per_worker):
            batch = [
                (idx, task_dict)
                for idx, task_dict in enumerate(task_dicts[i : i + tasks_per_worker], start=i)
            ]
            if batch:  # Only add non-empty batches
                task_batches.append(batch)

        logger.info(
            "Distributing %s tasks across %s workers (~%s tasks per worker)",
            num_tasks,
            len(task_batches),
            tasks_per_worker,
        )

        # Initialize results list
        results: list[Any] = [None] * len(task_dicts)

        # Create worker function with all needed context
        worker_func = partial(
            _process_worker,
            agent_class_module=agent_module,
            agent_class_name=agent_name,
            agent_config=agent_config,
            job_id=job_obj.id,
            job_name=name,
            max_steps=max_steps,
            auto_respond=auto_respond,
            total_workers=min(max_workers, len(task_batches)),
            max_concurrent_per_worker=max_concurrent_per_worker,
        )

        # Process batches in parallel using ProcessPoolExecutor
        executor = ProcessPoolExecutor(
            max_workers=max_workers,
            mp_context=multiprocessing.get_context("spawn"),
        )
        try:
            # Submit all batches to workers
            future_to_batch = {
                executor.submit(worker_func, batch, worker_id=i): batch
                for i, batch in enumerate(task_batches)
            }

            # Track progress
            completed = 0
            total = len(task_dicts)

            # Process results as they complete
            try:
                for future in as_completed(future_to_batch):
                    batch = future_to_batch[future]

                    try:
                        # Get results from this worker
                        batch_results = future.result()

                        # Place results in correct positions
                        for index, result in batch_results:
                            results[index] = result
                            completed += 1

                        # Calculate success rate so far
                        successful_so_far = sum(
                            1
                            for r in results[:completed]
                            if r is not None and getattr(r, "reward", 0) > 0
                        )

                        progress_msg = (
                            f"Progress: {completed}/{total} tasks completed "
                            f"({100 * completed / total:.1f}%) | "
                            f"Success rate: {successful_so_far}/{completed} "
                            f"({100 * successful_so_far / completed:.1f}%)"
                        )

                        logger.info(progress_msg)

                    except Exception as e:
                        # Handle worker failure
                        logger.error(
                            "Worker failed with exception: %s\n%s", e, traceback.format_exc()
                        )

                        # Mark all tasks in this batch as failed
                        for index, _ in batch:
                            results[index] = {
                                "error": f"Worker process failed: {e}",
                                "isError": True,
                                "reward": 0.0,
                                "done": False,
                                "content": f"Worker process failed: {e}",
                            }
                            completed += 1

            except KeyboardInterrupt:
                logger.warning("\n⚠️  Parallel evaluation interrupted by user (Ctrl+C)")
                logger.info("Cancelling pending tasks...")

                # Cancel all pending futures
                for future in future_to_batch:
                    if not future.done():
                        future.cancel()

                # Mark uncompleted tasks as interrupted
                for i, r in enumerate(results):
                    if r is None:
                        results[i] = {
                            "error": "Evaluation interrupted by user",
                            "isError": True,
                            "reward": 0.0,
                            "done": False,
                            "content": "Task interrupted (Ctrl+C)",
                        }

                logger.info("Interrupted after %s/%s tasks", completed, total)
                raise  # Re-raise to propagate the interrupt

        finally:
            # Always shutdown the executor properly
            executor.shutdown(wait=False, cancel_futures=True)

        # Verify all results are populated
        missing = [i for i, r in enumerate(results) if r is None]
        if missing:
            logger.warning("Missing results for task indices: %s...", missing[:10])
            for idx in missing:
                results[idx] = {
                    "error": "No result returned from worker",
                    "isError": True,
                    "reward": 0.0,
                    "done": False,
                    "content": "Task was not processed",
                }

        # Print final summary
        total_tasks = len(results)
        successful_tasks = sum(1 for r in results if getattr(r, "reward", 0) > 0)
        failed_tasks = sum(1 for r in results if isinstance(r, dict) and r.get("isError", False))

        logger.info("\n")
        logger.info("=" * 60)
        logger.info("📊 Parallel Evaluation Complete!")
        logger.info("=" * 60)
        logger.info("Total tasks: %s", total_tasks)
        logger.info("Successful: %s (%s%%)", successful_tasks, 100 * successful_tasks / total_tasks)
        logger.info("Failed: %s", failed_tasks)
        logger.info("Workers used: %s", max_workers)
        logger.info("=" * 60)

        logger.info(
            "Parallel dataset run completed: %s tasks, %s successful (%s%%)",
            total_tasks,
            successful_tasks,
            100 * successful_tasks / total_tasks,
        )

    return results


def calculate_optimal_workers(num_tasks: int, reserve_system_resources: bool = True) -> int:
    """
    Calculate optimal number of workers based on CPU cores and task count.

    Simple heuristic:
    - 1 worker per CPU core (minus 1-2 for system if reserve_system_resources)
    - But don't create more workers than tasks
    - Cap at reasonable maximum

    Args:
        num_tasks: Total number of tasks to process
        reserve_system_resources: Whether to leave CPU cores for system (default True)

    Returns:
        Optimal number of workers
    """
    # Get CPU count
    cpu_count = os.cpu_count() or 4

    # Reserve 1-2 cores for system if requested
    if reserve_system_resources:
        if cpu_count > 8:
            available_cpus = cpu_count - 2  # Reserve 2 for systems with many cores
        elif cpu_count > 2:
            available_cpus = cpu_count - 1  # Reserve 1 for typical systems
        else:
            available_cpus = 1  # Minimum 1 worker
    else:
        available_cpus = cpu_count

    # Cap at 32 workers to be reasonable
    max_workers = min(available_cpus, 32)

    # Don't create more workers than tasks
    # But try to have at least 5-10 tasks per worker for efficiency
    if num_tasks <= max_workers:
        return min(num_tasks, max_workers)
    else:
        # For many tasks, use all available workers
        # unless that would give us very few tasks per worker
        min_tasks_per_worker = 10
        ideal_workers = min(max_workers, max(1, num_tasks // min_tasks_per_worker))
        return ideal_workers


async def run_dataset_parallel(
    name: str,
    dataset: str | Dataset | list[dict[str, Any]],
    agent_class: type[MCPAgent],
    agent_config: dict[str, Any] | None = None,
    max_concurrent: int | None = None,
    metadata: dict[str, Any] | None = None,
    max_steps: int = 10,
    **kwargs: Any,
) -> list[Any]:
    """
    Run all tasks in a dataset using automatically optimized process-based parallelism.

    This function automatically determines the optimal number of workers
    and batch sizes based on system resources and dataset size. For manual control
    over worker configuration, use `run_dataset_parallel_manual`.

    Args:
        name: Name for the job
        dataset: Dataset to run
        agent_class: Agent class to use
        agent_config: Agent configuration
        max_concurrent: Maximum total concurrent tasks across all workers (prevents rate limits)
        metadata: Optional metadata
        max_steps: Maximum steps per task
        **kwargs: Additional arguments passed to run_dataset_parallel_manual

    Example:
        >>> # Automatically handles 400+ tasks efficiently
        >>> results = await run_dataset_parallel(
        ...     "Large Evaluation",
        ...     "hud-evals/benchmark-400",
        ...     ClaudeAgent,
        ...     max_concurrent=50,  # Limit to 50 concurrent API calls
        ... )
    """
    # Load dataset to get size
    num_tasks: int

    if isinstance(dataset, str):
        from datasets import load_dataset as hf_load_dataset

        dataset_obj = hf_load_dataset(dataset, split=kwargs.get("split", "train"))
        num_tasks = len(dataset_obj)  # type: ignore
    elif hasattr(dataset, "__len__"):
        num_tasks = len(dataset)
    else:
        # Convert to list to count
        dataset_list: list[dict[str, Any]] = list(dataset)  # type: ignore
        num_tasks = len(dataset_list)
        dataset = dataset_list

    # Calculate optimal configuration
    num_workers = calculate_optimal_workers(num_tasks)

    # Set default max_concurrent_per_worker if not using total limit
    if max_concurrent is None:
        max_concurrent_per_worker = 25  # Reasonable default
    else:
        max_concurrent_per_worker = max(1, max_concurrent // num_workers)

    logger.info(
        "Auto-configured for %s tasks: %s workers, %s concurrent per worker",
        num_tasks,
        num_workers,
        max_concurrent_per_worker,
    )

    # Add auto-configuration info to metadata
    if metadata is None:
        metadata = {}
    metadata["auto_configured"] = True
    metadata["auto_num_workers"] = num_workers

    # Run with optimized settings
    return await run_dataset_parallel_manual(
        name=name,
        dataset=dataset,
        agent_class=agent_class,
        agent_config=agent_config,
        max_workers=num_workers,
        max_concurrent_per_worker=max_concurrent_per_worker,
        max_concurrent=max_concurrent,
        metadata=metadata,
        max_steps=max_steps,
        **kwargs,
    )
