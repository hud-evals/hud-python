"""Standard asyncio-based dataset runner."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any, cast

from datasets import Dataset, load_dataset

from hud.agents.misc import ResponseAgent
from hud.types import Task

if TYPE_CHECKING:
    from hud.agents import MCPAgent

logger = logging.getLogger("hud.datasets")


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
) -> list[Any]:
    """Run all tasks in a dataset with automatic job and telemetry tracking.

    This function handles concurrent task execution with proper telemetry collection.
    All tasks are executed in parallel up to `max_concurrent`, with full telemetry
    automatically uploaded to the HUD platform.

    Args:
        name: Name for the job
        dataset: HuggingFace dataset identifier (e.g. "hud-evals/SheetBench-50"),
                Dataset object, OR list of Task objects
        agent_class: Agent class to instantiate (e.g., ClaudeAgent)
        agent_config: Configuration/kwargs for agent (model, etc.)
        max_concurrent: Maximum parallel task execution. Higher values improve throughput
                       but may increase memory usage. Recommended: 30-200 depending on
                       task complexity and available resources.
        metadata: Optional metadata for the job
        max_steps: Maximum steps per task
        split: Dataset split to use when loading from string (default: "train")
        auto_respond: Whether to use auto-response agent

    Returns:
        List of results from agent.run() in dataset order. Telemetry is automatically
        collected and uploaded for all tasks.

    Example:
        >>> from hud.agents import ClaudeAgent
        >>> # Basic usage with dataset identifier
        >>> results = await run_dataset(
        ...     "SheetBench Eval",
        ...     "hud-evals/SheetBench-50",
        ...     ClaudeAgent,
        ...     {"model": "claude-3-5-sonnet-20241022"},
        ...     max_concurrent=100,  # Adjust based on your needs
        ... )
        >>> # Option 2: From HuggingFace dataset object
        >>> from datasets import load_dataset
        >>> dataset = load_dataset("hud-evals/SheetBench-50", split="train")
        >>> results = await run_dataset("my_eval", dataset, ClaudeAgent)
        >>> # Option 3: From list of dicts
        >>> tasks = [{"prompt": "...", "mcp_config": {...}, ...}, ...]
        >>> results = await run_dataset("browser_eval", tasks, ClaudeAgent)

    Note:
        Telemetry collection and upload is handled automatically. The function ensures
        all telemetry is flushed before returning, even at high concurrency levels.
    """
    import hud  # Import here to avoid circular imports

    dataset_link = None

    # Load dataset from string if needed
    if isinstance(dataset, str):
        logger.info("Loading dataset %s from HuggingFace...", dataset)
        dataset_link = dataset

        # Load dataset from HuggingFace
        dataset = cast("Dataset", load_dataset(dataset, split=split))

    # Create job context
    job_metadata = metadata or {}
    job_metadata["agent_class"] = agent_class.__name__
    job_metadata["agent_config"] = agent_config

    # Extract dataset verification info if available
    if isinstance(dataset, Dataset) and not dataset_link:
        try:
            general_info = next(iter(dataset.info.__dict__["download_checksums"].keys())).split("/")
            project = general_info[3]
            dataset_name = general_info[4].split("@")[0]
            dataset_link = f"{project}/{dataset_name}"
        except Exception:
            logger.warning("Failed to extract dataset verification info")

    # Use async job context manager for high-concurrency telemetry
    async with hud.async_job(name, metadata=job_metadata, dataset_link=dataset_link) as job_obj:
        # Run tasks with semaphore for concurrency control
        sem = asyncio.Semaphore(max_concurrent)
        results: list[Any | None] = [None] * len(dataset)

        async def _worker(index: int, task_dict: Any, max_steps: int = 10) -> None:
            async with sem:
                try:
                    # Create trace for this task
                    task_name = task_dict.get("prompt") or f"Task {index}"

                    # Ensure task_id is a string for baggage propagation
                    raw_task_id = task_dict.get("id")
                    safe_task_id = str(raw_task_id) if raw_task_id is not None else None
                    async with hud.async_trace(task_name, job_id=job_obj.id, task_id=safe_task_id):
                        # with hud.trace(task_name, job_id=job_obj.id, task_id=safe_task_id):
                        # Convert dict to Task here, at trace level
                        task = Task(**task_dict)

                        agent = agent_class(**(agent_config or {}))

                        if auto_respond:
                            agent.response_agent = ResponseAgent()
                        results[index] = await agent.run(task, max_steps=max_steps)
                except Exception as e:
                    logger.exception("Task %s failed: %s", index, e)
                    results[index] = None

        # Execute all tasks
        worker_results = await asyncio.gather(
            *[_worker(i, task, max_steps=max_steps) for i, task in enumerate(dataset)],
            return_exceptions=True,  # Don't fail entire batch on one error
        )

        # Log any exceptions that occurred
        for i, result in enumerate(worker_results):
            if isinstance(result, Exception):
                logger.error("Worker %s failed with exception: %s", i, result, exc_info=result)

    # Ensure all telemetry is uploaded before returning
    await _flush_telemetry()

    return results


async def _flush_telemetry() -> None:
    """Flush all pending telemetry operations.

    Ensures complete telemetry upload by:
    1. Waiting for all async status updates to complete
    2. Forcing OpenTelemetry span processor to export remaining spans

    This prevents telemetry loss at high concurrency (200+ tasks) by ensuring
    all operations complete before process exit.
    """
    from hud.otel.config import is_telemetry_configured
    from hud.utils import hud_console
    from hud.utils.task_tracking import wait_all_tasks

    hud_console.info("Uploading telemetry...")

    # Step 1: Wait for async status updates (job/trace status)
    completed_tasks = await wait_all_tasks(timeout_seconds=20.0)
    if completed_tasks > 0:
        hud_console.info(f"Completed {completed_tasks} pending telemetry tasks")

    # Step 2: Flush OpenTelemetry span exports
    if is_telemetry_configured():
        try:
            from opentelemetry import trace
            from opentelemetry.sdk.trace import TracerProvider

            provider = trace.get_tracer_provider()
            if isinstance(provider, TracerProvider):
                provider.force_flush(timeout_millis=20000)
                logger.debug("OpenTelemetry spans flushed successfully")
        except Exception as e:
            logger.warning("Failed to flush OpenTelemetry: %s", e)

    hud_console.info("Telemetry uploaded successfully")
