"""Standard asyncio-based dataset runner."""

from __future__ import annotations

import asyncio
import logging
import math
from typing import TYPE_CHECKING, Any, cast

from datasets import Dataset, load_dataset

from hud.agents.misc import ResponseAgent
from hud.types import Task

if TYPE_CHECKING:
    from hud.agents import MCPAgent

logger = logging.getLogger("hud.datasets")


class AdaptiveSemaphore:
    """Semaphore that can adapt its concurrency based on rate limit errors."""
    
    def __init__(self, initial_value: int, min_value: int = 1):
        self._initial_value = initial_value
        self._current_value = initial_value
        self._min_value = min_value
        self._lock = asyncio.Lock()
        self._active_count = 0
        self._rate_limit_count = 0
        self._rate_limit_threshold = 3
        
    async def acquire(self):
        """Acquire a slot, respecting current concurrency limit."""
        async with self._lock:
            # Wait until we have available capacity
            while self._active_count >= self._current_value:
                # Release lock temporarily and wait
                pass
            self._active_count += 1
            
    async def release(self):
        """Release a slot."""
        async with self._lock:
            self._active_count -= 1
            
    async def __aenter__(self):
        # Wait until we can acquire
        while True:
            async with self._lock:
                if self._active_count < self._current_value:
                    self._active_count += 1
                    break
            await asyncio.sleep(0.1)  # Small delay before checking again
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        # Release slot
        async with self._lock:
            self._active_count -= 1
        
        # Check if this was a rate limit error
        if exc_val and self._is_rate_limit_error(exc_val):
            await self._handle_rate_limit()
        
        return False
    
    def _is_rate_limit_error(self, exc: Exception) -> bool:
        """Check if exception is a rate limit error."""
        exc_str = str(exc).lower()
        return "429" in exc_str or "rate limit" in exc_str or "quota" in exc_str or "overloaded" in exc_str
    
    async def _handle_rate_limit(self):
        """Handle rate limit by reducing concurrency by half (ceiling)."""
        async with self._lock:
            self._rate_limit_count += 1
            
            if self._rate_limit_count >= self._rate_limit_threshold:
                if self._current_value > self._min_value:
                    old_value = self._current_value
                    # Reduce by ceiling of half
                    reduction = math.ceil(self._current_value / 2)
                    self._current_value = max(reduction, self._min_value)
                    
                    logger.warning(
                        f"🔽 Rate limit threshold reached ({self._rate_limit_count} errors). "
                        f"Reducing concurrency by ~50%: {old_value} → {self._current_value}"
                    )
                    # Reset counter
                    self._rate_limit_count = 0
                else:
                    logger.warning(
                        f"⚠️  At minimum concurrency ({self._min_value}), "
                        f"cannot reduce further despite rate limits"
                    )
    
    def get_current_value(self) -> int:
        """Get current concurrency limit."""
        return self._current_value


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

    Args:
        name: Name for the job
        dataset: HuggingFace dataset identifier (e.g. "hud-evals/SheetBench-50"),
                Dataset object, OR list of Task objects
        agent_class: Agent class to instantiate (e.g., ClaudeAgent)
        agent_config: Configuration kwargs for agent initialization
        max_concurrent: Maximum concurrent tasks (recommended: 50-200)
        metadata: Optional job metadata
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

    async with hud.async_job(name, metadata=job_metadata, dataset_link=dataset_link) as job_obj:
        # Run tasks with adaptive semaphore for rate-limit-aware concurrency control
        sem = AdaptiveSemaphore(max_concurrent, min_value=1)
        logger.info(f"Starting with max_concurrent={max_concurrent}, will adapt if rate limits detected")
        results: list[Any | None] = [None] * len(dataset)

        async def _worker(index: int, task_dict: Any, max_steps: int = 10) -> None:
            async with sem:
                try:
                    # Create trace for this task
                    task_name = task_dict.get("prompt") or f"Task {index}"
                    raw_task_id = task_dict.get("id")
                    safe_task_id = str(raw_task_id) if raw_task_id is not None else None

                    async with hud.async_trace(task_name, job_id=job_obj.id, task_id=safe_task_id):
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
        
        # Report if concurrency was adapted
        final_concurrency = sem.get_current_value()
        if final_concurrency < max_concurrent:
            logger.warning(
                f"📊 Adaptive concurrency: Started at {max_concurrent}, "
                f"ended at {final_concurrency} due to rate limit errors"
            )

    return results
