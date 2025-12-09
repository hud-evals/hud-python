"""Standalone eval() context manager.

Provides hud.eval() for task-based evaluation without needing an existing environment.
"""

from __future__ import annotations

import inspect
import logging
import uuid
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

from hud.eval.display import print_complete, print_eval_stats, print_link
from hud.eval.parallel import (
    ASTExtractionError,
    expand_variants,
    find_user_frame,
    get_with_block_body,
    resolve_group_ids,
)

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from hud.eval.context import EvalContext
    from hud.types import Task

logger = logging.getLogger(__name__)


# Type alias for task source: can be slug strings or Task objects
TaskSource = "str | list[str] | Task | list[Task] | None"


def _parse_slug(slug: str) -> tuple[str, str | None]:
    """Parse a task slug into (base_slug, index_or_wildcard).

    Args:
        slug: Task slug like "my-org/task", "my-org/task:1", or "my-org/task:*"

    Returns:
        Tuple of (base_slug, index_str or None)
        - "my-org/task" -> ("my-org/task", None)
        - "my-org/task:1" -> ("my-org/task", "1")
        - "my-org/task:*" -> ("my-org/task", "*")
    """
    if ":" in slug:
        parts = slug.rsplit(":", 1)
        return parts[0], parts[1]
    return slug, None


def _get_eval_name(
    source: str | list[str] | None = None,
    tasks: list[Task] | None = None,
) -> str:
    """Extract a nice name for job display.

    Args:
        source: Single slug or list of slugs (if string-based)
        tasks: List of Task objects (if using direct tasks)

    Returns:
        Name like "evalset", task ID, or "eval" if no source
    """
    # If we have tasks with IDs, use first task ID
    if tasks:
        first_task = tasks[0]
        if first_task.id:
            # Extract name from task ID (might be "evalset/task_name")
            task_id = str(first_task.id)
            if "/" in task_id:
                return task_id.rsplit("/", 1)[1]
            return task_id
        # Fall back to prompt excerpt
        if first_task.prompt:
            return first_task.prompt[:30].strip()

    # If we have string slugs
    if source is not None:
        # Get the first slug
        first_slug = source if isinstance(source, str) else source[0]

        # Remove index/wildcard suffix (":1" or ":*")
        base_slug, _ = _parse_slug(first_slug)

        # Extract the evalset name (part after last "/")
        if "/" in base_slug:
            return base_slug.rsplit("/", 1)[1]

        return base_slug

    return "eval"


def _load_tasks_from_slugs(slugs: str | list[str]) -> list[Task]:
    """Load tasks from platform by slugs.

    Args:
        slugs: Single slug or list of slugs. Slugs can be:
            - "my-org/task" - single task
            - "my-org/task:N" - task at index N
            - "my-org/task:*" - all tasks matching pattern

    Returns:
        List of Task objects
    """
    import httpx

    from hud.settings import settings
    from hud.types import Task

    if isinstance(slugs, str):
        slugs = [slugs]

    tasks: list[Task] = []

    headers = {}
    if settings.api_key:
        headers["Authorization"] = f"Bearer {settings.api_key}"

    with httpx.Client() as client:
        for slug in slugs:
            base_slug, index_str = _parse_slug(slug)

            if index_str == "*":
                # Fetch all tasks for this evalset
                logger.info("Loading all tasks for: %s", base_slug)
                response = client.get(
                    f"{settings.hud_api_url}/tasks/{base_slug}",
                    headers=headers,
                    params={"all": "true"},
                )
                response.raise_for_status()
                data = response.json()

                if isinstance(data, list):
                    tasks.extend(Task(**item) for item in data)
                else:
                    tasks.append(Task(**data))

            elif index_str is not None:
                # Fetch specific task by index
                logger.info("Loading task: %s (index %s)", base_slug, index_str)
                response = client.get(
                    f"{settings.hud_api_url}/tasks/{base_slug}",
                    headers=headers,
                    params={"index": index_str},
                )
                response.raise_for_status()
                data = response.json()
                tasks.append(Task(**data))

            else:
                # Fetch single task
                logger.info("Loading task: %s", slug)
                response = client.get(
                    f"{settings.hud_api_url}/tasks/{slug}",
                    headers=headers,
                )
                response.raise_for_status()
                data = response.json()
                tasks.append(Task(**data))

    return tasks


@asynccontextmanager
async def run_eval(
    source: str | list[str] | Task | list[Task] | None = None,
    *,
    variants: dict[str, Any] | None = None,
    group: int = 1,
    group_ids: list[str] | None = None,
    job_id: str | None = None,
    api_key: str | None = None,
    max_concurrent: int | None = None,
) -> AsyncGenerator[EvalContext, None]:
    """Standalone eval context manager.

    Creates an EvalContext for evaluation, optionally loading task configuration
    from slugs or using Task objects directly.

    Args:
        source: Task source. Can be:
            - None: Create blank eval context
            - str: Task slug like "my-org/task", "my-org/task:N", "my-org/task:*"
            - list[str]: Multiple task slugs
            - Task: Single Task object (for backwards compat with run_tasks)
            - list[Task]: List of Task objects (for backwards compat with run_tasks)
        variants: A/B test configuration (dict with list values expanded)
        group: Runs per variant for statistical significance
        group_ids: Optional list of group IDs
        job_id: Job ID to link to
        api_key: API key for backend calls
        max_concurrent: Maximum concurrent evals (None = unlimited)

    Yields:
        EvalContext: Environment with evaluation tracking

    Example:
        ```python
        # Blank eval (for manual reward)
        async with hud.eval() as ctx:
            ctx.reward = compute_reward()

        # With task slug
        async with hud.eval("my-org/browser-task:1") as ctx:
            await agent.run(ctx)
            ctx.reward = result.reward

        # Multiple tasks
        async with hud.eval(["task:1", "task:2"]) as ctx:
            await agent.run(ctx)

        # All tasks in evalset
        async with hud.eval("my-org/evalset:*") as ctx:
            await agent.run(ctx)

        # With Task objects directly
        from hud.types import Task

        tasks = [Task(prompt="Do X", mcp_config={...})]
        async with hud.eval(tasks) as ctx:
            await agent.run(ctx)

        # With variants and group
        async with hud.eval(
            "task",
            variants={"model": ["gpt-4o", "claude"]},
            group=3,
        ) as ctx:
            model = ctx.variants["model"]
            await run_agent(model)
            ctx.reward = evaluate()

        # With concurrency limit
        async with hud.eval("my-org/evalset:*", max_concurrent=10) as ctx:
            await agent.run(ctx)

        # Access results after parallel run
        for e in ctx.results:
            print(f"{e.variants}: reward={e.reward}")
        ```
    """
    from hud.types import Task

    if group <= 0:
        raise ValueError("group must be >= 1")

    # Expand variants
    variant_combos = expand_variants(variants)

    # Parse source into tasks list
    tasks: list[Task] = []
    slugs: str | list[str] | None = None  # Track if we had string slugs (for naming)

    if source is not None:
        if isinstance(source, Task):
            # Single Task object
            tasks = [source]
        elif isinstance(source, list) and source and isinstance(source[0], Task):
            # List of Task objects
            tasks = source  # type: ignore[assignment]
        elif isinstance(source, str):
            # String slug
            slugs = source
            tasks = _load_tasks_from_slugs(source)
        elif isinstance(source, list) and source and isinstance(source[0], str):
            # List of string slugs
            slugs = source  # type: ignore[assignment]
            tasks = _load_tasks_from_slugs(source)  # type: ignore[arg-type]

    # Calculate total evaluations
    # If we have tasks, each task gets (variants x group) runs
    # If no tasks, we have a single blank eval with (variants x group) runs
    total_evals = len(tasks) * len(variant_combos) * group if tasks else len(variant_combos) * group

    # Capture code snippet for parallel execution
    code_snippet: str | None = None
    if total_evals > 1:
        frame = inspect.currentframe()
        if frame is not None:
            try:
                caller = frame.f_back
                if caller is not None:
                    code_snippet, _, _ = get_with_block_body(caller)
            except ASTExtractionError:
                pass
            finally:
                del frame

    # Lazy import to avoid circular dependency
    from hud.eval.context import EvalContext

    if total_evals == 1:
        # Simple case: single eval
        if tasks:
            # Single task
            ctx = EvalContext.from_task(
                task=tasks[0],
                api_key=api_key,
                job_id=job_id,
                variants=variant_combos[0],
                code_snippet=code_snippet,
            )
        else:
            # Blank eval
            ctx = EvalContext(
                name="eval",
                api_key=api_key,
                job_id=job_id,
                variants=variant_combos[0],
                code_snippet=code_snippet,
            )

        async with ctx:
            yield ctx

    else:
        # Parallel execution: create implicit job to group traces
        eval_name = _get_eval_name(source=slugs, tasks=tasks)
        implicit_job_id = job_id or str(uuid.uuid4())
        job_url = f"https://hud.ai/jobs/{implicit_job_id}"

        # Print job URL (not individual trace URLs)
        print_link(job_url, f"🚀 Job '{eval_name}'")

        error_occurred = False
        try:
            # Run parallel evals with job_id
            completed = await _run_parallel_eval(
                tasks=tasks,
                variant_combos=variant_combos,
                group=group,
                group_ids=group_ids,
                job_id=implicit_job_id,  # Propagate job_id to child traces
                api_key=api_key,
                code_snippet=code_snippet,
                max_concurrent=max_concurrent,
            )

            # Create summary context (no trace, just aggregates results)
            if tasks:
                ctx = EvalContext.from_task(
                    task=tasks[0],
                    api_key=api_key,
                    job_id=implicit_job_id,
                )
            else:
                ctx = EvalContext(
                    name="eval",
                    api_key=api_key,
                    job_id=implicit_job_id,
                )

            ctx._is_summary = True  # Skip trace tracking
            ctx.results = completed

            # Compute aggregate reward
            rewards = [e.reward for e in completed if e.reward is not None]
            if rewards:
                ctx.reward = sum(rewards) / len(rewards)

            # Check if any failed
            error_occurred = any(e.error is not None for e in completed)

            yield ctx
        except Exception:
            error_occurred = True
            raise
        finally:
            print_complete(job_url, eval_name, error=error_occurred)


async def _run_parallel_eval(
    tasks: list[Task],
    variant_combos: list[dict[str, Any]],
    group: int,
    group_ids: list[str] | None,
    job_id: str | None,
    api_key: str | None,
    code_snippet: str | None,
    max_concurrent: int | None,
) -> list[EvalContext]:
    """Run parallel evaluation.

    Creates EvalContexts from tasks (or blank) and runs them in parallel.
    """
    # Lazy import to avoid circular dependency
    from hud.eval.context import EvalContext
    from hud.eval.parallel import log_eval_stats, run_parallel_evals

    # Find user code frame and extract the with block body
    caller_frame = find_user_frame()
    body_source, captured_locals, context_var = get_with_block_body(caller_frame)

    # Calculate total evals and resolve group IDs
    total_evals = len(tasks) * len(variant_combos) * group if tasks else len(variant_combos) * group
    resolved_group_ids = resolve_group_ids(group_ids, total_evals)

    # Create EvalContexts
    eval_contexts: list[EvalContext] = []
    idx = 0

    if tasks:
        # Create context for each (task, variant, run) combination
        for task in tasks:
            for variant in variant_combos:
                for _ in range(group):
                    ctx = EvalContext.from_task(
                        task=task,
                        api_key=api_key,
                        job_id=job_id,
                        group_id=resolved_group_ids[idx],
                        index=idx,
                        variants=variant,
                        code_snippet=code_snippet,
                    )
                    ctx._suppress_link = True  # Suppress individual links, job URL shown instead
                    eval_contexts.append(ctx)
                    idx += 1
    else:
        # Blank evals for each (variant, run) combination
        for variant in variant_combos:
            for _ in range(group):
                ctx = EvalContext(
                    name="eval",
                    api_key=api_key,
                    job_id=job_id,
                    group_id=resolved_group_ids[idx],
                    index=idx,
                    variants=variant,
                    code_snippet=code_snippet,
                )
                ctx._suppress_link = True  # Suppress individual links, job URL shown instead
                eval_contexts.append(ctx)
                idx += 1

    # Run in parallel
    logger.info(
        "Running %d evals (%d tasks x %d variants x %d runs)%s",
        len(eval_contexts),
        max(len(tasks), 1),
        len(variant_combos),
        group,
        f", max_concurrent={max_concurrent}" if max_concurrent else "",
    )
    completed = await run_parallel_evals(
        eval_contexts, body_source, captured_locals, context_var, max_concurrent
    )

    # Log and print stats
    eval_name = completed[0].eval_name if completed else "eval"
    log_eval_stats(completed)
    print_eval_stats(completed, eval_name)

    return completed


__all__ = ["run_eval"]
