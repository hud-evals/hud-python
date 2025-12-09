"""Standalone eval() context manager.

Provides hud.eval() for task-based evaluation without needing an existing environment.
"""

from __future__ import annotations

import inspect
import logging
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

from hud.eval.parallel import (
    ASTExtractionError,
    execute_parallel_evals,
    expand_variants,
    get_with_block_body,
    resolve_group_ids,
)

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from hud.eval.context import EvalContext
    from hud.types import Task

logger = logging.getLogger(__name__)


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
                    for item in data:
                        tasks.append(Task(**item))
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
    slugs: str | list[str] | None = None,
    *,
    variants: dict[str, Any] | None = None,
    group: int = 1,
    group_ids: list[str] | None = None,
    job_id: str | None = None,
    api_key: str | None = None,
) -> AsyncGenerator[EvalContext, None]:
    """Standalone eval context manager.

    Creates an EvalContext for evaluation, optionally loading task configuration
    from slugs.

    Args:
        slugs: Task slug(s) to load. Can be:
            - None: Create blank eval context
            - "my-org/task": Single task
            - "my-org/task:N": Task at index N
            - "my-org/task:*": All tasks matching pattern
            - List of any above: Multiple tasks
        variants: A/B test configuration (dict with list values expanded)
        group: Runs per variant for statistical significance
        group_ids: Optional list of group IDs
        job_id: Job ID to link to
        api_key: API key for backend calls

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

        # With variants and group
        async with hud.eval(
            "task",
            variants={"model": ["gpt-4o", "claude"]},
            group=3,
        ) as ctx:
            model = ctx.variants["model"]
            await run_agent(model)
            ctx.reward = evaluate()

        # Access results after parallel run
        for e in ctx.results:
            print(f"{e.variants}: reward={e.reward}")
        ```
    """
    if group <= 0:
        raise ValueError("group must be >= 1")

    # Expand variants
    variant_combos = expand_variants(variants)

    # Load tasks if slugs provided
    tasks: list[Task] = []
    if slugs is not None:
        tasks = _load_tasks_from_slugs(slugs)

    # Calculate total evaluations
    # If we have tasks, each task gets (variants x group) runs
    # If no tasks, we have a single blank eval with (variants x group) runs
    if tasks:
        total_evals = len(tasks) * len(variant_combos) * group
    else:
        total_evals = len(variant_combos) * group

    # Capture code snippet for parallel execution
    code_snippet: str | None = None
    if total_evals > 1:
        frame = inspect.currentframe()
        if frame is not None:
            try:
                caller = frame.f_back
                if caller is not None:
                    code_snippet, _ = get_with_block_body(caller)
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
        # Parallel execution
        completed = await _run_parallel_eval(
            tasks=tasks,
            variant_combos=variant_combos,
            group=group,
            group_ids=group_ids,
            job_id=job_id,
            api_key=api_key,
            code_snippet=code_snippet,
        )

        # Create parent ctx with results
        if tasks:
            ctx = EvalContext.from_task(
                task=tasks[0],
                api_key=api_key,
                job_id=job_id,
            )
        else:
            ctx = EvalContext(
                name="eval",
                api_key=api_key,
                job_id=job_id,
            )

        ctx.results = completed

        # Compute aggregate reward
        rewards = [e.reward for e in completed if e.reward is not None]
        if rewards:
            ctx.reward = sum(rewards) / len(rewards)

        yield ctx


async def _run_parallel_eval(
    tasks: list[Task],
    variant_combos: list[dict[str, Any]],
    group: int,
    group_ids: list[str] | None,
    job_id: str | None,
    api_key: str | None,
    code_snippet: str | None,
) -> list[EvalContext]:
    """Run parallel evaluation.

    Creates EvalContexts from tasks (or blank) and runs them in parallel.
    """
    # Lazy import to avoid circular dependency
    from hud.eval.context import EvalContext

    # Calculate total evals and resolve group IDs
    if tasks:
        total_evals = len(tasks) * len(variant_combos) * group
    else:
        total_evals = len(variant_combos) * group

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
                eval_contexts.append(ctx)
                idx += 1

    # Run in parallel (frame depth: _run_parallel_eval -> eval -> user code)
    return await execute_parallel_evals(eval_contexts, caller_frame_depth=3)


__all__ = ["run_eval"]

