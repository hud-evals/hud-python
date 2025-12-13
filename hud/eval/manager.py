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
from hud.eval.types import ParallelEvalComplete

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from hud.eval.context import EvalContext
    from hud.eval.task import Task

logger = logging.getLogger(__name__)


def _get_eval_name(tasks: list[Task] | None = None) -> str:
    """Extract a nice name for job display.

    Args:
        tasks: List of Task objects

    Returns:
        Name like "scenario with val1, val2" or "eval" if no tasks
    """
    from hud.eval.task import build_eval_name

    # If we have Task objects, derive name from first one
    if tasks:
        if tasks[0].scenario:
            return build_eval_name(tasks[0].scenario, tasks[0].args)
        # Fall back to env name or prompt
        if tasks[0].env and hasattr(tasks[0].env, "name"):
            return tasks[0].env.name
        if tasks[0].env and hasattr(tasks[0].env, "prompt") and tasks[0].env.prompt:
            return tasks[0].env.prompt[:30].strip()
        if tasks[0].id:
            return tasks[0].id

    return "eval"


@asynccontextmanager
async def run_eval(
    source: Task | list[Task] | None = None,
    *,
    variants: dict[str, Any] | None = None,
    group: int = 1,
    group_ids: list[str] | None = None,
    job_id: str | None = None,
    api_key: str | None = None,
    max_concurrent: int | None = None,
    trace: bool = True,
    quiet: bool = False,
) -> AsyncGenerator[EvalContext, None]:
    """Standalone eval context manager.

    Creates an EvalContext for evaluation using Task objects (or deprecated LegacyTask).
    For loading tasks from datasets, use load_dataset() first.

    Args:
        source: Task source. Can be:
            - None: Create blank eval context
            - Task: Single Task object (from env() or load_dataset())
            - list[Task]: List of Task objects
            - LegacyTask: Single LegacyTask object (deprecated, use Task.from_v4())
            - list[LegacyTask]: List of LegacyTask objects (deprecated)
        variants: A/B test configuration (dict with list values expanded)
        group: Runs per variant for statistical significance
        group_ids: Optional list of group IDs
        job_id: Job ID to link to
        api_key: API key for backend calls
        max_concurrent: Maximum concurrent evals (None = unlimited)
        trace: Whether to send trace data to backend (default True)
        quiet: Whether to suppress printing links (default False)

    Yields:
        EvalContext: Environment with evaluation tracking

    Example:
        ```python
        from hud.datasets import load_dataset

        # Blank eval (for manual reward)
        async with hud.eval() as ctx:
            ctx.reward = compute_reward()

        # With Task objects (from env())
        env = Environment("my-env").connect_hub("browser")
        tasks = [env("checkout", user_id="alice"), env("checkout", user_id="bob")]
        async with hud.eval(tasks, variants={"model": ["gpt-4o"]}, group=4) as ctx:
            await agent.run(ctx.prompt)

        # Load tasks from dataset first
        tasks = load_dataset("hud-evals/SheetBench-50")
        async with hud.eval(tasks) as ctx:
            await agent.run(ctx)

        # With variants and group
        async with hud.eval(
            tasks,
            variants={"model": ["gpt-4o", "claude"]},
            group=3,
        ) as ctx:
            model = ctx.variants["model"]
            await run_agent(model)
            ctx.reward = evaluate()

        # With concurrency limit
        async with hud.eval(tasks, max_concurrent=10) as ctx:
            await agent.run(ctx)

        # Access results after parallel run
        for e in ctx.results:
            print(f"{e.variants}: reward={e.reward}")
        ```
    """
    from hud.eval.task import Task
    from hud.types import LegacyTask

    if group <= 0:
        raise ValueError("group must be >= 1")

    # Expand variants
    variant_combos = expand_variants(variants)

    # Parse source into tasks list - only Task objects accepted
    tasks: list[Task] = []

    if source is not None:
        if isinstance(source, Task):
            # Single Task object
            tasks = [source]
        elif isinstance(source, list) and source and isinstance(source[0], Task):
            # List of Task objects
            tasks = source  # type: ignore[assignment]
        elif isinstance(source, LegacyTask) or (
            isinstance(source, list) and source and isinstance(source[0], LegacyTask)
        ):
            # LegacyTask no longer accepted - user must convert first
            raise TypeError(
                "LegacyTask is no longer accepted by hud.eval(). "
                "Convert first with Task.from_v4(legacy_task), or use load_dataset()."
            )
        elif isinstance(source, str):
            # String slugs no longer supported - use load_dataset()
            raise TypeError(
                f"String slugs are no longer supported in hud.eval(). "
                f"Use load_dataset('{source}') first, then pass the tasks list."
            )
        elif isinstance(source, list) and source and isinstance(source[0], str):
            # List of string slugs no longer supported
            raise TypeError(
                "String slugs are no longer supported in hud.eval(). "
                "Use load_dataset() first, then pass the tasks list."
            )

    # Calculate total evaluations
    # Each task gets (variants x group) runs; no tasks = single blank eval
    base_count = len(tasks) or 1
    total_evals = base_count * len(variant_combos) * group

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
        # Simple case: single eval - always use Task for consistent flow
        if tasks:
            single_task = tasks[0]
        else:
            # Blank eval
            single_task = Task(
                env=None,
                scenario=None,
                api_key=api_key,
                job_id=job_id,
                variants=variant_combos[0],
                code_snippet=code_snippet,
                _trace=trace,
                _quiet=quiet,
            )

        # Apply common settings
        single_task.api_key = api_key
        single_task.job_id = job_id
        single_task.variants = variant_combos[0]
        single_task.code_snippet = code_snippet
        single_task._trace = trace
        single_task._quiet = quiet

        async with single_task as ctx:
            yield ctx

    else:
        # Parallel execution: create implicit job to group traces
        eval_name = _get_eval_name(tasks=tasks)
        implicit_job_id = job_id or str(uuid.uuid4())
        job_url = f"https://hud.ai/jobs/{implicit_job_id}"

        # Print job URL (not individual trace URLs)
        if not quiet:
            print_link(job_url, f"🚀 {eval_name}")

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
                trace=trace,
                quiet=quiet,
            )

            # Create summary context (no trace, just aggregates results)
            if tasks:
                # Create summary from first task
                ctx = EvalContext(
                    name=eval_name,  # Use the same smart name
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
        except ParallelEvalComplete:
            # Expected - body re-executed on summary context, skip it
            pass
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
    trace: bool = True,
    quiet: bool = False,
) -> list[EvalContext]:
    """Run parallel evaluation.

    Creates EvalContexts from Tasks (or blank) and runs them in parallel.
    """
    import asyncio
    import textwrap

    from hud.eval.parallel import log_eval_stats

    # Lazy import to avoid circular dependency
    from hud.eval.task import Task

    # Find user code frame and extract the with block body
    caller_frame = find_user_frame()
    body_source, captured_locals, context_var = get_with_block_body(caller_frame)

    # Calculate total evals and resolve group IDs
    base_count = len(tasks) or 1
    total_evals = base_count * len(variant_combos) * group
    resolved_group_ids = resolve_group_ids(group_ids, total_evals)

    # Create Task objects for parallel execution
    task_objects: list[Task] = []
    idx = 0

    if tasks:
        # Create Task for each (task, variant, run) combination
        for base_task in tasks:
            for variant in variant_combos:
                for _ in range(group):
                    task_copy = base_task.copy()
                    task_copy.api_key = api_key
                    task_copy.job_id = job_id
                    task_copy.group_id = resolved_group_ids[idx]
                    task_copy.index = idx
                    task_copy.variants = variant
                    task_copy.code_snippet = code_snippet
                    task_copy._suppress_link = True  # Individual traces don't print links
                    task_copy._trace = trace
                    task_copy._quiet = quiet
                    task_objects.append(task_copy)
                    idx += 1
    else:
        # Blank tasks for each (variant, run) combination
        for variant in variant_combos:
            for _ in range(group):
                blank_task = Task(
                    env=None,
                    scenario=None,
                    args={},
                    api_key=api_key,
                    job_id=job_id,
                    group_id=resolved_group_ids[idx],
                    index=idx,
                    variants=variant,
                    code_snippet=code_snippet,
                    _suppress_link=True,
                    _trace=trace,
                    _quiet=quiet,
                )
                task_objects.append(blank_task)
                idx += 1

    # Create runner function using the actual variable name from the 'as' clause
    wrapped = f"async def __runner__({context_var}):\n{textwrap.indent(body_source, '    ')}"
    code = compile(wrapped, "<parallel_eval>", "exec")
    namespace = captured_locals.copy()
    exec(code, namespace)  # noqa: S102
    runner = namespace["__runner__"]

    # Create semaphore for concurrency control
    sem = asyncio.Semaphore(max_concurrent) if max_concurrent else None

    async def run_one(task_obj: Task) -> EvalContext:
        """Run a single Task and return its EvalContext."""
        try:
            if sem:
                async with sem, task_obj as ctx:
                    await runner(ctx)
            else:
                async with task_obj as ctx:
                    await runner(ctx)
            return ctx
        except Exception as e:
            logger.warning("Parallel eval %d failed: %s", task_obj.index, e)
            # Create a failed context from the task
            ctx = task_obj.to_eval_context()
            ctx.error = e
            return ctx

    # Run in parallel
    logger.info(
        "Running %d tasks (%d base x %d variants x %d runs)%s",
        len(task_objects),
        base_count,
        len(variant_combos),
        group,
        f", max_concurrent={max_concurrent}" if max_concurrent else "",
    )
    completed = await asyncio.gather(*[run_one(t) for t in task_objects])

    # Log and print stats
    eval_name = completed[0].eval_name if completed else "eval"
    log_eval_stats(completed)
    print_eval_stats(completed, eval_name)

    return list(completed)


__all__ = ["run_eval"]
