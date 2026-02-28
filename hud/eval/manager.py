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


def _get_eval_name(tasks: list[Task] | None = None, group: int = 1) -> str:
    """Build a job display name.

    Convention:
        1 task, group=1:  "Task Run: {scenario}"
        1 task, group>1:  "Task Run: {scenario} (4 times)"
        N tasks, group=1: "Batch Run: N tasks"
        N tasks, group>1: "Batch Run: N tasks (4 times)"
    """
    suffix = f" ({group} times)" if group > 1 else ""

    if not tasks:
        return f"Task Run: eval{suffix}"

    if len(tasks) == 1:
        name = tasks[0].scenario
        if not name and tasks[0].env and hasattr(tasks[0].env, "name"):
            name = tasks[0].env.name
        return f"Task Run: {name or 'eval'}{suffix}"

    return f"Batch Run: {len(tasks)} tasks{suffix}"


async def _send_job_enter(
    job_id: str,
    name: str,
    variants: dict[str, Any] | None,
    group: int,
    api_key: str | None,
    taskset_id: str | None = None,
    hud_eval_config: dict[str, Any] | None = None,
) -> None:
    """Send job enter payload (async request before traces start).

    Registers the job with the platform.
    """
    import httpx

    from hud.eval.types import JobEnterPayload
    from hud.settings import settings

    api_key = api_key or settings.api_key
    if not settings.telemetry_enabled or not api_key:
        return

    payload = JobEnterPayload(
        name=name,
        variants=variants,
        group=group,
        taskset_id=taskset_id,
        hud_eval_config=hud_eval_config,
    )

    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.post(
            f"{settings.hud_api_url}/trace/job/{job_id}/enter",
            json=payload.model_dump(exclude_none=True),
            headers={"Authorization": f"Bearer {api_key}"},
        )

    resp.raise_for_status()


@asynccontextmanager
async def run_eval(
    source: Task | list[Task] | None = None,
    *,
    name: str | None = None,
    variants: dict[str, Any] | None = None,
    group: int = 1,
    group_ids: list[str] | None = None,
    job_id: str | None = None,
    group_id: str | None = None,
    trace_id: str | None = None,
    api_key: str | None = None,
    max_concurrent: int | None = None,
    taskset_id: str | None = None,
    trace: bool = True,
    quiet: bool = False,
) -> AsyncGenerator[EvalContext, None]:
    """Standalone eval context manager.

    Creates an EvalContext for evaluation using Task objects (or deprecated LegacyTask).
    For loading tasks from datasets, use load_tasks() first.

    Args:
        source: Task source. Can be:
            - None: Create blank eval context
            - Task: Single Task object (from env() or load_tasks())
            - list[Task]: List of Task objects
            - LegacyTask: Single LegacyTask object (deprecated, use Task.from_v4())
            - list[LegacyTask]: List of LegacyTask objects (deprecated)
        name: Optional name for the eval (used in trace)
        variants: A/B test configuration (dict with list values expanded)
        group: Runs per variant for statistical significance
        group_ids: Optional list of group IDs
        job_id: Pre-registered job ID. Skips implicit job creation if provided.
        group_id: Group ID for parallel evaluations
        trace_id: Pre-assigned trace ID (auto-generated if not provided)
        api_key: API key for backend calls
        max_concurrent: Maximum concurrent evals (None = unlimited)
        taskset_id: Taskset UUID to associate the job with on the platform.
        trace: Whether to send trace data to backend (default True)
        quiet: Whether to suppress printing links (default False)

    Yields:
        EvalContext: Environment with evaluation tracking

    Example:
        ```python
        from hud.datasets import load_tasks

        # Blank eval (for manual reward)
        async with hud.eval() as ctx:
            ctx.reward = compute_reward()

        # With Task objects (from env())
        env = Environment("my-env").connect_hub("browser")
        tasks = [env("checkout", user_id="alice"), env("checkout", user_id="bob")]
        async with hud.eval(tasks, variants={"model": ["gpt-4o"]}, group=4) as ctx:
            await agent.run(ctx.prompt)

        # Load tasks from file or API
        tasks = load_tasks("hud-evals/SheetBench-50")
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
                "Convert first with Task.from_v4(legacy_task), or use load_tasks()."
            )
        elif isinstance(source, str):
            # String slugs no longer supported - use load_dataset()
            raise TypeError(
                f"String slugs are no longer supported in hud.eval(). "
                f"Use load_tasks('{source}') first, then pass the tasks list."
            )
        elif isinstance(source, list) and source and isinstance(source[0], str):
            # List of string slugs no longer supported
            raise TypeError(
                "String slugs are no longer supported in hud.eval(). "
                "Use load_tasks() first, then pass the tasks list."
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

    # Register job if not already provided by caller
    eval_name = _get_eval_name(tasks=tasks, group=group)
    if not job_id and (taskset_id or total_evals > 1):
        job_id = str(uuid.uuid4())
        await _send_job_enter(
            job_id=job_id,
            name=eval_name,
            variants=variants,
            group=group,
            api_key=api_key,
            taskset_id=taskset_id,
        )

    if total_evals == 1:
        if tasks:
            ctx = EvalContext.from_task(
                tasks[0],
                name=name,
                trace_id=trace_id,
                api_key=api_key,
                job_id=job_id,
                group_id=group_id,
                variants=variant_combos[0],
                code_snippet=code_snippet,
                trace=trace,
                quiet=quiet,
            )
            async with ctx:
                yield ctx
        else:
            ctx = EvalContext(
                name=name or "eval",
                trace_id=trace_id,
                api_key=api_key,
                job_id=job_id,
                group_id=group_id,
                variants=variant_combos[0],
                code_snippet=code_snippet,
                trace=trace,
                quiet=quiet,
            )
            async with ctx:
                yield ctx

    else:
        job_url = f"https://hud.ai/jobs/{job_id}"

        if not quiet:
            print_link(job_url, f"🚀 {eval_name}")

        error_occurred = False
        try:
            completed = await _run_parallel_eval(
                tasks=tasks,
                variant_combos=variant_combos,
                group=group,
                group_ids=group_ids,
                job_id=job_id,
                api_key=api_key,
                code_snippet=code_snippet,
                max_concurrent=max_concurrent,
                trace=trace,
                quiet=quiet,
            )

            ctx = EvalContext(
                name=eval_name,
                api_key=api_key,
                job_id=job_id,
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

    # Find user code frame and extract the with block body
    caller_frame = find_user_frame()
    body_source, captured_locals, context_var = get_with_block_body(caller_frame)

    # Calculate total evals and resolve group IDs
    base_count = len(tasks) or 1
    total_evals = base_count * len(variant_combos) * group
    resolved_group_ids = resolve_group_ids(group_ids, total_evals)

    # Build list of (task_or_none, runtime_params) for each parallel eval
    from hud.eval.context import EvalContext

    eval_configs: list[tuple[Task | None, dict[str, Any]]] = []
    idx = 0

    if tasks:
        for base_task in tasks:
            for variant in variant_combos:
                for _ in range(group):
                    runtime_params = {
                        "api_key": api_key,
                        "job_id": job_id,
                        "group_id": resolved_group_ids[idx],
                        "index": idx,
                        "variants": variant,
                        "code_snippet": code_snippet,
                        "trace": trace,
                        "quiet": True,  # Individual traces don't print links
                    }
                    eval_configs.append((base_task, runtime_params))
                    idx += 1
    else:
        for variant in variant_combos:
            for _ in range(group):
                runtime_params = {
                    "api_key": api_key,
                    "job_id": job_id,
                    "group_id": resolved_group_ids[idx],
                    "index": idx,
                    "variants": variant,
                    "code_snippet": code_snippet,
                    "trace": trace,
                    "quiet": True,
                }
                eval_configs.append((None, runtime_params))
                idx += 1

    # Create runner function using the actual variable name from the 'as' clause
    wrapped = f"async def __runner__({context_var}):\n{textwrap.indent(body_source, '    ')}"
    code = compile(wrapped, "<parallel_eval>", "exec")
    namespace = captured_locals.copy()
    exec(code, namespace)  # noqa: S102
    runner = namespace["__runner__"]

    # Create semaphore for concurrency control
    sem = asyncio.Semaphore(max_concurrent) if max_concurrent else None

    async def run_one(config: tuple[Task | None, dict[str, Any]]) -> EvalContext:
        """Run a single eval and return its EvalContext."""
        task, params = config
        idx = params["index"]

        # Create context from task or blank
        if task is not None:
            ctx = EvalContext.from_task(task, **params)
        else:
            ctx = EvalContext(name="eval", **params)

        # Remove sensitive data from params after context creation to prevent
        # accidental logging if an exception includes local variables
        params.pop("api_key", None)

        try:
            if sem:
                async with sem, ctx:
                    await runner(ctx)
            else:
                async with ctx:
                    await runner(ctx)
            return ctx
        except Exception as e:
            logger.warning("Parallel eval %d failed: %s", idx, e)
            ctx.error = e
            return ctx

    # Run in parallel
    logger.info(
        "Running %d evals (%d base x %d variants x %d runs)%s",
        len(eval_configs),
        base_count,
        len(variant_combos),
        group,
        f", max_concurrent={max_concurrent}" if max_concurrent else "",
    )
    completed = await asyncio.gather(*[run_one(cfg) for cfg in eval_configs])

    # Log and print stats
    eval_name = completed[0].eval_name if completed else "eval"
    log_eval_stats(completed)
    print_eval_stats(completed, name=eval_name)

    return list(completed)


__all__ = ["run_eval"]
