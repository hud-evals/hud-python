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
from hud.eval.types import ParallelEvalComplete
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
    from hud.eval.eval import Eval
    from hud.types import Task

logger = logging.getLogger(__name__)


# Type alias for eval source: slug strings, Eval objects, or deprecated Task objects
EvalSource = "str | list[str] | Eval | list[Eval] | Task | list[Task] | None"


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
    evals: list[Eval] | None = None,
    tasks: list[Task] | None = None,  # Deprecated
) -> str:
    """Extract a nice name for job display.

    Args:
        source: Single slug or list of slugs (if string-based)
        evals: List of Eval objects (primary path)
        tasks: List of Task objects (deprecated)

    Returns:
        Name like "script with val1, val2" or "eval" if no source
    """
    from hud.eval.eval import build_eval_name

    # If we have Eval objects, derive name from first one
    if evals and evals[0].script:
        return build_eval_name(evals[0].script, evals[0].args)

    # Deprecated: If we have tasks with IDs, use first task ID
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


def _load_evals_from_slugs(slugs: str | list[str]) -> list[Eval]:
    """Load Eval configs from platform by slugs.

    Args:
        slugs: Single slug or list of slugs. Slugs can be:
            - "my-org/eval" - single eval
            - "my-org/eval:N" - eval at index N
            - "my-org/eval:*" - all evals matching pattern

    Returns:
        List of Eval objects
    """
    import httpx

    from hud.eval.eval import Eval
    from hud.settings import settings

    if isinstance(slugs, str):
        slugs = [slugs]

    evals: list[Eval] = []

    headers = {}
    if settings.api_key:
        headers["Authorization"] = f"Bearer {settings.api_key}"

    with httpx.Client() as client:
        for slug in slugs:
            base_slug, index_str = _parse_slug(slug)

            if index_str == "*":
                # Fetch all evals for this evalset
                logger.info("Loading all evals for: %s", base_slug)
                response = client.get(
                    f"{settings.hud_api_url}/evals/{base_slug}",
                    headers=headers,
                    params={"all": "true"},
                )
                response.raise_for_status()
                data = response.json()

                if isinstance(data, list):
                    evals.extend(_eval_from_api(item) for item in data)
                else:
                    evals.append(_eval_from_api(data))

            elif index_str is not None:
                # Fetch specific eval by index
                logger.info("Loading eval: %s (index %s)", base_slug, index_str)
                response = client.get(
                    f"{settings.hud_api_url}/evals/{base_slug}",
                    headers=headers,
                    params={"index": index_str},
                )
                response.raise_for_status()
                data = response.json()
                evals.append(_eval_from_api(data))

            else:
                # Fetch single eval
                logger.info("Loading eval: %s", slug)
                response = client.get(
                    f"{settings.hud_api_url}/evals/{slug}",
                    headers=headers,
                )
                response.raise_for_status()
                data = response.json()
                evals.append(_eval_from_api(data))

    return evals


def _eval_from_api(data: dict[str, Any]) -> Eval:
    """Convert API response to Eval object.

    Expected API response format:
    {
        "env_config": {...},  # EnvConfig dict
        "script": "script_name",  # Optional
        "args": {...},  # Script arguments
    }
    """
    from hud.eval.eval import Eval

    return Eval(
        env=data.get("env_config"),  # Serialized config from backend
        script=data.get("script"),
        args=data.get("args", {}),
    )


@asynccontextmanager
async def run_eval(
    source: str | list[str] | Task | list[Task] | Eval | list[Eval] | None = None,
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

    Creates an EvalContext for evaluation, optionally loading task configuration
    from slugs, using Task objects, or using Eval objects directly.

    Args:
        source: Eval source. Can be:
            - None: Create blank eval context
            - str: Task slug like "my-org/task", "my-org/task:N", "my-org/task:*"
            - list[str]: Multiple task slugs
            - Task: Single Task object (for backwards compat with run_tasks)
            - list[Task]: List of Task objects (for backwards compat with run_tasks)
            - Eval: Single Eval object (from env())
            - list[Eval]: List of Eval objects (from env())
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

        # With Eval objects (from env())
        env = Environment("my-env").connect_hub("browser")
        evals = [env("checkout", user_id="alice"), env("checkout", user_id="bob")]
        async with hud.eval(evals, variants={"model": ["gpt-4o"]}, group=4) as ctx:
            await agent.run(ctx.prompt)

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
    import warnings

    from hud.eval.eval import Eval
    from hud.types import Task

    if group <= 0:
        raise ValueError("group must be >= 1")

    # Expand variants
    variant_combos = expand_variants(variants)

    # Parse source into evals list (or deprecated tasks list)
    evals: list[Eval] = []
    tasks: list[Task] = []  # Deprecated path
    slugs: str | list[str] | None = None  # Track if we had string slugs (for naming)

    if source is not None:
        if isinstance(source, Eval):
            # Single Eval object
            evals = [source]
        elif isinstance(source, list) and source and isinstance(source[0], Eval):
            # List of Eval objects
            evals = source  # type: ignore[assignment]
        elif isinstance(source, Task):
            # Single Task object (deprecated)
            warnings.warn(
                "Passing Task objects to hud.eval() is deprecated. "
                "Use Eval objects from env() or string slugs instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            tasks = [source]
        elif isinstance(source, list) and source and isinstance(source[0], Task):
            # List of Task objects (deprecated)
            warnings.warn(
                "Passing Task objects to hud.eval() is deprecated. "
                "Use Eval objects from env() or string slugs instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            tasks = source  # type: ignore[assignment]
        elif isinstance(source, str):
            # String slug - load as Eval
            slugs = source
            evals = _load_evals_from_slugs(source)
        elif isinstance(source, list) and source and isinstance(source[0], str):
            # List of string slugs - load as Eval
            slugs = source  # type: ignore[assignment]
            evals = _load_evals_from_slugs(source)  # type: ignore[arg-type]

    # Calculate total evaluations
    # If we have evals, each eval gets (variants x group) runs
    # If we have tasks, each task gets (variants x group) runs
    # If neither, we have a single blank eval with (variants x group) runs
    base_count = len(evals) or len(tasks) or 1
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
        # Simple case: single eval - always use Eval for consistent flow
        if evals:
            single_eval = evals[0]
        elif tasks:
            # Wrap deprecated Task in Eval
            single_eval = Eval(
                env=None,
                script=None,
                api_key=api_key,
                job_id=job_id,
                variants=variant_combos[0],
                code_snippet=code_snippet,
                _trace=trace,
                _quiet=quiet,
            )
            single_eval._task = tasks[0]  # type: ignore[attr-defined]
        else:
            # Blank eval
            single_eval = Eval(
                env=None,
                script=None,
                api_key=api_key,
                job_id=job_id,
                variants=variant_combos[0],
                code_snippet=code_snippet,
                _trace=trace,
                _quiet=quiet,
            )
        
        # Apply common settings
        single_eval.api_key = api_key
        single_eval.job_id = job_id
        single_eval.variants = variant_combos[0]
        single_eval.code_snippet = code_snippet
        single_eval._trace = trace
        single_eval._quiet = quiet
        
        async with single_eval as ctx:
            yield ctx

    else:
        # Parallel execution: create implicit job to group traces
        eval_name = _get_eval_name(source=slugs, evals=evals, tasks=tasks)
        implicit_job_id = job_id or str(uuid.uuid4())
        job_url = f"https://hud.ai/jobs/{implicit_job_id}"

        # Print job URL (not individual trace URLs)
        if not quiet:
            print_link(job_url, f"🚀 {eval_name}")

        error_occurred = False
        try:
            # Run parallel evals with job_id
            completed = await _run_parallel_eval(
                evals=evals,
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
            if evals:
                # Create summary from first eval's env_config
                ctx = EvalContext(
                    name=eval_name,  # Use the same smart name
                    api_key=api_key,
                    job_id=implicit_job_id,
                    env_config=evals[0].env_config,
                )
            elif tasks:
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
        except ParallelEvalComplete:
            # Expected - body re-executed on summary context, skip it
            pass
        except Exception:
            error_occurred = True
            raise
        finally:
            print_complete(job_url, eval_name, error=error_occurred)


async def _run_parallel_eval(
    evals: list[Eval],
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

    Creates EvalContexts from Evals, tasks (or blank) and runs them in parallel.
    """
    import asyncio
    import textwrap

    # Lazy import to avoid circular dependency
    from hud.eval.context import EvalContext
    from hud.eval.eval import Eval
    from hud.eval.parallel import log_eval_stats

    # Find user code frame and extract the with block body
    caller_frame = find_user_frame()
    body_source, captured_locals, context_var = get_with_block_body(caller_frame)

    # Calculate total evals and resolve group IDs
    base_count = len(evals) or len(tasks) or 1
    total_evals = base_count * len(variant_combos) * group
    resolved_group_ids = resolve_group_ids(group_ids, total_evals)

    # Create Eval objects for parallel execution
    eval_objects: list[Eval] = []
    idx = 0

    if evals:
        # Create Eval for each (eval, variant, run) combination
        for base_eval in evals:
            for variant in variant_combos:
                for _ in range(group):
                    eval_copy = base_eval.copy()
                    eval_copy.api_key = api_key
                    eval_copy.job_id = job_id
                    eval_copy.group_id = resolved_group_ids[idx]
                    eval_copy.index = idx
                    eval_copy.variants = variant
                    eval_copy.code_snippet = code_snippet
                    eval_copy._suppress_link = True  # Individual traces don't print links
                    eval_copy._trace = trace
                    eval_copy._quiet = quiet
                    eval_objects.append(eval_copy)
                    idx += 1
    elif tasks:
        # Create Eval from Task for each (task, variant, run) combination
        for task in tasks:
            for variant in variant_combos:
                for _ in range(group):
                    # Convert Task to Eval (backwards compatibility)
                    task_eval = Eval(
                        env=None,  # Task has its own mcp_config
                        script=None,
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
                    # Store task reference for EvalContext creation
                    task_eval._task = task  # type: ignore[attr-defined]
                    eval_objects.append(task_eval)
                    idx += 1
    else:
        # Blank evals for each (variant, run) combination
        for variant in variant_combos:
            for _ in range(group):
                blank_eval = Eval(
                    env=None,
                    script=None,
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
                eval_objects.append(blank_eval)
                idx += 1

    # Create runner function using the actual variable name from the 'as' clause
    wrapped = f"async def __runner__({context_var}):\n{textwrap.indent(body_source, '    ')}"
    code = compile(wrapped, "<parallel_eval>", "exec")
    namespace = captured_locals.copy()
    exec(code, namespace)  # noqa: S102
    runner = namespace["__runner__"]

    # Create semaphore for concurrency control
    sem = asyncio.Semaphore(max_concurrent) if max_concurrent else None

    async def run_one(eval_obj: Eval) -> EvalContext:
        """Run a single Eval and return its EvalContext."""
        try:
            if sem:
                async with sem, eval_obj as ctx:
                    await runner(ctx)
            else:
                async with eval_obj as ctx:
                    await runner(ctx)
            return ctx
        except Exception as e:
            logger.warning("Parallel eval %d failed: %s", eval_obj.index, e)
            # Create a failed context from the eval
            ctx = eval_obj.to_eval_context()
            ctx.error = e
            return ctx

    # Run in parallel
    logger.info(
        "Running %d evals (%d base x %d variants x %d runs)%s",
        len(eval_objects),
        base_count,
        len(variant_combos),
        group,
        f", max_concurrent={max_concurrent}" if max_concurrent else "",
    )
    completed = await asyncio.gather(*[run_one(e) for e in eval_objects])

    # Log and print stats
    eval_name = completed[0].eval_name if completed else "eval"
    log_eval_stats(completed)
    print_eval_stats(completed, eval_name)

    return list(completed)


__all__ = ["run_eval"]
