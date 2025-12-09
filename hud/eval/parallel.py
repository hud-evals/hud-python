"""Parallel execution support for evaluations.

This module provides AST extraction and parallel execution for running
the same eval body N times concurrently.
"""

from __future__ import annotations

import ast
import asyncio
import inspect
import itertools
import linecache
import logging
import textwrap
import uuid
from types import FrameType
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from hud.eval.context import EvalContext

logger = logging.getLogger(__name__)

# Frames to skip when walking the call stack to find user code
# These are internal implementation details that shouldn't be considered user code
_SKIP_FRAME_PATTERNS = (
    # Python stdlib
    "contextlib.py",
    "asyncio",
    # Third-party
    "site-packages",
    # HUD eval internals (both Unix and Windows paths)
    "hud/eval/mixin.py",
    "hud/eval/manager.py",
    "hud/eval/parallel.py",
    "hud\\eval\\mixin.py",
    "hud\\eval\\manager.py",
    "hud\\eval\\parallel.py",
)


def find_user_frame() -> FrameType:
    """Walk the call stack to find the first user code frame.

    Skips internal frames from contextlib, asyncio, site-packages,
    and hud.eval internals.

    Returns:
        The frame containing user code (typically the async with statement).

    Raises:
        ASTExtractionError: If no user code frame can be found.
    """
    frame = inspect.currentframe()
    if frame is None:
        raise ASTExtractionError("Cannot get current frame")

    try:
        caller_frame = frame.f_back
        while caller_frame is not None:
            filename = caller_frame.f_code.co_filename
            # Stop at first frame not matching skip patterns
            if not any(pattern in filename for pattern in _SKIP_FRAME_PATTERNS):
                return caller_frame
            caller_frame = caller_frame.f_back

        raise ASTExtractionError("Cannot find user code frame in call stack")
    finally:
        del frame


def expand_variants(
    variants: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """Expand variants dict into all combinations.

    Args:
        variants: Dict where values can be:
            - Single value: {"model": "gpt-4o"} → fixed
            - List: {"model": ["gpt-4o", "claude"]} → expand

    Returns:
        List of variant assignments, one per combination.

    Examples:
        >>> expand_variants(None)
        [{}]
        >>> expand_variants({"model": "gpt-4o"})
        [{"model": "gpt-4o"}]
        >>> expand_variants({"model": ["gpt-4o", "claude"]})
        [{"model": "gpt-4o"}, {"model": "claude"}]
    """
    if not variants:
        return [{}]

    expanded: dict[str, list[Any]] = {}
    for key, value in variants.items():
        if isinstance(value, list):
            expanded[key] = value
        else:
            expanded[key] = [value]

    keys = list(expanded.keys())
    value_lists = [expanded[k] for k in keys]

    return [dict(zip(keys, combo, strict=True)) for combo in itertools.product(*value_lists)]


def resolve_group_ids(
    group_ids: list[str] | None,
    total_count: int,
) -> list[str]:
    """Resolve group IDs for parallel execution.

    Args:
        group_ids: Optional list of group IDs (must match total_count if provided)
        total_count: Total number of evals

    Returns:
        List of group IDs (one per eval)

    Raises:
        ValueError: If group_ids length doesn't match total_count
    """
    if group_ids:
        if len(group_ids) != total_count:
            raise ValueError(
                f"group_ids length ({len(group_ids)}) must match total evals ({total_count})"
            )
        return group_ids
    else:
        shared_group_id = str(uuid.uuid4())
        return [shared_group_id] * total_count


def log_eval_stats(completed: list[EvalContext], context: str = "") -> None:
    """Log statistics for completed evaluations.

    Args:
        completed: List of completed EvalContext objects
        context: Optional context string for the log message
    """
    rewards = [ctx.reward for ctx in completed if ctx.reward is not None]
    mean_reward = sum(rewards) / len(rewards) if rewards else 0.0
    success_count = sum(1 for ctx in completed if ctx.success)

    logger.info(
        "Evals complete%s: %d/%d succeeded, mean_reward=%.3f",
        f" ({context})" if context else "",
        success_count,
        len(completed),
        mean_reward,
    )


async def execute_parallel_evals(
    contexts: list[EvalContext],
    caller_frame_depth: int = 2,
) -> list[EvalContext]:
    """Execute evaluations in parallel using AST extraction.

    This is the shared implementation for parallel execution. It:
    1. Captures the caller's frame and extracts with-block body
    2. Runs all provided EvalContexts in parallel
    3. Logs statistics

    Args:
        contexts: Pre-created EvalContext instances to run
        caller_frame_depth: How many frames to go up to find user code
                           (default 2: execute_parallel_evals -> caller -> user)

    Returns:
        List of completed EvalContext objects with results
    """
    import inspect

    # Get the caller's frame
    frame = inspect.currentframe()
    if frame is None:
        raise ASTExtractionError("Cannot get current frame")

    try:
        # Go up the specified number of frames
        caller_frame = frame
        for _ in range(caller_frame_depth):
            if caller_frame is not None:
                caller_frame = caller_frame.f_back
        if caller_frame is None:
            raise ASTExtractionError("Cannot get caller frame")

        body_source, captured_locals, context_var = get_with_block_body(caller_frame)

    finally:
        del frame

    # Run in parallel
    logger.info("Running %d parallel evals", len(contexts))
    completed = await run_parallel_evals(contexts, body_source, captured_locals, context_var)

    # Log stats
    log_eval_stats(completed)

    return completed


class ASTExtractionError(Exception):
    """Error extracting AST from source."""


def get_with_block_body(frame: Any) -> tuple[str, dict[str, Any], str]:
    """Extract the body of a with-block from the calling frame.

    Args:
        frame: The calling frame (from inspect.currentframe())

    Returns:
        Tuple of (body_source, captured_locals, context_var_name)
    """
    filename = frame.f_code.co_filename
    lineno = frame.f_lineno

    # Check for interactive session
    if filename.startswith("<") or filename in ("<stdin>", "<string>"):
        raise ASTExtractionError("Cannot extract source from interactive session. Use a .py file.")

    # Read and parse source
    lines = linecache.getlines(filename)
    if not lines:
        with open(filename, encoding="utf-8") as f:
            lines = f.readlines()

    source = "".join(lines)
    tree = ast.parse(source, filename=filename)

    # Find the async with containing this line
    with_node = _find_async_with(tree, lineno)
    if with_node is None:
        raise ASTExtractionError(f"Cannot find 'async with' statement at line {lineno}")

    # Extract body source
    body_source = _extract_body(lines, with_node)

    # Extract the context variable name from 'as' clause
    context_var = _extract_context_var(with_node)

    return body_source, frame.f_locals.copy(), context_var


def _extract_context_var(with_node: ast.AsyncWith) -> str:
    """Extract the variable name from the 'as' clause of an async with statement."""
    if not with_node.items or not with_node.items[0].optional_vars:
        raise ASTExtractionError("async with statement must use 'as' clause for parallel execution")

    var_node = with_node.items[0].optional_vars
    if not isinstance(var_node, ast.Name):
        raise ASTExtractionError("async with 'as' clause must be a simple variable name")

    return var_node.id


def _find_async_with(tree: ast.AST, target_line: int) -> ast.AsyncWith | None:
    """Find AsyncWith node containing the target line."""
    for node in ast.walk(tree):
        if isinstance(node, ast.AsyncWith):
            end_line = _get_end_line(node)
            if node.lineno <= target_line <= end_line:
                return node
    return None


def _get_end_line(node: ast.AST) -> int:
    """Get the last line number of an AST node."""
    end = getattr(node, "end_lineno", getattr(node, "lineno", 0))
    for child in ast.walk(node):
        child_end = getattr(child, "end_lineno", 0)
        if child_end > end:
            end = child_end
    return end


def _extract_body(lines: list[str], with_node: ast.AsyncWith) -> str:
    """Extract the body source from an AsyncWith node."""
    if not with_node.body:
        return "pass"

    start = with_node.body[0].lineno - 1
    end = _get_end_line(with_node.body[-1])

    body = "".join(lines[start:end])
    return textwrap.dedent(body)


async def run_parallel_evals(
    eval_contexts: list[EvalContext],
    body_source: str,
    captured_locals: dict[str, Any],
    context_var: str,
) -> list[EvalContext]:
    """Run the eval body in parallel for multiple contexts.

    Returns the EvalContext objects after execution - they contain:
    - trace_id
    - index
    - reward
    - duration
    - Any error is captured in the context

    Args:
        eval_contexts: List of EvalContext instances to run
        body_source: The source code of the with-block body
        captured_locals: Local variables captured from the caller
        context_var: The variable name used in the 'as' clause
    """

    # Create runner function using the actual variable name from the 'as' clause
    wrapped = f"async def __runner__({context_var}):\n{textwrap.indent(body_source, '    ')}"
    code = compile(wrapped, "<parallel_eval>", "exec")
    namespace = captured_locals.copy()
    exec(code, namespace)  # noqa: S102
    runner = namespace["__runner__"]

    async def run_one(ctx: EvalContext) -> EvalContext:
        try:
            async with ctx:
                await runner(ctx)
        except Exception as e:
            logger.warning("Parallel eval %d failed: %s", ctx.index, e)
            ctx.error = e
        return ctx

    results = await asyncio.gather(*[run_one(ctx) for ctx in eval_contexts])
    return list(results)


__all__ = [
    "ASTExtractionError",
    "execute_parallel_evals",
    "expand_variants",
    "get_with_block_body",
    "log_eval_stats",
    "resolve_group_ids",
    "run_parallel_evals",
]

