"""Parallel execution support for traces.

This module provides AST extraction and parallel execution for running
the same trace body N times concurrently.
"""

from __future__ import annotations

import ast
import asyncio
import linecache
import logging
import textwrap
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from hud.trace.context import TraceContext

logger = logging.getLogger(__name__)


class ASTExtractionError(Exception):
    """Error extracting AST from source."""


def _get_with_block_body(frame: Any) -> tuple[str, dict[str, Any]]:
    """Extract the body of a with-block from the calling frame.
    
    Args:
        frame: The calling frame (from inspect.currentframe())
        
    Returns:
        Tuple of (body_source, captured_locals)
    """
    filename = frame.f_code.co_filename
    lineno = frame.f_lineno
    
    # Check for interactive session
    if filename.startswith("<") or filename in ("<stdin>", "<string>"):
        raise ASTExtractionError(
            "Cannot extract source from interactive session. Use a .py file."
        )
    
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
        raise ASTExtractionError(
            f"Cannot find 'async with' statement at line {lineno}"
        )
    
    # Extract body source
    body_source = _extract_body(lines, with_node)
    
    return body_source, frame.f_locals.copy()


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


async def run_parallel_traces(
    trace_contexts: list[TraceContext],
    body_source: str,
    captured_locals: dict[str, Any],
) -> list[TraceContext]:
    """Run the trace body in parallel for multiple contexts.
    
    Returns the TraceContext objects after execution - they contain:
    - trace_id
    - index
    - reward
    - duration
    - Any error is captured in the context
    """
    
    # Create runner function
    wrapped = f"async def __runner__(tc):\n{textwrap.indent(body_source, '    ')}"
    code = compile(wrapped, "<parallel_trace>", "exec")
    namespace = captured_locals.copy()
    exec(code, namespace)  # noqa: S102
    runner = namespace["__runner__"]
    
    async def run_one(tc: TraceContext) -> TraceContext:
        try:
            async with tc:
                await runner(tc)
        except Exception as e:
            logger.warning("Parallel trace %d failed: %s", tc.index, e)
            # Store error in context for inspection
            tc._error = e  # type: ignore[attr-defined]
        return tc
    
    results = await asyncio.gather(*[run_one(tc) for tc in trace_contexts])
    return list(results)
