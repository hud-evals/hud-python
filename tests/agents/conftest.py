"""Reading the tool steps an agent left on its trace."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mcp.types import TextContent

from hud.agents.types import ToolStep

if TYPE_CHECKING:
    from hud.eval import Run


def tool_results(run: Run) -> list[tuple[bool, str]]:
    """Each tool step on ``run``'s trace, in order, as (whether it failed, its text)."""
    results = []
    for step in run.trace.steps:
        if isinstance(step, ToolStep):
            assert step.result is not None
            text = "".join(b.text for b in step.result.content if isinstance(b, TextContent))
            results.append((step.result.isError, text))
    return results
