"""Trace retrieval and replay functionality.

This module provides APIs to retrieve collected traces for analysis,
debugging, and replay purposes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from hud.otel.collector import clear_trace as _clear_trace
from hud.otel.collector import get_trace as _get_trace

if TYPE_CHECKING:
    from hud.types import Trace

__all__ = ["clear_trace", "get_trace"]


def get_trace(task_run_id: str) -> Trace | None:
    """Retrieve the collected trace for a task run.

    Returns None if trace collection was disabled or the trace doesn't exist.

    Args:
        task_run_id: The task run ID to retrieve the trace for

    Returns:
        Trace object containing all collected steps, or None if not found

    Usage:
        import hud

        # Run agent with tracing
        with hud.trace() as task_run_id:
            agent = MyAgent()
            result = await agent.run("solve task")

        # Get the trace for analysis
        trace = hud.get_trace(task_run_id)
        if trace:
            print(f"Collected {len(trace.trace)} steps")

            # Analyze agent vs MCP steps
            agent_steps = [s for s in trace.trace if s.category == "agent"]
            mcp_steps = [s for s in trace.trace if s.category == "mcp"]

            print(f"Agent steps: {len(agent_steps)}")
            print(f"MCP steps: {len(mcp_steps)}")

            # Replay or analyze individual steps
            for step in trace.trace:
                if step.category == "agent" and step.result:
                    print(f"Agent: {step.result.get('content') if isinstance(step.result, dict) else step.result}")
                if step.category == "mcp" and step.request:
                    print(f"MCP: {step.request.method if hasattr(step.request, 'method') else step.request}")
    """  # noqa: E501
    return _get_trace(task_run_id)


def clear_trace(task_run_id: str) -> None:
    """Clear the collected trace for a task run ID.

    Useful for cleaning up memory after processing large traces.

    Args:
        task_run_id: The task run ID to clear the trace for

    Usage:
        trace = hud.get_trace(task_run_id)
        # Process trace...
        hud.clear_trace(task_run_id)  # Free memory
    """
    _clear_trace(task_run_id)
