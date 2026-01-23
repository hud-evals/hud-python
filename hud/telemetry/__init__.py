"""HUD Telemetry - Lightweight telemetry for HUD SDK.

This module provides:
- @instrument decorator for recording function calls
- High-performance span export to HUD API
- parallel_agent_group context manager for tracking parallel agent execution

Usage:
    import hud

    @hud.instrument
    async def my_function():
        ...

    # Within an eval context, calls are recorded
    async with hud.eval(task) as ctx:
        result = await my_function()

    # Track parallel agents
    from hud.telemetry import parallel_agent_group

    async with parallel_agent_group(
        title="Deep Research",
        description="Collect profiles...",
        agents=[{"name": "Worker 1"}, {"name": "Worker 2"}],
    ) as group:
        # Run agents in parallel...
        pass
"""

from hud.telemetry.exporter import flush, queue_span, shutdown
from hud.telemetry.instrument import instrument
from hud.telemetry.parallel_group import (
    ParallelAgentGroup,
    ParallelAgentInfo,
    parallel_agent_group,
)

__all__ = [
    "ParallelAgentGroup",
    "ParallelAgentInfo",
    "flush",
    "instrument",
    "parallel_agent_group",
    "queue_span",
    "shutdown",
]
