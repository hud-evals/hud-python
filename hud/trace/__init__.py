"""
HUD Trace System - Context management for agent runs.

The trace system provides:
- TraceContext: Core abstraction for recording agent runs
- TraceMixin: Mixin that adds trace() method to Environment
- Auto-instrumentation of httpx for inference.hud.ai
- Parallel execution with group=N

Usage (single execution):
    ```python
    async with env.trace("google-search") as tc:
        await tc.call_tool("navigate", {"url": "..."})
        tc.reward = 0.9

    # tc has the results
    print(tc.trace_id, tc.reward, tc.duration, tc.success)
    ```

Usage (parallel execution):
    ```python
    async with env.trace("google-search", group=4) as tc:
        # This body runs 4 times, each with a different tc!
        await tc.call_tool("navigate", {"url": "..."})
        tc.reward = evaluate()

    # tc.results contains all parallel traces
    # tc.reward is the mean reward
    print(f"Mean reward: {tc.reward}")
    for trace in tc.results:
        print(f"  {trace.trace_id}: {trace.reward}")
    ```
"""

from hud.trace.context import TraceContext, get_current_trace_headers
from hud.trace.mixin import TraceMixin

__all__ = [
    "TraceContext",
    "TraceMixin",
    "get_current_trace_headers",
]
