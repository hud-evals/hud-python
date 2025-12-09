"""HUD Telemetry - Instrumentation for agent execution.

This module provides:
- instrument: Function instrumentation decorator

For other APIs, import directly from submodules:
- hud.telemetry.job: Job, job, create_job, get_current_job
- hud.telemetry.trace: Trace, trace
- hud.telemetry.async_context: async_job, async_trace
- hud.telemetry.replay: clear_trace, get_trace

Recommended: Use hud.eval() or env.eval() instead.
"""

from __future__ import annotations

from .instrument import instrument

__all__ = ["instrument"]
