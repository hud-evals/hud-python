"""HUD Telemetry - Lightweight telemetry for HUD SDK.

This module provides:
- @instrument decorator for recording function calls
- High-performance span export to HUD API

Usage:
    import hud

    @hud.instrument
    async def my_function():
        ...

    # Within an eval context, calls are recorded
    async with hud.eval(task) as ctx:
        result = await my_function()
"""

from hud.telemetry.exporter import flush, queue_span, shutdown
from hud.telemetry.instrument import instrument

__all__ = [
    "flush",
    "instrument",
    "queue_span",
    "shutdown",
]
