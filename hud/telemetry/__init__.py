"""HUD Telemetry - Lightweight telemetry for HUD SDK.

This module provides the @instrument decorator for recording function calls.

Usage:
    import hud

    @hud.instrument
    async def my_function():
        ...

    # Within an eval context, calls are recorded
    async with hud.eval(task) as ctx:
        result = await my_function()
"""

from hud.telemetry.instrument import instrument

__all__ = [
    "instrument",
]
