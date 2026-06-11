"""HUD Telemetry - Lightweight telemetry for HUD SDK.

This module provides:
- @instrument decorator for recording function calls
- High-performance span export to HUD API
- Off-loop trajectory recording for robot envs (EpisodeRecorder + TraceSink)

The LeRobot v3 dataset sink lives in :mod:`hud.telemetry.lerobot` (requires the
``lerobot`` extra).

Usage:
    import hud

    @hud.instrument
    async def my_function():
        ...

    # Within an eval context, calls are recorded
    async with hud.eval(task) as ctx:
        result = await my_function()
"""

from __future__ import annotations

from hud.telemetry.exporter import flush, queue_span
from hud.telemetry.instrument import instrument
from hud.telemetry.recorder import EpisodeRecorder, Frame, TraceSink

__all__ = [
    "EpisodeRecorder",
    "Frame",
    "TraceSink",
    "flush",
    "instrument",
    "queue_span",
]
