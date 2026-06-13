"""HUD Telemetry - Lightweight telemetry for HUD SDK.

This module provides:
- The OTel-shaped ``Span`` wire format (:mod:`hud.telemetry.span`)
- ``@instrument`` debug spans for any function
- High-performance batched span export to the HUD platform

Usage:
    import hud

    @hud.instrument
    async def my_function():
        ...
"""

from __future__ import annotations

from hud.telemetry.exporter import flush, queue_span
from hud.telemetry.instrument import instrument

__all__ = [
    "flush",
    "instrument",
    "queue_span",
]
