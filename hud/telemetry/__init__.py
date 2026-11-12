"""HUD telemetry: OTel-shaped spans, ``@instrument`` debug spans, and batched export."""

from __future__ import annotations

from hud.telemetry.exporter import flush
from hud.telemetry.instrument import instrument

__all__ = [
    "flush",
    "instrument",
]
