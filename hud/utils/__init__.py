from __future__ import annotations

from .deprecation import deprecated, emit_deprecation_warning
from .telemetry import stream

__all__ = [
    "deprecated",
    "emit_deprecation_warning",
    "stream",
]
