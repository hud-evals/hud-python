"""Deprecated v5 eval context aliases."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from hud.telemetry.context import get_current_trace_id, set_trace_context


@dataclass(slots=True)
class EvalContext:
    """Minimal v5 compatibility context for legacy imports."""

    trace_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


__all__ = ["EvalContext", "get_current_trace_id", "set_trace_context"]
