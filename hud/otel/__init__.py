"""HUD OpenTelemetry integration.

This package provides the internal OpenTelemetry implementation for HUD telemetry.
Users should interact with the telemetry APIs through hud.telemetry instead.

Internal Components:
- config: OpenTelemetry configuration and setup
- context: Trace context management and utilities
- processors: Span enrichment with HUD context
- exporters: Sending spans to HUD backend
- collector: In-memory span collection for replay
- instrumentation: Auto-instrumentation for agents and MCP
"""

from __future__ import annotations

from .collector import enable_trace_collection
from .config import configure_telemetry, shutdown_telemetry
from .context import get_current_task_run_id, is_root_trace, span_context, trace

__all__ = [
    # Configuration
    "configure_telemetry",
    # Collection
    "enable_trace_collection",
    "get_current_task_run_id",
    "is_root_trace",
    "shutdown_telemetry",
    "span_context",
    # Context management
    "trace",
]
