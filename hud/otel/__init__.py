"""HUD OpenTelemetry integration.

.. deprecated::
    The `hud.otel` module is deprecated and will be removed in a future version.
    Use `env.trace()` from `hud.environment.Environment` instead.

    This module requires the [agents] extra:
        pip install hud-python[agents]

This package provides the internal OpenTelemetry implementation for HUD telemetry.

Internal Components:
- config: OpenTelemetry configuration and setup
- context: Trace context management and utilities
- processors: Span enrichment with HUD context
- exporters: Sending spans to HUD backend
- collector: In-memory span collection for replay
- instrumentation: Auto-instrumentation for agents and MCP
"""

from __future__ import annotations

import warnings

# Show deprecation warning when module is imported
warnings.warn(
    "The hud.otel module is deprecated. Use env.trace() instead. "
    "This module requires pip install hud-python[agents].",
    DeprecationWarning,
    stacklevel=2,
)

from .collector import enable_trace_collection
from .config import configure_telemetry, is_telemetry_configured, shutdown_telemetry
from .context import (
    get_current_task_run_id,
    is_root_trace,
    span_context,
    trace,
)

__all__ = [
    "configure_telemetry",
    "enable_trace_collection",
    "get_current_task_run_id",
    "is_root_trace",
    "is_telemetry_configured",
    "shutdown_telemetry",
    "span_context",
    "trace",
]
