from __future__ import annotations

import logging
from typing import Any

from opentelemetry import baggage
from opentelemetry.sdk.trace import ReadableSpan, Span, SpanProcessor

from .context import increment_step_count

logger = logging.getLogger(__name__)

# Lifecycle tools that should not increment step count
LIFECYCLE_TOOLS = {"setup", "evaluate"}


class HudEnrichmentProcessor(SpanProcessor):
    """Span processor that enriches every span with HUD-specific context.

    • Adds ``hud.task_run_id`` attribute if available.
    • Adds ``hud.job_id`` attribute if available in baggage.
    • Adds ``hud.step_count`` attribute if available in baggage.
    """

    def __init__(self) -> None:
        # No state, everything comes from context vars
        super().__init__()

    # --- callback hooks -------------------------------------------------
    def on_start(self, span: Span, parent_context: Any) -> None:  # type: ignore[override]
        try:
            # Get task_run_id from baggage in parent context
            run_id = baggage.get_baggage("hud.task_run_id", context=parent_context)
            if run_id and span.is_recording():
                span.set_attribute("hud.task_run_id", str(run_id))

            # Get job_id from baggage if available
            job_id = baggage.get_baggage("hud.job_id", context=parent_context)
            if job_id and span.is_recording():
                span.set_attribute("hud.job_id", str(job_id))

            # Check if this is an MCP tool call that should increment step count
            if span.is_recording() and self._should_increment_step(span):
                # Increment step count and update span
                new_count = increment_step_count()
                span.set_attribute("hud.step_count", new_count)
                logger.debug("Incremented step count to %d for tool call", new_count)
            else:
                # Get current step_count from baggage if available
                step_count = baggage.get_baggage("hud.step_count", context=parent_context)
                if step_count and isinstance(step_count, str) and span.is_recording():
                    try:
                        span.set_attribute("hud.step_count", int(step_count))
                    except ValueError:
                        logger.warning("Error setting step count: %s", step_count)

        except Exception as exc:  # defensive; never fail the tracer
            logger.debug("HudEnrichmentProcessor.on_start error: %s", exc, exc_info=False)

    def _should_increment_step(self, span: Span) -> bool:
        """Determine if this span represents a tool call that should increment step count."""
        # Check span attributes
        attrs = span.attributes or {}

        # Look for MCP tool call indicators
        # Option 1: Direct category attribute
        if attrs.get("category") == "mcp":
            # Check for tool name in various places
            tool_name = None

            # Try method name - check multiple possible locations
            method_name = (
                attrs.get("method_name")  # Direct attribute
                or attrs.get("mcp.method.name")
                or attrs.get("semconv_ai.mcp.method_name")
            )
            if method_name == "tools/call":
                # For tools/call, the actual tool name might be in the request
                request = attrs.get("request")
                if request and isinstance(request, dict):
                    tool_name = request.get("name")

            # Check if it's a lifecycle tool
            if tool_name and tool_name not in LIFECYCLE_TOOLS:
                return True

        # Option 2: Check span name pattern (e.g., "tool_name.mcp")
        span_name = span.name
        if span_name and span_name.endswith(".mcp"):
            tool_name = span_name[:-4]  # Remove .mcp suffix
            if tool_name not in LIFECYCLE_TOOLS and (
                attrs.get("mcp.method.name") == "tools/call"
                or attrs.get("semconv_ai.mcp.method_name") == "tools/call"
            ):
                return True

        return False

    def on_end(self, span: ReadableSpan) -> None:
        # Nothing to do enrichment is on_start only
        pass

    # Required to fully implement abstract base, but we don't batch spans
    def shutdown(self) -> None:  # type: ignore[override]
        pass

    def force_flush(self, timeout_millis: int | None = None) -> bool:  # type: ignore[override]
        return True
