from __future__ import annotations

import logging
import time
from typing import Any

from opentelemetry import baggage
from opentelemetry.sdk.trace import ReadableSpan, Span, SpanProcessor

from .context import (
    get_agent_steps,
    get_base_mcp_steps,
    get_mcp_tool_steps,
    increment_agent_steps,
    increment_base_mcp_steps,
    increment_mcp_tool_steps,
)

logger = logging.getLogger(__name__)


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

            # Check what type of step this is and increment appropriate counters
            if span.is_recording():
                step_type = self._get_step_type(span)

                if step_type == "agent":
                    # Increment agent steps
                    new_agent_count = increment_agent_steps()
                    span.set_attribute("hud.agent_steps", new_agent_count)
                    logger.debug("Incremented agent steps to %d", new_agent_count)

                elif step_type == "base_mcp":
                    # Increment base MCP steps
                    new_base_count = increment_base_mcp_steps()
                    span.set_attribute("hud.base_mcp_steps", new_base_count)
                    logger.debug("Incremented base MCP steps to %d", new_base_count)

                elif step_type == "mcp_tool":
                    # Increment both base MCP and MCP tool steps
                    new_base_count = increment_base_mcp_steps()
                    new_tool_count = increment_mcp_tool_steps()
                    span.set_attribute("hud.base_mcp_steps", new_base_count)
                    span.set_attribute("hud.mcp_tool_steps", new_tool_count)
                    logger.debug(
                        "Incremented MCP steps to base=%d, tool=%d", new_base_count, new_tool_count
                    )

                # Always set all current step counts on the span
                span.set_attribute("hud.base_mcp_steps", get_base_mcp_steps())
                span.set_attribute("hud.mcp_tool_steps", get_mcp_tool_steps())
                span.set_attribute("hud.agent_steps", get_agent_steps())

        except Exception as exc:  # defensive; never fail the tracer
            logger.debug("HudEnrichmentProcessor.on_start error: %s", exc, exc_info=False)

    def _get_step_type(self, span: Span) -> str | None:
        """Determine what type of step this span represents.

        Returns:
            'base_mcp' for any MCP span
            'mcp_tool' for MCP tool calls (tools/call.mcp)
            'agent' for agent spans
            None if not a step
        """
        # Check span attributes
        attrs = span.attributes or {}
        span_name = span.name

        # Check for agent steps (instrumented with span_type="agent")
        if attrs.get("category") == "agent":
            return "agent"

        # Check span name pattern for MCP calls
        if span_name:
            # tools/call.mcp is an mcp_tool step
            if span_name == "tools/call.mcp":
                return "mcp_tool"

            # Any other .mcp suffixed span is a base MCP step
            elif span_name.endswith(".mcp"):
                return "base_mcp"

        return None

    def on_end(self, span: ReadableSpan) -> None:
        # Nothing to do enrichment is on_start only
        pass

    # Required to fully implement abstract base, but we don't batch spans
    def shutdown(self) -> None:  # type: ignore[override]
        pass

    def force_flush(self, timeout_millis: int | None = None) -> bool:  # type: ignore[override]
        # No-op and return immediately. We never want a SpanProcessor to block
        # shutdown/flush in worker processes. BatchSpanProcessor handles flushing
        # its own queue and exporter.
        return True
