"""Telemetry utility functions for managing trace and span lifecycle."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


async def flush_telemetry() -> None:
    """Flush OpenTelemetry span processor to export buffered spans immediately.

    Called automatically by async_trace (standalone) and async_job on exit.

    Example:
        >>> # Custom evaluation loop
        >>> for task in tasks:
        ...     async with hud.async_trace(task.name):
        ...         await process(task)
        >>> # Spans already flushed by each async_trace
    """
    from hud.otel.config import is_telemetry_configured
    from hud.utils import hud_console

    logger.debug("Flushing telemetry spans...")
    if not is_telemetry_configured():
        return

    try:
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider

        provider = trace.get_tracer_provider()
        if isinstance(provider, TracerProvider):
            success = provider.force_flush(timeout_millis=5000)
            if success:
                hud_console.info("✓ Telemetry uploaded successfully")
                logger.debug("OpenTelemetry spans flushed successfully")
            else:
                logger.debug("OpenTelemetry flush timed out (will export on exit)")
    except Exception as e:
        logger.debug("Failed to flush OpenTelemetry: %s", e)
