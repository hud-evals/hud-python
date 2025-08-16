"""MCP instrumentation support for HUD.

This module provides functions to enable MCP OpenTelemetry instrumentation
for automatic tracing of MCP protocol communication.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Callable

    from opentelemetry.trace import TracerProvider

logger = logging.getLogger(__name__)

LIFECYCLE_TOOLS = {"setup", "evaluate"}


def install_mcp_instrumentation(provider: TracerProvider) -> None:
    """Enable community MCP OpenTelemetry instrumentation if present.

    Args:
        provider: The TracerProvider to use for instrumentation
    """
    import logging

    logger = logging.getLogger(__name__)

    try:
        from opentelemetry.instrumentation.mcp.instrumentation import (
            McpInstrumentor,
        )

        # First, patch the instrumentation to handle 3-value transports correctly
        _patch_mcp_instrumentation()

        McpInstrumentor().instrument(tracer_provider=provider)
        logger.debug("MCP instrumentation installed with fastmcp compatibility patch")
    except ImportError:
        logger.debug("opentelemetry-instrumentation-mcp not available, skipping")
    except Exception as exc:
        logger.warning("Failed to install MCP instrumentation: %s", exc)


def _patch_mcp_instrumentation() -> None:
    """Patch MCP instrumentation to handle 3-value transport yields correctly."""
    from contextlib import asynccontextmanager

    try:
        from opentelemetry.instrumentation.mcp.instrumentation import McpInstrumentor

        def patched_transport_wrapper(self: Any, tracer: Any) -> Callable[..., Any]:
            @asynccontextmanager
            async def traced_method(
                wrapped: Callable[..., Any], instance: Any, args: Any, kwargs: Any
            ) -> AsyncGenerator[Any, None]:
                async with wrapped(*args, **kwargs) as result:
                    # Check if we got a tuple with 3 values
                    if isinstance(result, tuple) and len(result) == 3:
                        read_stream, write_stream, third_value = result
                        # Import here to avoid circular imports
                        from opentelemetry.instrumentation.mcp.instrumentation import (
                            InstrumentedStreamReader,
                            InstrumentedStreamWriter,
                        )

                        yield (
                            InstrumentedStreamReader(read_stream, tracer),
                            InstrumentedStreamWriter(write_stream, tracer),
                            third_value,
                        )
                    else:
                        # Fall back to 2-value case
                        read_stream, write_stream = result
                        from opentelemetry.instrumentation.mcp.instrumentation import (
                            InstrumentedStreamReader,
                            InstrumentedStreamWriter,
                        )

                        yield (
                            InstrumentedStreamReader(read_stream, tracer),
                            InstrumentedStreamWriter(write_stream, tracer),
                        )

            return traced_method

        # Apply the patch
        McpInstrumentor._transport_wrapper = patched_transport_wrapper

    except Exception as e:
        import logging

        logger = logging.getLogger(__name__)
        logger.warning("Failed to patch MCP instrumentation: %s", e)
