"""MCP instrumentation support for HUD.

This module provides functions to enable MCP OpenTelemetry instrumentation
for automatic tracing of MCP protocol communication.

Note: This module requires the [agents] extra to be installed:
    pip install hud-python[agents]
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Callable

    from opentelemetry.trace import TracerProvider

logger = logging.getLogger(__name__)

# Check if OpenTelemetry is available
_HAS_OPENTELEMETRY = False
try:
    from opentelemetry import trace as _otel_trace  # noqa: F401

    _HAS_OPENTELEMETRY = True
except ImportError:
    pass


def install_mcp_instrumentation(provider: TracerProvider | None = None) -> None:
    """Enable community MCP OpenTelemetry instrumentation if present.

    Args:
        provider: The TracerProvider to use for instrumentation
    """
    import logging

    logger = logging.getLogger(__name__)

    try:
        # First, patch the _instruments to use our fork
        import opentelemetry.instrumentation.mcp.instrumentation as mcp_inst

        mcp_inst._instruments = ("hud-mcp-python-sdk >= 3.13.1",)

        from opentelemetry.instrumentation.mcp.instrumentation import (
            McpInstrumentor,
        )

        # Then, patch the instrumentation to handle 3-value transports correctly
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

        # First, patch the get_error_type function to handle invalid HTTP status codes
        _patch_get_error_type()

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


def _patch_get_error_type() -> None:
    """Patch get_error_type to handle invalid HTTP status codes gracefully."""
    import re
    from http import HTTPStatus

    try:
        import opentelemetry.instrumentation.mcp.instrumentation as mcp_inst

        def patched_get_error_type(error_message: str) -> str | None:
            """Extract HTTP status from error message, handling invalid codes."""
            if not isinstance(error_message, str):
                return None
            match = re.search(r"\b(4\d{2}|5\d{2})\b", error_message)
            if match:
                num = int(match.group())
                try:
                    # Only return if it's a valid HTTPStatus
                    if 400 <= num <= 599:
                        return HTTPStatus(num).name
                except ValueError:
                    # Not a valid HTTP status code
                    logger.debug("Ignoring invalid HTTP status code: %s", num)
            return None

        # Apply the patch
        mcp_inst.get_error_type = patched_get_error_type
        logger.debug("Patched get_error_type to handle invalid HTTP status codes")

    except Exception as e:
        logger.warning("Failed to patch get_error_type: %s", e)
