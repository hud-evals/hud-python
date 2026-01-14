"""
Runtime patches for the standard mcp package.

These patches apply fixes from the HUD fork without requiring a separate package.
Import this module early (e.g., in hud/__init__.py) to apply patches.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def patch_streamable_http_error_handling() -> None:
    """
    Patch StreamableHTTPTransport.post_writer to handle request errors properly.

    The original implementation doesn't catch errors in handle_request_async,
    which can cause the client to hang indefinitely. This patch wraps the handler
    to send a proper JSONRPCError response when transport errors occur (e.g.,
    ReadTimeout), allowing the waiting caller to receive the error and fail
    gracefully instead of hanging.
    """
    try:
        from mcp.client.streamable_http import StreamableHTTPTransport

        async def patched_post_writer(
            self: Any,
            client: Any,
            write_stream_reader: Any,
            read_stream_writer: Any,
            write_stream: Any,
            start_get_stream: Any,
            tg: Any,
        ) -> None:
            import asyncio
            import ssl
            import time

            import httpx
            from mcp.client.streamable_http import RequestContext
            from mcp.shared.message import ClientMessageMetadata, SessionMessage
            from mcp.types import ErrorData, JSONRPCError, JSONRPCMessage, JSONRPCRequest

            from hud.settings import settings

            async def handle_request_async(ctx: RequestContext, is_resumption: bool) -> None:
                msg = ctx.session_message.message
                deadline = (
                    time.monotonic() + settings.client_timeout
                    if settings.client_timeout > 0
                    else None
                )
                retryable = (
                    httpx.ConnectError,
                    httpx.ReadError,
                    httpx.TimeoutException,
                    ssl.SSLError,
                )

                while True:
                    try:
                        if is_resumption:
                            await self._handle_resumption_request(ctx)
                        else:
                            await self._handle_post_request(ctx)
                        return
                    except retryable as e:
                        if deadline and time.monotonic() >= deadline:
                            raise
                        logger.warning("Retrying MCP request after error: %s", e)
                        await asyncio.sleep(2.0)
                    except asyncio.CancelledError:
                        raise
                    except Exception as e:
                        logger.exception("Request handler error: %s", e)
                        if isinstance(msg.root, JSONRPCRequest):
                            error_response = JSONRPCError(
                                jsonrpc="2.0",
                                id=msg.root.id,
                                error=ErrorData(
                                    code=-32000,
                                    message=f"Transport error: {type(e).__name__}",
                                    data={"error_type": type(e).__name__, "detail": str(e)},
                                ),
                            )
                            await ctx.read_stream_writer.send(
                                SessionMessage(JSONRPCMessage(error_response))
                            )
                        else:
                            await ctx.read_stream_writer.send(e)
                        return

            try:
                async with write_stream_reader:
                    async for session_message in write_stream_reader:
                        message = session_message.message
                        metadata = (
                            session_message.metadata
                            if isinstance(session_message.metadata, ClientMessageMetadata)
                            else None
                        )
                        is_resumption = bool(metadata and metadata.resumption_token)

                        logger.debug("Sending client message: %s", message)

                        if self._is_initialized_notification(message):
                            start_get_stream()

                        ctx = RequestContext(
                            client=client,
                            headers=self.request_headers,
                            session_id=self.session_id,
                            session_message=session_message,
                            metadata=metadata,
                            read_stream_writer=read_stream_writer,
                            sse_read_timeout=self.sse_read_timeout,
                        )

                        if isinstance(message.root, JSONRPCRequest):
                            tg.start_soon(handle_request_async, ctx, is_resumption)
                        else:
                            await handle_request_async(ctx, is_resumption)

            except Exception:
                logger.exception("Error in post_writer")
            finally:
                await read_stream_writer.aclose()
                await write_stream.aclose()

        StreamableHTTPTransport.post_writer = patched_post_writer
        logger.debug("Patched StreamableHTTPTransport.post_writer")

    except ImportError:
        logger.debug("mcp.client.streamable_http not available, skipping patch")
    except Exception as e:
        logger.warning("Failed to patch streamable_http: %s", e)


def patch_client_session_validation() -> None:
    """
    Patch ClientSession to skip structured output validation.

    The original validation is strict and raises errors for non-conforming
    but usable responses. We replace it with a no-op.
    """
    try:
        from mcp.client.session import ClientSession

        async def noop_validate(self: Any, name: str, result: Any) -> None:
            """Skip structured output validation entirely."""

        ClientSession._validate_tool_result = noop_validate
        logger.debug("Patched ClientSession._validate_tool_result to skip validation")

    except ImportError:
        logger.debug("mcp.client.session not available, skipping patch")
    except Exception as e:
        logger.warning("Failed to patch client session: %s", e)


def suppress_fastmcp_logging(level: int = logging.WARNING) -> None:
    """
    Suppress verbose fastmcp logging.

    FastMCP logs a lot of INFO-level messages that clutter output.
    This sets all fastmcp loggers to the specified level.

    Args:
        level: Logging level to set (default: WARNING)
    """
    loggers_to_suppress = [
        "fastmcp",
        "fastmcp.server.server",
        "fastmcp.server.openapi",
        "fastmcp.tools.tool_manager",
    ]
    for logger_name in loggers_to_suppress:
        logging.getLogger(logger_name).setLevel(level)
    logger.debug("Suppressed fastmcp logging to level %s", level)


def apply_all_patches() -> None:
    """Apply all MCP patches."""
    patch_streamable_http_error_handling()
    patch_client_session_validation()
    suppress_fastmcp_logging()
    logger.debug("All MCP patches applied")
