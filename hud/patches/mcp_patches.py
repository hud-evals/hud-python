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
                # Use configured timeout, minimum 30s to prevent instant failures
                timeout = max(settings.client_timeout, 15.0)
                deadline = time.monotonic() + timeout
                retryable_exceptions = (
                    httpx.ConnectError,
                    httpx.ReadError,
                    httpx.TimeoutException,
                    ssl.SSLError,
                )
                retryable_status_codes = (502, 503, 504)

                def is_retryable(exc: Exception) -> bool:
                    if isinstance(exc, retryable_exceptions):
                        return True
                    if isinstance(exc, httpx.HTTPStatusError):
                        return exc.response.status_code in retryable_status_codes
                    return False

                async def send_error_response(exc: Exception) -> None:
                    """Send an error response to the client."""
                    if isinstance(msg.root, JSONRPCRequest):
                        error_response = JSONRPCError(
                            jsonrpc="2.0",
                            id=msg.root.id,
                            error=ErrorData(
                                code=-32000,
                                message=f"Transport error: {type(exc).__name__}",
                                data={"error_type": type(exc).__name__, "detail": str(exc)},
                            ),
                        )
                        await ctx.read_stream_writer.send(
                            SessionMessage(JSONRPCMessage(error_response))
                        )
                    else:
                        await ctx.read_stream_writer.send(exc)

                while True:
                    try:
                        if is_resumption:
                            await self._handle_resumption_request(ctx)
                        else:
                            await self._handle_post_request(ctx)
                        return
                    except asyncio.CancelledError:
                        raise
                    except Exception as e:
                        if is_retryable(e):
                            if time.monotonic() >= deadline:
                                logger.error("MCP request failed after timeout: %s", e)
                                await send_error_response(e)
                                return
                            logger.warning("Retrying MCP request after error: %s", e)
                            await asyncio.sleep(2.0)
                        else:
                            logger.exception("Request handler error: %s", e)
                            await send_error_response(e)
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


def patch_server_output_validation() -> None:
    """
    Patch MCP server to skip structured output validation and auto-generate
    structuredContent for FastMCP tools with x-fastmcp-wrap-result.
    """
    try:
        import json

        import mcp.types as types
        from mcp.server.lowlevel.server import Server

        def patched_call_tool(
            self: Any, validate_input: bool = True, validate_output: bool = False
        ) -> Any:
            """Patched call_tool that skips output validation."""

            def decorator(func: Any) -> Any:
                async def handler(req: types.CallToolRequest) -> Any:
                    try:
                        tool_name = req.params.name
                        arguments = req.params.arguments or {}
                        tool = await self._get_cached_tool_definition(tool_name)

                        if validate_input and tool:
                            try:
                                import jsonschema

                                jsonschema.validate(instance=arguments, schema=tool.inputSchema)
                            except jsonschema.ValidationError as e:
                                return self._make_error_result(
                                    f"Input validation error: {e.message}"
                                )

                        results = await func(tool_name, arguments)

                        # output normalization
                        unstructured_content: list[Any]
                        maybe_structured_content: dict[str, Any] | None
                        if isinstance(results, types.CallToolResult):
                            return types.ServerResult(results)
                        elif isinstance(results, tuple) and len(results) == 2:
                            unstructured_content, maybe_structured_content = results
                        elif isinstance(results, dict):
                            maybe_structured_content = results
                            text = json.dumps(results, indent=2)
                            unstructured_content = [types.TextContent(type="text", text=text)]
                        elif results is None:
                            # None means success with no content
                            unstructured_content = []
                            maybe_structured_content = None
                        elif isinstance(results, (str, bytes, bytearray, memoryview)):
                            # Handle string/bytes explicitly before iterable check
                            # (these are iterable but should not be split into chars/ints)
                            if isinstance(results, str):
                                text = results
                            elif isinstance(results, memoryview):
                                text = bytes(results).decode("utf-8", errors="replace")
                            else:
                                text = bytes(results).decode("utf-8", errors="replace")
                            unstructured_content = [types.TextContent(type="text", text=text)]
                            maybe_structured_content = None
                        elif isinstance(results, (int, float, bool)):
                            # Primitives -> string representation
                            unstructured_content = [
                                types.TextContent(type="text", text=str(results))
                            ]
                            maybe_structured_content = None
                        elif hasattr(results, "__iter__"):
                            unstructured_content = list(results)
                            maybe_structured_content = None
                        else:
                            return self._make_error_result(
                                f"Unexpected return type: {type(results).__name__}"
                            )

                        # Auto-generate structuredContent for FastMCP tools
                        # FastMCP generates outputSchema but doesn't populate it
                        if maybe_structured_content is None and tool:
                            output_schema = getattr(tool, "outputSchema", None)
                            if output_schema and output_schema.get("x-fastmcp-wrap-result"):
                                for item in unstructured_content:
                                    if isinstance(item, types.TextContent):
                                        try:
                                            parsed = json.loads(item.text)
                                            maybe_structured_content = {"result": parsed}
                                        except json.JSONDecodeError:
                                            maybe_structured_content = {"result": item.text}
                                        break

                        return types.ServerResult(
                            types.CallToolResult(
                                content=list(unstructured_content),
                                structuredContent=maybe_structured_content,
                                isError=False,
                            )
                        )
                    except Exception as e:
                        return self._make_error_result(str(e))

                self.request_handlers[types.CallToolRequest] = handler
                return func

            return decorator

        Server.call_tool = patched_call_tool
        logger.debug("Patched Server.call_tool to skip output validation")

    except ImportError:
        logger.debug("mcp.server.lowlevel.server not available, skipping patch")
    except Exception as e:
        logger.warning("Failed to patch server output validation: %s", e)


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
    patch_server_output_validation()
    suppress_fastmcp_logging()
    logger.debug("All MCP patches applied")
