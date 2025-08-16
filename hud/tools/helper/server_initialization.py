"""Helper for MCP server initialization with progress notifications.

Example:
    ```python
    from hud.tools.helper import mcp_intialize_wrapper


    @mcp_intialize_wrapper
    async def initialize_environment(session=None, progress_token=None):
        # Send progress if available
        if session and progress_token:
            await session.send_progress_notification(
                progress_token=progress_token, progress=0, total=100, message="Starting services..."
            )

        # Your initialization code works with or without session
        start_services()


    # Create and run server - initialization happens automatically
    mcp = FastMCP("My Server")
    mcp.run()
    ```
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import mcp.types as types
from mcp.server.session import ServerSession

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from mcp.shared.session import RequestResponder

# Store the original _received_request method
_original_received_request = ServerSession._received_request
_init_function: Callable | None = None
_initialized = False


async def _patched_received_request(
    self: ServerSession, responder: RequestResponder[types.ClientRequest, types.ServerResult]
) -> types.ServerResult | None:
    """Intercept initialization to run custom setup with progress notifications."""
    global _initialized, _init_function

    # Check if this is an initialization request
    if isinstance(responder.request.root, types.InitializeRequest):
        params = responder.request.root.params
        # Extract progress token if present
        progress_token = None
        if hasattr(params, "meta") and params.meta and hasattr(params.meta, "progressToken"):
            progress_token = params.meta.progressToken

        # Run our initialization function if provided and not already done
        if _init_function and not _initialized:
            try:
                await _init_function(session=self, progress_token=progress_token)
            except Exception as e:
                if progress_token:
                    await self.send_progress_notification(
                        progress_token=progress_token,
                        progress=0,
                        total=100,
                        message=f"Initialization failed: {e!s}",
                    )
                raise

        # Call the original handler to send the InitializeResult
        result = await _original_received_request(self, responder)
        _initialized = True

        # Restore the original method AFTER initialization is complete
        ServerSession._received_request = _original_received_request

        return result

    # For non-initialization requests, use the original handler
    return await _original_received_request(self, responder)


def mcp_intialize_wrapper(
    init_function: Callable[[ServerSession | None, str | None], Awaitable[None]] | None = None,
) -> Callable:
    """Decorator to enable progress notifications during MCP server initialization.

    Your init function receives optional session and progress_token parameters.
    If provided, use them to send progress updates. If not, the function still works.

    Usage:
        @mcp_intialize_wrapper
        async def initialize(session=None, progress_token=None):
            if session and progress_token:
                await session.send_progress_notification(...)
            # Your init code here

    Must be applied before creating FastMCP instance or calling mcp.run().
    """
    global _init_function

    def decorator(func: Callable[[ServerSession | None, str | None], Awaitable[None]]) -> Callable:
        global _init_function
        # Store the initialization function
        _init_function = func

        # Apply the monkey patch if not already applied
        if ServerSession._received_request != _patched_received_request:
            ServerSession._received_request = _patched_received_request  # type: ignore[assignment]

        return func

    # If called with a function directly
    if init_function is not None:
        return decorator(init_function)

    # If used as @decorator
    return decorator
