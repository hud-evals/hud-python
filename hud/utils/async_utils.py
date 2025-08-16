"""Async utilities for HUD SDK.

This module provides utilities for running async code in various environments,
including Jupyter notebooks and synchronous contexts.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Coroutine

logger = logging.getLogger(__name__)


def fire_and_forget(coro: Coroutine[Any, Any, Any], description: str = "task") -> None:
    """Execute a coroutine in a fire-and-forget manner.

    This function handles running async code in various contexts:
    - When an event loop is already running (normal async context)
    - When no event loop exists (sync context, some Jupyter setups)
    - Gracefully handles interpreter shutdown

    Args:
        coro: The coroutine to execute
        description: Description of the task for logging (e.g., "update job status")

    Example:
        fire_and_forget(
            some_async_function(),
            description="update status"
        )
    """
    try:
        # Try to get current event loop
        loop = asyncio.get_running_loop()
        # Schedule the coroutine
        task = loop.create_task(coro)
        # Add error handler to prevent unhandled exceptions
        task.add_done_callback(lambda t: t.exception() if not t.cancelled() else None)
    except RuntimeError:
        # No running event loop (e.g., Jupyter without %autoawait, sync context)
        try:
            # Try to run in a thread as a fallback
            def run_in_thread() -> None:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    loop.run_until_complete(coro)
                except Exception as e:
                    # Suppress warnings about interpreter shutdown
                    if "interpreter shutdown" not in str(e):
                        logger.debug("Error in threaded %s: %s", description, e)

            thread = threading.Thread(target=run_in_thread, daemon=True)
            thread.start()
        except Exception as e:
            # If that fails too, just log and continue
            # Special case: suppress "cannot schedule new futures after interpreter shutdown"
            if "interpreter shutdown" not in str(e):
                logger.debug("Could not %s - no event loop available: %s", description, e)
