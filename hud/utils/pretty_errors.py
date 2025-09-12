from __future__ import annotations

import asyncio
import logging
import sys
from typing import Any

from hud.utils.hud_console import hud_console

logger = logging.getLogger(__name__)


def _render_and_fallback(exc_type: type[BaseException], value: BaseException, tb: Any) -> None:
    """Render exceptions via HUD design, then delegate to default excepthook.

    Only formats for HudException family or when running in a TTY; otherwise,
    defers to the default handler to avoid swallowing useful tracebacks in code.
    """
    # First, print the full traceback
    sys.__excepthook__(exc_type, value, tb)

    # Then print our formatted error
    try:
        from hud.shared.exceptions import HudException  # lazy import

        if isinstance(value, HudException):
            # Flush stderr to ensure traceback is printed first
            sys.stderr.flush()
            # Add separator and render our formatted error
            hud_console.console.print("")
            hud_console.render_exception(value)
    except Exception:
        # If rendering fails for any reason, silently continue
        logger.warning("Failed to render exception: %s, %s, %s", exc_type, value, tb)


def _async_exception_handler(loop: asyncio.AbstractEventLoop, context: dict[str, Any]) -> None:
    exc = context.get("exception")
    msg = context.get("message")
    try:
        if exc is not None:
            hud_console.render_exception(exc)
        elif msg:
            hud_console.error(msg)
            hud_console.render_support_hint()
    except Exception:
        logger.warning("Failed to render exception: %s, %s, %s", exc, msg, context)

    # Delegate to default handler
    loop.default_exception_handler(context)


def install_pretty_errors() -> None:
    """Install global pretty error handlers for sync and async exceptions."""
    sys.excepthook = _render_and_fallback
    try:
        # Try to get the running loop first
        loop = asyncio.get_running_loop()
        loop.set_exception_handler(_async_exception_handler)
    except RuntimeError:
        # No running loop, try to create one
        try:
            loop = asyncio.new_event_loop()
            loop.set_exception_handler(_async_exception_handler)
        except Exception:
            logger.warning("No running loop, could not set exception handler")
    except Exception:
        logger.warning("No running loop, could not set exception handler")
