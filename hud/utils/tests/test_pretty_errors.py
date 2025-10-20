from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

from hud.utils.pretty_errors import (
    _async_exception_handler,
    _render_and_fallback,
    install_pretty_errors,
)


def test_render_and_fallback_hud_exception():
    """Test _render_and_fallback with HudException."""
    from hud.shared.exceptions import HudException

    exc = HudException("Test error")

    with (
        patch("sys.__excepthook__") as mock_excepthook,
        patch("hud.utils.pretty_errors.hud_console") as mock_console,
        patch("sys.stderr.flush"),
    ):
        _render_and_fallback(HudException, exc, None)

        mock_excepthook.assert_called_once()
        mock_console.render_exception.assert_called_once_with(exc)


def test_render_and_fallback_non_hud_exception():
    """Test _render_and_fallback with non-HudException."""
    exc = ValueError("Test error")

    with (
        patch("sys.__excepthook__") as mock_excepthook,
        patch("hud.utils.pretty_errors.hud_console") as mock_console,
    ):
        _render_and_fallback(ValueError, exc, None)

        mock_excepthook.assert_called_once()
        # Should not render for non-HudException
        mock_console.render_exception.assert_not_called()


def test_render_and_fallback_rendering_error():
    """Test _render_and_fallback handles rendering errors gracefully."""
    from hud.shared.exceptions import HudException

    exc = HudException("Test error")

    with (
        patch("sys.__excepthook__") as mock_excepthook,
        patch("hud.utils.pretty_errors.hud_console") as mock_console,
    ):
        mock_console.render_exception.side_effect = Exception("Render failed")

        # Should not raise
        _render_and_fallback(HudException, exc, None)

        mock_excepthook.assert_called_once()


def test_async_exception_handler_with_exception():
    """Test _async_exception_handler with exception in context."""
    mock_loop = MagicMock()
    context = {"exception": ValueError("Test error")}

    with patch("hud.utils.pretty_errors.hud_console") as mock_console:
        _async_exception_handler(mock_loop, context)

        mock_console.render_exception.assert_called_once()
        mock_loop.default_exception_handler.assert_called_once_with(context)


def test_async_exception_handler_with_message():
    """Test _async_exception_handler with message only."""
    mock_loop = MagicMock()
    context = {"message": "Error message"}

    with patch("hud.utils.pretty_errors.hud_console") as mock_console:
        _async_exception_handler(mock_loop, context)

        mock_console.error.assert_called_once_with("Error message")
        mock_console.render_support_hint.assert_called_once()
        mock_loop.default_exception_handler.assert_called_once()


def test_async_exception_handler_rendering_error():
    """Test _async_exception_handler handles rendering errors."""
    mock_loop = MagicMock()
    context = {"exception": ValueError("Test")}

    with patch("hud.utils.pretty_errors.hud_console") as mock_console:
        mock_console.render_exception.side_effect = Exception("Render failed")

        # Should not raise, should call default handler
        _async_exception_handler(mock_loop, context)

        mock_loop.default_exception_handler.assert_called_once()


def test_install_pretty_errors_with_running_loop():
    """Test install_pretty_errors with a running event loop."""
    mock_loop = MagicMock()

    with patch("asyncio.get_running_loop", return_value=mock_loop):
        install_pretty_errors()

        assert sys.excepthook == _render_and_fallback
        mock_loop.set_exception_handler.assert_called_once_with(_async_exception_handler)


def test_install_pretty_errors_no_running_loop():
    """Test install_pretty_errors without a running loop."""
    with (
        patch("asyncio.get_running_loop", side_effect=RuntimeError("No running loop")),
        patch("asyncio.new_event_loop") as mock_new_loop,
    ):
        mock_loop = MagicMock()
        mock_new_loop.return_value = mock_loop

        install_pretty_errors()

        assert sys.excepthook == _render_and_fallback
        mock_loop.set_exception_handler.assert_called_once()


def test_install_pretty_errors_new_loop_fails():
    """Test install_pretty_errors when creating new loop fails."""
    with (
        patch("asyncio.get_running_loop", side_effect=RuntimeError("No running loop")),
        patch("asyncio.new_event_loop", side_effect=Exception("Can't create loop")),
    ):
        # Should not raise
        install_pretty_errors()

        assert sys.excepthook == _render_and_fallback


def test_install_pretty_errors_set_handler_fails():
    """Test install_pretty_errors when set_exception_handler fails."""
    mock_loop = MagicMock()
    mock_loop.set_exception_handler.side_effect = Exception("Can't set handler")

    with patch("asyncio.get_running_loop", return_value=mock_loop):
        # Should not raise
        install_pretty_errors()

        assert sys.excepthook == _render_and_fallback


def test_async_exception_handler_no_exception_or_message():
    """Test _async_exception_handler with empty context."""
    mock_loop = MagicMock()
    context = {}

    with patch("hud.utils.pretty_errors.hud_console") as mock_console:
        _async_exception_handler(mock_loop, context)

        mock_console.render_exception.assert_not_called()
        mock_console.error.assert_not_called()
        mock_loop.default_exception_handler.assert_called_once()


def test_render_and_fallback_with_traceback():
    """Test _render_and_fallback includes traceback."""
    from hud.shared.exceptions import HudException

    exc = HudException("Test error")

    # Create a fake traceback
    try:
        raise exc
    except HudException as e:
        tb = e.__traceback__

    with (
        patch("sys.__excepthook__") as mock_excepthook,
        patch("hud.utils.pretty_errors.hud_console"),
        patch("sys.stderr.flush"),
    ):
        _render_and_fallback(HudException, exc, tb)

        # Should call excepthook with traceback
        call_args = mock_excepthook.call_args[0]
        assert call_args[2] == tb
