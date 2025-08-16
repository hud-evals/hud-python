"""Tests for the helper module."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from mcp.server.session import ServerSession

from hud.tools.helper import mcp_intialize_wrapper


class TestMCPInitializeWrapper:
    """Test the MCP initialize wrapper functionality."""

    @pytest.mark.asyncio
    async def test_basic_wrapper_functionality(self):
        """Test basic wrapper functionality without session."""
        init_called = False

        @mcp_intialize_wrapper
        async def initialize(session=None, progress_token=None):
            nonlocal init_called
            init_called = True
            assert session is None
            assert progress_token is None

        # Function should be decorated but not called yet
        assert not init_called
        assert initialize.__name__ == "initialize"

    @pytest.mark.asyncio
    async def test_wrapper_with_session(self):
        """Test wrapper with session and progress token."""
        init_values = {}

        @mcp_intialize_wrapper
        async def initialize(session=None, progress_token=None):
            init_values["session"] = session
            init_values["progress_token"] = progress_token

            if session and progress_token:
                await session.send_progress_notification(
                    progress_token=progress_token, progress=50, total=100, message="Testing"
                )

        # Simulate the patched _received_request being called
        mock_session = MagicMock(spec=ServerSession)
        mock_session.send_progress_notification = AsyncMock()

        # The wrapper should have patched ServerSession._received_request
        # Let's test that the initialization function gets called
        from hud.tools.helper.server_initialization import _init_function

        # Verify our function was stored
        assert _init_function is not None

        # Call the init function directly to test it
        await _init_function(session=mock_session, progress_token="test_token")

        # Verify it was called with correct values
        assert init_values["session"] is mock_session
        assert init_values["progress_token"] == "test_token"
        mock_session.send_progress_notification.assert_called_once()

    @pytest.mark.asyncio
    async def test_wrapper_with_exception(self):
        """Test wrapper handling exceptions during initialization."""

        @mcp_intialize_wrapper
        async def failing_initialize(session=None, progress_token=None):
            raise RuntimeError("Initialization failed!")

        mock_session = MagicMock(spec=ServerSession)
        mock_session.send_progress_notification = AsyncMock()

        from hud.tools.helper.server_initialization import (
            _patched_received_request,
        )

        # Mock the original _received_request to prevent calling it
        with patch("hud.tools.helper.server_initialization._original_received_request"):
            # Create a mock responder with initialization request
            mock_responder = MagicMock()

            # Check if it's an InitializeRequest
            from mcp import types

            # Create params with proper meta structure
            params = types.InitializeRequestParams(
                protocolVersion="1.0.0",
                capabilities=types.ClientCapabilities(),
                clientInfo=types.Implementation(name="test", version="1.0"),
            )
            # Add meta with progressToken as an attribute
            params.meta = MagicMock()
            params.meta.progressToken = "test_token"

            mock_responder.request.root = types.InitializeRequest(
                method="initialize", params=params
            )

            # Should raise the exception when calling the patched method
            with pytest.raises(RuntimeError, match="Initialization failed!"):
                await _patched_received_request(mock_session, mock_responder)

            # Should have sent error notification
            mock_session.send_progress_notification.assert_called_once()
            call_args = mock_session.send_progress_notification.call_args
            assert call_args.kwargs["progress_token"] == "test_token"
            assert "Initialization failed" in call_args.kwargs["message"]

    @pytest.mark.asyncio
    async def test_direct_function_call(self):
        """Test decorator called directly with a function."""

        async def my_init(session=None, progress_token=None):
            return None

        # Apply decorator directly
        decorated = mcp_intialize_wrapper(my_init)

        assert decorated is my_init  # Should return the same function

        from hud.tools.helper.server_initialization import _init_function

        assert _init_function is my_init

    @pytest.mark.asyncio
    async def test_monkey_patch_applied(self):
        """Test that the monkey patch is applied to ServerSession."""
        # Reset the module state
        import importlib

        import hud.tools.helper.server_initialization as init_module

        # Store original
        original_method = ServerSession._received_request

        # Reload module to reset state
        importlib.reload(init_module)

        try:
            # Apply decorator
            @mcp_intialize_wrapper
            async def test_init(session=None, progress_token=None):
                pass

            # Check that patch was applied
            assert ServerSession._received_request != original_method
        finally:
            # Restore original to not affect other tests
            ServerSession._received_request = original_method

    @pytest.mark.asyncio
    async def test_import_from_package(self):
        """Test that the wrapper can be imported from the package."""
        from hud.tools.helper import mcp_intialize_wrapper as wrapper

        assert wrapper is mcp_intialize_wrapper
        assert callable(wrapper)

    @pytest.mark.asyncio
    async def test_wrapper_without_progress_token(self):
        """Test wrapper when no progress token is provided."""
        init_called = False

        @mcp_intialize_wrapper
        async def initialize(session=None, progress_token=None):
            nonlocal init_called
            init_called = True
            # Should still work without progress token
            if session and progress_token:
                await session.send_progress_notification(
                    progress_token=progress_token, progress=100, total=100, message="Done"
                )

        from hud.tools.helper.server_initialization import _init_function

        # Call without progress token
        mock_session = MagicMock(spec=ServerSession)
        assert _init_function is not None
        await _init_function(session=mock_session, progress_token=None)

        assert init_called

    @pytest.mark.asyncio
    async def test_multiple_decorators(self):
        """Test behavior when decorator is used multiple times."""

        @mcp_intialize_wrapper
        async def first_init(session=None, progress_token=None):
            return None

        @mcp_intialize_wrapper
        async def second_init(session=None, progress_token=None):
            return None

        from hud.tools.helper.server_initialization import _init_function

        # The last decorated function should be the active one
        assert _init_function is second_init
