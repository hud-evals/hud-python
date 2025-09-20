"""Tests for low-level server components."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from hud.server.low_level import InitSession, LowLevelServerWithInit


class TestInitSession:
    """Test InitSession class functionality."""

    def test_init_session_basic(self) -> None:
        """Test InitSession initialization."""
        # Mock the required streams and options
        read_stream = MagicMock()
        write_stream = MagicMock()
        init_opts = MagicMock()

        init_fn = AsyncMock()

        session = InitSession(
            read_stream=read_stream,
            write_stream=write_stream,
            init_opts=init_opts,
            init_fn=init_fn
        )

        assert session._init_fn is init_fn
        assert session._did_init is False

    def test_init_session_stateless(self) -> None:
        """Test InitSession with stateless=True skips initialization."""
        read_stream = MagicMock()
        write_stream = MagicMock()
        init_opts = MagicMock()

        init_fn = AsyncMock()

        session = InitSession(
            read_stream=read_stream,
            write_stream=write_stream,
            init_opts=init_opts,
            init_fn=init_fn,
            stateless=True
        )

        assert session._did_init is True  # Should be True for stateless

    @pytest.mark.asyncio
    async def test_init_session_initialization_hook(self) -> None:
        """Test that InitSession calls init_fn on initialize request."""
        read_stream = MagicMock()
        write_stream = MagicMock()
        init_opts = MagicMock()

        init_fn = AsyncMock(return_value=None)

        session = InitSession(
            read_stream=read_stream,
            write_stream=write_stream,
            init_opts=init_opts,
            init_fn=init_fn
        )

        # Mock the responder for an initialize request
        responder = MagicMock()

        # Create a mock InitializeRequest
        init_request = MagicMock()
        init_request.root = MagicMock()
        init_request.root.id = "test-id"
        init_request.params.meta = MagicMock()

        # Mock isinstance check to return True for InitializeRequest
        fake_init_class = type('FakeInitializeRequest', (), {})
        with patch('hud.server.low_level.types.InitializeRequest', fake_init_class):
            init_request.root.__class__ = fake_init_class
            responder.request = init_request

            # Mock the parent _received_request to return None
            with patch('mcp.server.session.ServerSession._received_request', return_value=None) as mock_super:
                result = await session._received_request(responder)

                # Should call init_fn
                init_fn.assert_called_once()
                args, kwargs = init_fn.call_args
                ctx = args[0]

                # Check context was created properly
                assert ctx.request_id == "test-id"
                assert ctx.session is session
                assert ctx.meta is init_request.root.params.meta

                # Should call parent method
                mock_super.assert_called_once_with(responder)

                # Should mark as initialized
                assert session._did_init is True

    @pytest.mark.asyncio
    async def test_init_session_skip_already_initialized(self) -> None:
        """Test that InitSession skips init_fn if already initialized."""
        read_stream = MagicMock()
        write_stream = MagicMock()
        init_opts = MagicMock()

        init_fn = AsyncMock()

        session = InitSession(
            read_stream=read_stream,
            write_stream=write_stream,
            init_opts=init_opts,
            init_fn=init_fn
        )

        # Mark as already initialized
        session._did_init = True

        responder = MagicMock()
        init_request = MagicMock()
        init_request.root = MagicMock()

        fake_init_class = type('FakeInitializeRequest', (), {})
        with patch('hud.server.low_level.types.InitializeRequest', fake_init_class):
            init_request.root.__class__ = fake_init_class
            responder.request = init_request

            with patch('mcp.server.session.ServerSession._received_request', return_value=None) as mock_super:
                await session._received_request(responder)

                # Should NOT call init_fn
                init_fn.assert_not_called()

                # Should call parent method
                mock_super.assert_called_once_with(responder)

    @pytest.mark.asyncio
    async def test_init_session_no_init_fn(self) -> None:
        """Test InitSession when no init_fn is provided."""
        read_stream = MagicMock()
        write_stream = MagicMock()
        init_opts = MagicMock()

        session = InitSession(
            read_stream=read_stream,
            write_stream=write_stream,
            init_opts=init_opts,
            init_fn=None
        )

        responder = MagicMock()
        init_request = MagicMock()
        init_request.root = MagicMock()

        fake_init_class = type('FakeInitializeRequest', (), {})
        with patch('hud.server.low_level.types.InitializeRequest', fake_init_class):
            init_request.root.__class__ = fake_init_class
            responder.request = init_request

            with patch('mcp.server.session.ServerSession._received_request', return_value=None) as mock_super:
                await session._received_request(responder)

                # Should call parent method
                mock_super.assert_called_once_with(responder)

    @pytest.mark.asyncio
    async def test_init_session_init_fn_exception(self) -> None:
        """Test InitSession handles exceptions in init_fn."""
        read_stream = MagicMock()
        write_stream = MagicMock()
        init_opts = MagicMock()

        init_fn = AsyncMock(side_effect=ValueError("Init failed"))

        session = InitSession(
            read_stream=read_stream,
            write_stream=write_stream,
            init_opts=init_opts,
            init_fn=init_fn
        )

        responder = MagicMock()
        init_request = MagicMock()
        init_request.root = MagicMock()
        init_request.root.id = "test-id"
        init_request.root.params = MagicMock()
        init_request.root.params.meta = MagicMock()
        init_request.root.params.meta.progressToken = "progress-123"

        fake_init_class = type('FakeInitializeRequest', (), {})
        with patch('hud.server.low_level.types.InitializeRequest', fake_init_class):
            init_request.root.__class__ = fake_init_class
            responder.request = init_request

            # Mock progress notification
            session.send_progress_notification = AsyncMock()

            with pytest.raises(ValueError, match="Init failed"):
                await session._received_request(responder)

            # Should still mark as initialized even on failure
            assert session._did_init is True

    @pytest.mark.asyncio
    async def test_init_session_non_initialize_request(self) -> None:
        """Test InitSession passes through non-initialize requests."""
        read_stream = MagicMock()
        write_stream = MagicMock()
        init_opts = MagicMock()

        init_fn = AsyncMock()

        session = InitSession(
            read_stream=read_stream,
            write_stream=write_stream,
            init_opts=init_opts,
            init_fn=init_fn
        )

        responder = MagicMock()
        # Non-initialize request
        other_request = MagicMock()
        other_request.root = MagicMock()
        responder.request = other_request

        # Don't patch InitializeRequest - the request is not an InitializeRequest, so isinstance should return False

        with patch('mcp.server.session.ServerSession._received_request', return_value="result") as mock_super:
            result = await session._received_request(responder)

            # Should NOT call init_fn
            init_fn.assert_not_called()

            # Should call parent method and return its result
            mock_super.assert_called_once_with(responder)
            assert result == "result"


class TestLowLevelServerWithInit:
    """Test LowLevelServerWithInit class functionality."""

    def test_low_level_server_init(self) -> None:
        """Test LowLevelServerWithInit initialization."""
        init_fn = AsyncMock()

        server = LowLevelServerWithInit(
            name="test_server",
            version="1.0.0",
            instructions="Test instructions",
            init_fn=init_fn
        )

        assert server._init_fn is init_fn
        assert server.name == "test_server"
        assert server.version == "1.0.0"
        assert server.instructions == "Test instructions"

    def test_low_level_server_init_no_init_fn(self) -> None:
        """Test LowLevelServerWithInit initialization without init_fn."""
        server = LowLevelServerWithInit(
            name="test_server",
            version="1.0.0",
            instructions="Test instructions"
        )

        assert server._init_fn is None

