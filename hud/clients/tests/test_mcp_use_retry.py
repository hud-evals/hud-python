"""Tests for MCP-use client retry functionality."""

from __future__ import annotations

from unittest.mock import AsyncMock, Mock, patch

import pytest
import requests
from mcp import types

from hud.clients.mcp_use import MCPUseHUDClient
from hud.clients.utils.mcp_use_retry import (
    create_async_retry_wrapper,
    create_retry_session,
    patch_all_sessions,
    patch_mcp_session_http_client,
)
from hud.types import MCPToolCall


class TestRetrySession:
    """Test the retry session creation."""

    def test_create_retry_session(self):
        """Test that retry session is configured correctly."""
        session = create_retry_session(
            max_retries=5,
            retry_status_codes=(500, 502, 503, 504),
            retry_delay=0.5,
            backoff_factor=2.0,
        )

        # Check that session has adapters mounted
        assert "http://" in session.adapters
        assert "https://" in session.adapters

        # Check adapter configuration
        adapter = session.adapters["http://"]
        assert adapter.max_retries.total == 5
        assert 500 in adapter.max_retries.status_forcelist
        assert 502 in adapter.max_retries.status_forcelist
        assert adapter.max_retries.backoff_factor == 2.0

    def test_retry_session_default_values(self):
        """Test retry session with default values."""
        session = create_retry_session()

        adapter = session.adapters["https://"]
        assert adapter.max_retries.total == 3
        assert 502 in adapter.max_retries.status_forcelist
        assert 503 in adapter.max_retries.status_forcelist
        assert 504 in adapter.max_retries.status_forcelist


class TestAsyncRetryWrapper:
    """Test the async retry wrapper functionality."""

    @pytest.mark.asyncio
    async def test_retry_on_error_status_codes(self):
        """Test that async wrapper retries on specific status codes."""
        call_count = 0

        async def mock_func(*args, **kwargs):
            nonlocal call_count
            call_count += 1

            # First two calls fail, third succeeds
            if call_count < 3:
                result = Mock()
                result.status_code = 503  # Service unavailable
                return result

            result = Mock()
            result.status_code = 200
            return result

        wrapped = create_async_retry_wrapper(
            mock_func,
            max_retries=3,
            retry_status_codes=(503,),
            retry_delay=0.01,  # Short delay for testing
        )

        result = await wrapped()
        assert call_count == 3
        assert result.status_code == 200

    @pytest.mark.asyncio
    async def test_retry_on_exception(self):
        """Test that async wrapper retries on exceptions with status codes."""
        call_count = 0

        async def mock_func(*args, **kwargs):
            nonlocal call_count
            call_count += 1

            if call_count < 3:
                raise Exception("HTTP 503 Service Unavailable")

            return Mock(status_code=200)

        wrapped = create_async_retry_wrapper(
            mock_func,
            max_retries=3,
            retry_status_codes=(503,),
            retry_delay=0.01,
        )

        result = await wrapped()
        assert call_count == 3
        assert result.status_code == 200

    @pytest.mark.asyncio
    async def test_no_retry_on_success(self):
        """Test that successful calls don't trigger retries."""
        call_count = 0

        async def mock_func(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return Mock(status_code=200)

        wrapped = create_async_retry_wrapper(mock_func)

        result = await wrapped()
        assert call_count == 1
        assert result.status_code == 200

    @pytest.mark.asyncio
    async def test_max_retries_exceeded(self):
        """Test that retries stop after max attempts."""
        call_count = 0

        async def mock_func(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            raise Exception("HTTP 503 Service Unavailable")

        wrapped = create_async_retry_wrapper(
            mock_func,
            max_retries=2,
            retry_status_codes=(503,),
            retry_delay=0.01,
        )

        with pytest.raises(Exception) as exc_info:
            await wrapped()

        assert "503" in str(exc_info.value)
        assert call_count == 3  # Initial + 2 retries


class TestSessionPatching:
    """Test the session patching functionality."""

    def test_patch_sync_session(self):
        """Test patching a synchronous session."""
        # Create mock session with connector
        mock_session = Mock()
        mock_session.connector = Mock()
        mock_session.connector._connection_manager = Mock()
        mock_session.connector._connection_manager._session = requests.Session()

        # Patch the session
        patch_mcp_session_http_client(mock_session)

        # Verify the session was replaced with retry-enabled one
        patched_session = mock_session.connector._connection_manager._session
        assert "http://" in patched_session.adapters
        assert "https://" in patched_session.adapters

        # Check that it has retry configuration
        adapter = patched_session.adapters["http://"]
        assert hasattr(adapter, "max_retries")

    @pytest.mark.asyncio
    async def test_patch_async_session(self):
        """Test patching an async session."""
        # Create mock async session
        mock_session = Mock()
        mock_session.connector = Mock()
        mock_session.connector.client_session = Mock()

        async def mock_send_request(*args, **kwargs):
            return Mock(status_code=200)

        mock_session.connector.client_session._send_request = mock_send_request

        # Patch the session
        patch_mcp_session_http_client(mock_session)

        # Verify _send_request was wrapped
        wrapped_func = mock_session.connector.client_session._send_request
        assert wrapped_func != mock_send_request  # Function was replaced

        # Test that wrapped function still works
        result = await wrapped_func()
        assert result.status_code == 200

    def test_patch_all_sessions(self):
        """Test patching multiple sessions."""
        # Create mock sessions
        session1 = Mock()
        session1.connector = Mock()
        session1.connector._connection_manager = Mock()
        session1.connector._connection_manager.session = requests.Session()

        session2 = Mock()
        session2.connector = Mock()
        session2.connector.client_session = Mock()
        session2.connector.client_session._send_request = AsyncMock()

        sessions = {"server1": session1, "server2": session2}

        # Patch all sessions
        patch_all_sessions(sessions)

        # Verify both were patched
        assert "http://" in session1.connector._connection_manager.session.adapters
        assert session2.connector.client_session._send_request != AsyncMock


class TestMCPUseClientRetry:
    """Test retry functionality integrated into MCPUseHUDClient."""

    @pytest.mark.asyncio
    async def test_client_applies_retry_on_connect(self):
        """Test that MCPUseHUDClient applies retry logic during connection."""
        config = {"test_server": {"url": "http://localhost:8080"}}
        client = MCPUseHUDClient(config)

        # Mock the MCPUseClient and session creation
        with patch("hud.clients.mcp_use.MCPUseClient") as MockMCPUseClient:
            mock_client = Mock()
            MockMCPUseClient.from_dict.return_value = mock_client

            # Create mock session
            mock_session = Mock()
            mock_session.connector = Mock()
            mock_session.connector.client_session = Mock()
            mock_session.connector.client_session._send_request = AsyncMock()
            mock_session.connector.client_session.list_tools = AsyncMock(
                return_value=Mock(tools=[])
            )

            mock_client.create_all_sessions = AsyncMock(return_value={"test_server": mock_session})

            # Initialize client (which applies retry logic)
            await client.initialize()

            # Verify session was created and patched
            assert len(client._sessions) == 1
            assert "test_server" in client._sessions

    @pytest.mark.asyncio
    async def test_tool_call_with_retry(self):
        """Test that tool calls work with retry logic."""
        config = {"test_server": {"url": "http://localhost:8080"}}
        client = MCPUseHUDClient(config)

        with patch("hud.clients.mcp_use.MCPUseClient") as MockMCPUseClient:
            mock_client = Mock()
            MockMCPUseClient.from_dict.return_value = mock_client

            # Create mock session
            mock_session = Mock()
            mock_session.connector = Mock()
            mock_session.connector.client_session = Mock()

            # Mock tool listing
            test_tool = types.Tool(
                name="test_tool",
                description="Test tool",
                inputSchema={"type": "object"},
            )
            mock_session.connector.client_session.list_tools = AsyncMock(
                return_value=Mock(tools=[test_tool])
            )

            # Mock tool call with simulated retry
            call_count = 0

            async def mock_call_tool(name, arguments):
                nonlocal call_count
                call_count += 1

                # First call fails, second succeeds
                if call_count == 1:
                    raise Exception("HTTP 503 Service Unavailable")

                return Mock(
                    content=[types.TextContent(type="text", text="Success")],
                    isError=False,
                    structuredContent=None,
                )

            mock_session.connector.client_session.call_tool = mock_call_tool
            mock_session.connector.client_session._send_request = AsyncMock()

            mock_client.create_all_sessions = AsyncMock(return_value={"test_server": mock_session})

            # Initialize and call tool
            await client.initialize()

            # Wrap call_tool with retry for this test
            original_call = mock_session.connector.client_session.call_tool
            mock_session.connector.client_session.call_tool = create_async_retry_wrapper(
                original_call,
                max_retries=2,
                retry_status_codes=(503,),
                retry_delay=0.01,
            )

            result = await client.call_tool(MCPToolCall(name="test_tool", arguments={}))

            # Verify retry worked
            assert call_count == 2  # Failed once, then succeeded
            assert not result.isError
            assert result.content[0].text == "Success"

    @pytest.mark.asyncio
    async def test_resource_read_with_retry(self):
        """Test that resource reading works with retry logic."""
        config = {"test_server": {"url": "http://localhost:8080"}}
        client = MCPUseHUDClient(config)

        with patch("hud.clients.mcp_use.MCPUseClient") as MockMCPUseClient:
            mock_client = Mock()
            MockMCPUseClient.from_dict.return_value = mock_client

            # Create mock session
            mock_session = Mock()
            mock_session.connector = Mock()
            mock_session.connector.client_session = Mock()
            mock_session.connector.client_session.list_tools = AsyncMock(
                return_value=Mock(tools=[])
            )

            # Mock resource read with simulated retry
            call_count = 0

            async def mock_read_resource(uri):
                nonlocal call_count
                call_count += 1

                # First call fails, second succeeds
                if call_count == 1:
                    raise Exception("HTTP 502 Bad Gateway")

                return Mock(contents=[Mock(text='{"status": "ok"}')])

            mock_session.connector.client_session.read_resource = mock_read_resource
            mock_session.connector.client_session._send_request = AsyncMock()

            mock_client.create_all_sessions = AsyncMock(return_value={"test_server": mock_session})

            # Initialize
            await client.initialize()

            # Wrap read_resource with retry for this test
            original_read = mock_session.connector.client_session.read_resource
            mock_session.connector.client_session.read_resource = create_async_retry_wrapper(
                original_read,
                max_retries=2,
                retry_status_codes=(502,),
                retry_delay=0.01,
            )

            result = await client.read_resource("test://resource")

            # Verify retry worked
            assert call_count == 2  # Failed once, then succeeded
            assert result is not None
            assert result.contents[0].text == '{"status": "ok"}'


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
