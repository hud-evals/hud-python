"""Tests for the HUD SDK Exception System.

This module tests the intelligent exception handling with automatic error
classification and helpful hints for users.
"""

from __future__ import annotations

import json
from unittest.mock import Mock

import httpx
import pytest

from hud.shared.exceptions import (
    HudAuthenticationError,
    HudClientError,
    HudConfigError,
    HudException,
    HudRateLimitError,
    HudRequestError,
    HudTimeoutError,
    HudToolNotFoundError,
)
from hud.shared.hints import (
    CLIENT_NOT_INITIALIZED,
    HUD_API_KEY_MISSING,
    INVALID_CONFIG,
    PRO_PLAN_REQUIRED,
    RATE_LIMIT_HIT,
    TOOL_NOT_FOUND,
)


class TestHudExceptionAutoConversion:
    """Test automatic exception conversion via 'raise HudException() from e'."""

    def test_client_not_initialized_error(self):
        """Test that 'not initialized' errors become HudClientError."""
        try:
            raise ValueError("Client not initialized - call initialize() first")
        except Exception as e:
            with pytest.raises(HudClientError) as exc_info:
                raise HudException from e

            assert exc_info.value.hints == [CLIENT_NOT_INITIALIZED]
            assert str(exc_info.value) == "Client not initialized - call initialize() first"

    def test_not_connected_error(self):
        """Test that 'not connected' errors become HudClientError."""
        try:
            raise RuntimeError("Session not connected to server")
        except Exception as e:
            with pytest.raises(HudClientError) as exc_info:
                raise HudException from e

            assert exc_info.value.hints == [CLIENT_NOT_INITIALIZED]

    def test_config_invalid_json_error(self):
        """Test that JSON errors become HudConfigError."""
        try:
            json.loads("{invalid json}")
        except json.JSONDecodeError as e:
            with pytest.raises(HudConfigError) as exc_info:
                raise HudException from e

            assert exc_info.value.hints == [INVALID_CONFIG]

    def test_config_error_keyword(self):
        """Test that errors with 'config' become HudConfigError."""
        try:
            raise ValueError("Invalid config: missing required field 'url'")
        except Exception as e:
            with pytest.raises(HudConfigError) as exc_info:
                raise HudException from e

            assert exc_info.value.hints == [INVALID_CONFIG]

    def test_tool_not_found_error(self):
        """Test that tool not found errors become HudToolNotFoundError."""
        try:
            raise KeyError("Tool 'missing_tool' not found in registry")
        except Exception as e:
            with pytest.raises(HudToolNotFoundError) as exc_info:
                raise HudException from e

            assert exc_info.value.hints == [TOOL_NOT_FOUND]

    def test_tool_not_exist_error(self):
        """Test that tool not exist errors become HudToolNotFoundError."""
        try:
            raise RuntimeError("Tool does not exist: calculator")
        except Exception as e:
            with pytest.raises(HudToolNotFoundError) as exc_info:
                raise HudException from e

            assert exc_info.value.hints == [TOOL_NOT_FOUND]

    def test_hud_api_key_error(self):
        """Test that HUD API key errors become HudAuthenticationError."""
        try:
            raise ValueError("API key missing for mcp.hud.ai")
        except Exception as e:
            with pytest.raises(HudAuthenticationError) as exc_info:
                raise HudException from e

            assert exc_info.value.hints == [HUD_API_KEY_MISSING]

    def test_hud_authorization_error(self):
        """Test that HUD authorization errors become HudAuthenticationError."""
        try:
            raise PermissionError("Authorization failed for HUD API")
        except Exception as e:
            with pytest.raises(HudAuthenticationError) as exc_info:
                raise HudException from e

            assert exc_info.value.hints == [HUD_API_KEY_MISSING]

    def test_rate_limit_error(self):
        """Test that rate limit errors become HudRateLimitError."""
        try:
            raise RuntimeError("Rate limit exceeded")
        except Exception as e:
            with pytest.raises(HudRateLimitError) as exc_info:
                raise HudException from e

            assert exc_info.value.hints == [RATE_LIMIT_HIT]

    def test_too_many_requests_error(self):
        """Test that 'too many request' errors become HudRateLimitError."""
        try:
            raise httpx.HTTPStatusError("Too many requests", request=Mock(), response=Mock())
        except Exception as e:
            with pytest.raises(HudRateLimitError) as exc_info:
                raise HudException from e

            assert exc_info.value.hints == [RATE_LIMIT_HIT]

    def test_timeout_error(self):
        """Test that TimeoutError becomes HudTimeoutError."""
        try:
            raise TimeoutError("Operation timed out")
        except Exception as e:
            with pytest.raises(HudTimeoutError) as exc_info:
                raise HudException from e

            assert exc_info.value.hints == []  # No default hints for timeout

    def test_asyncio_timeout_error(self):
        """Test that asyncio.TimeoutError becomes HudTimeoutError."""
        try:
            raise TimeoutError("Async operation timed out")
        except Exception as e:
            with pytest.raises(HudTimeoutError) as exc_info:
                raise HudException from e

            assert str(exc_info.value) == "Async operation timed out"

    def test_generic_error_remains_hudexception(self):
        """Uncategorized errors become base HudException with original message."""
        try:
            raise ValueError("Some random error")
        except Exception as e:
            with pytest.raises(HudException) as exc_info:
                raise HudException from e
            # Should be base HudException, not subclass
            assert type(exc_info.value) is HudException
            assert str(exc_info.value) == "Some random error"

    def test_custom_message_override(self):
        """Custom message should be used for categorized errors."""
        try:
            raise ValueError("Client not initialized - call initialize() first")
        except Exception as e:
            with pytest.raises(HudClientError) as exc_info:
                raise HudException("Custom error message") from e
            assert str(exc_info.value) == "Custom error message"

    def test_already_hud_exception_passthrough(self):
        """Test that existing HudExceptions are not re-wrapped."""
        original = HudAuthenticationError("Already a HUD exception")

        try:
            raise original
        except Exception as e:
            with pytest.raises(HudAuthenticationError) as exc_info:
                raise HudException from e

            # Should be the same instance
            assert exc_info.value is original


class TestHudRequestError:
    """Test HudRequestError specific behavior."""

    def test_401_adds_auth_hint(self):
        """Test that 401 status adds authentication hint."""
        error = HudRequestError("Unauthorized", status_code=401)
        assert HUD_API_KEY_MISSING in error.hints

    def test_403_adds_auth_hint(self):
        """Test that 403 status adds authentication hint."""
        error = HudRequestError("Forbidden", status_code=403)
        assert HUD_API_KEY_MISSING in error.hints

    def test_403_pro_plan_message_sets_pro_hint(self):
        """403 with Pro wording should map to PRO_PLAN_REQUIRED, not auth."""
        error = HudRequestError("Feature requires Pro plan", status_code=403)
        assert PRO_PLAN_REQUIRED in error.hints
        assert HUD_API_KEY_MISSING not in error.hints

    def test_403_pro_plan_detail_sets_pro_hint(self):
        """403 with detail indicating Pro should map to PRO_PLAN_REQUIRED."""
        error = HudRequestError(
            "Forbidden",
            status_code=403,
            response_json={"detail": "Requires Pro plan"},
        )
        assert PRO_PLAN_REQUIRED in error.hints
        assert HUD_API_KEY_MISSING not in error.hints

    def test_429_adds_rate_limit_hint(self):
        """Test that 429 status adds rate limit hint."""
        error = HudRequestError("Too Many Requests", status_code=429)
        assert RATE_LIMIT_HIT in error.hints

    def test_other_status_no_default_hints(self):
        """Test that other status codes don't add default hints."""
        error = HudRequestError("Server Error", status_code=500)
        assert error.hints == []

    def test_explicit_hints_override_defaults(self):
        """Test that explicit hints override status-based defaults."""
        from hud.shared.hints import Hint

        custom_hint = Hint(title="Custom Error", message="This is a custom hint")
        error = HudRequestError("Unauthorized", status_code=401, hints=[custom_hint])
        assert error.hints == [custom_hint]
        assert HUD_API_KEY_MISSING not in error.hints

    def test_from_httpx_error(self):
        """Test creating from HTTPx error."""
        request = httpx.Request("GET", "https://api.test.com")
        response = httpx.Response(404, json={"detail": "Not found"}, request=request)
        httpx_error = httpx.HTTPStatusError("Not found", request=request, response=response)

        error = HudRequestError.from_httpx_error(httpx_error, context="Testing")

        assert error.status_code == 404
        assert "Testing" in str(error)
        assert "Not found" in str(error)
        assert error.response_json == {"detail": "Not found"}


class TestMCPErrorHandling:
    """Test handling of MCP-specific errors."""

    @pytest.mark.asyncio
    async def test_mcp_error_handling(self):
        """Test that McpError is handled appropriately."""
        # Create a dynamic class named "McpError" to trigger name-based detection
        McpError = type("McpError", (Exception,), {})

        try:
            raise McpError("MCP protocol error: Unknown method")
        except Exception as e:
            # This would typically be caught in the client code
            # and re-raised as HudException
            with pytest.raises(HudException) as exc_info:
                raise HudException from e

            assert "MCP protocol error" in str(exc_info.value)
            assert "MCP protocol error" in str(exc_info.value)

    def test_mcp_tool_error_result(self):
        """Test handling of MCP tool execution errors (isError: true)."""
        # Simulate an MCP tool result with error
        tool_result = {
            "content": [{"type": "text", "text": "Failed to fetch data: API rate limit exceeded"}],
            "isError": True,
        }

        # In real usage, this would be checked in the client
        if tool_result.get("isError"):
            error_text = tool_result["content"][0]["text"]

            try:
                raise RuntimeError(error_text)
            except Exception as e:
                with pytest.raises(HudRateLimitError) as exc_info:
                    raise HudException from e

                assert exc_info.value.hints == [RATE_LIMIT_HIT]


class TestExceptionIntegration:
    """Test exception handling in integrated scenarios."""

    @pytest.mark.asyncio
    async def test_client_initialization_flow(self):
        """Test exception flow during client initialization."""
        from hud.clients.base import BaseHUDClient

        # Mock a client that fails initialization
        client = Mock(spec=BaseHUDClient)

        # Simulate missing config
        try:
            if not hasattr(client, "_mcp_config"):
                raise ValueError("MCP config not set")
        except Exception as e:
            with pytest.raises(HudConfigError) as exc_info:
                raise HudException from e

            assert exc_info.value.hints == [INVALID_CONFIG]

    def test_json_parsing_flow(self):
        """Test exception flow during JSON parsing."""
        invalid_json = '{"incomplete": '

        try:
            _ = json.loads(invalid_json)
        except json.JSONDecodeError as e:
            with pytest.raises(HudConfigError) as exc_info:
                raise HudException from e

            assert "Expecting value" in str(exc_info.value)
            assert exc_info.value.hints == [INVALID_CONFIG]

    @pytest.mark.asyncio
    async def test_network_error_flow(self):
        """Test exception flow during network operations."""
        # Simulate a connection error
        try:
            raise ConnectionError("Connection refused")
        except Exception as e:
            with pytest.raises(HudException) as exc_info:
                raise HudException("Failed to connect to server") from e

            # Should remain base HudException for generic connection errors
            assert type(exc_info.value) is HudException
            assert str(exc_info.value) == "Failed to connect to server"


class TestExceptionRendering:
    """Test how exceptions are rendered and displayed."""

    def test_exception_string_representation(self):
        """Test __str__ method of exceptions."""
        error = HudRequestError(
            "Request failed", status_code=404, response_json={"error": "Not found"}
        )

        error_str = str(error)
        assert "Request failed" in error_str
        assert "Status: 404" in error_str
        assert "Response JSON: {'error': 'Not found'}" in error_str

    def test_exception_with_hints(self):
        """Test that exceptions carry their hints properly."""
        error = HudAuthenticationError("API key missing")

        assert len(error.hints) == 1
        assert error.hints[0] == HUD_API_KEY_MISSING
        assert error.hints[0].title == "HUD API key required"
        # Hint copy evolved; keep the assertion robust to minor copy changes
        assert "Set HUD_API_KEY" in error.hints[0].tips[0]

    def test_exception_type_preservation(self):
        """Test that exception types are preserved through conversion."""
        test_cases = [
            ("Client not initialized", HudClientError),
            ("Invalid JSON config", HudConfigError),
            ("Tool 'test' not found", HudToolNotFoundError),
            ("API key missing for HUD", HudAuthenticationError),
            ("Rate limit exceeded", HudRateLimitError),
            (TimeoutError("Timeout"), HudTimeoutError),
        ]

        for error_msg, expected_type in test_cases:
            try:
                if isinstance(error_msg, Exception):
                    raise error_msg
                else:
                    raise ValueError(error_msg)
            except Exception as e:
                with pytest.raises(expected_type):
                    raise HudException from e


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_none_exception_handling(self):
        """Test handling when no exception context exists."""
        # When there's no active exception, should create normal HudException
        error = HudException("No chained exception")
        assert type(error) is HudException
        assert str(error) == "No chained exception"

    def test_baseexception_not_converted(self):
        """Test that BaseException (not Exception) is not converted."""
        try:
            raise KeyboardInterrupt("User interrupted")
        except BaseException:
            # Should not attempt to convert BaseException
            error = HudException("Interrupted")
            assert type(error) is HudException

    def test_empty_error_message(self):
        """Empty message still results in a HudException instance."""
        try:
            raise ValueError("")
        except Exception as e:
            with pytest.raises(HudException):
                raise HudException from e

    def test_circular_exception_chain(self):
        """Test that we don't create circular exception chains."""
        original = HudAuthenticationError("Original")

        try:
            raise original
        except HudException as e:
            # Raising HudException from HudException should not re-wrap
            with pytest.raises(HudAuthenticationError) as exc_info:
                raise HudException from e

            assert exc_info.value is original
