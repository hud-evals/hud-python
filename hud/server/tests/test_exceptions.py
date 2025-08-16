"""Tests for server exceptions."""

from __future__ import annotations

from unittest.mock import MagicMock

import httpx

from hud.server.exceptions import (
    GymMakeException,
    HudAuthenticationError,
    HudException,
    HudNetworkError,
    HudRequestError,
    HudTimeoutError,
)


class TestHudRequestError:
    """Test HudRequestError exception."""

    def test_from_httpx_error_with_json_detail(self):
        """Test creating from httpx error with JSON detail."""
        response = MagicMock()
        response.status_code = 400
        response.json.return_value = {"detail": "Bad request details"}

        error = httpx.HTTPStatusError("Test", request=MagicMock(), response=response)

        hud_error = HudRequestError.from_httpx_error(error, context="Test context")

        assert hud_error.status_code == 400
        assert "Test context" in str(hud_error)
        assert "Bad request details" in str(hud_error)

    def test_from_httpx_error_with_small_json_no_detail(self):
        """Test creating from httpx error with small JSON but no detail field."""
        response = MagicMock()
        response.status_code = 400
        response.json.return_value = {"error": "test", "code": 123}

        error = httpx.HTTPStatusError("Test", request=MagicMock(), response=response)

        hud_error = HudRequestError.from_httpx_error(error)

        assert hud_error.status_code == 400
        assert "JSON response:" in str(hud_error)
        # Check for the dictionary representation (not exact JSON string)
        assert "'error': 'test'" in str(hud_error)
        assert "'code': 123" in str(hud_error)

    def test_from_httpx_error_json_parse_failure(self):
        """Test creating from httpx error when JSON parsing fails."""
        response = MagicMock()
        response.status_code = 500
        response.json.side_effect = ValueError("Invalid JSON")

        error = httpx.HTTPStatusError("Test", request=MagicMock(), response=response)

        hud_error = HudRequestError.from_httpx_error(error)

        assert hud_error.status_code == 500
        assert "Request failed with status 500" in str(hud_error)

    def test_from_httpx_error_large_json_response(self):
        """Test creating from httpx error with large JSON response."""
        response = MagicMock()
        response.status_code = 400
        # Large JSON object (more than 5 keys)
        response.json.return_value = {
            "field1": "value1",
            "field2": "value2",
            "field3": "value3",
            "field4": "value4",
            "field5": "value5",
            "field6": "value6",
        }

        error = httpx.HTTPStatusError("Test", request=MagicMock(), response=response)

        hud_error = HudRequestError.from_httpx_error(error)

        assert hud_error.status_code == 400
        # Should not include JSON in message since it's large
        assert "JSON response:" not in str(hud_error)
        assert "Request failed with status 400" in str(hud_error)

    def test_str_method(self):
        """Test string representation of HudRequestError."""
        error = HudRequestError("Test error message", 404, '{"extra": "data"}')

        error_str = str(error)
        assert "Test error message" in error_str
        assert "404" in error_str
        assert "extra" in error_str


class TestHudNetworkError:
    """Test HudNetworkError exception."""

    def test_initialization_and_str(self):
        """Test HudNetworkError initialization and string representation."""
        error = HudNetworkError("Network failure: Connection refused")

        error_str = str(error)
        assert "Network failure" in error_str
        assert "Connection refused" in error_str


class TestHudTimeoutError:
    """Test HudTimeoutError exception."""

    def test_initialization(self):
        """Test HudTimeoutError initialization."""
        error = HudTimeoutError("Request timed out after 30.0 seconds")

        error_str = str(error)
        assert "Request timed out" in error_str
        assert "30.0" in error_str

    def test_str_method(self):
        """Test string representation of HudTimeoutError."""
        error = HudTimeoutError("Timeout occurred after 60.0 seconds")

        error_str = str(error)
        assert "Timeout occurred" in error_str
        assert "60.0" in error_str


class TestHudAuthenticationError:
    """Test HudAuthenticationError exception."""

    def test_inheritance(self):
        """Test that HudAuthenticationError inherits from HudException."""
        error = HudAuthenticationError("Auth failed")

        assert isinstance(error, HudException)
        error_str = str(error)
        assert "Auth failed" in error_str


class TestGymMakeException:
    """Test GymMakeException."""

    def test_initialization_and_str(self):
        """Test GymMakeException initialization and string representation."""
        data = {"env_id": "test-env", "error": "invalid config"}
        error = GymMakeException("Failed to create environment", data)

        assert error.data == data

        error_str = str(error)
        assert "Failed to create environment" in error_str
        assert "Data:" in error_str
        assert "env_id" in error_str
        assert "test-env" in error_str
        assert "invalid config" in error_str
