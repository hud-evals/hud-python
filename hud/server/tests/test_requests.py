"""Tests for the HTTP request utilities in the HUD API."""

from __future__ import annotations

from http import HTTPStatus
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, Mock, patch

import httpx
import pytest

from hud.server.exceptions import (
    HudAuthenticationError,
    HudNetworkError,
    HudRequestError,
    HudTimeoutError,
)
from hud.server.requests import (
    _handle_retry,
    make_request,
    make_request_sync,
)

if TYPE_CHECKING:
    from collections.abc import Callable


def _create_mock_response(
    status_code: int = 200,
    json_data: dict[str, Any] | None = None,
    raise_exception: Exception | None = None,
) -> Callable[[httpx.Request], httpx.Response]:
    """Create a mock response handler for httpx.MockTransport."""

    def handler(request: httpx.Request) -> httpx.Response:
        if "Authorization" not in request.headers:
            return httpx.Response(HTTPStatus.UNAUTHORIZED, json={"error": "Unauthorized"})

        if raise_exception:
            raise raise_exception

        return httpx.Response(status_code, json=json_data or {"result": "success"}, request=request)

    return handler


@pytest.mark.asyncio
async def test_handle_retry():
    """Test the retry handler."""
    with patch("asyncio.sleep") as mock_sleep:
        mock_sleep.return_value = None
        await _handle_retry(
            attempt=2,
            max_retries=3,
            retry_delay=1.0,
            url="https://example.com",
            error_msg="Test error",
        )

        # Check exponential backoff formula: delay * (2 ^ (attempt - 1))
        mock_sleep.assert_awaited_once_with(2.0)


@pytest.mark.asyncio
async def test_make_request_success():
    """Test successful async request."""
    expected_data = {"id": "123", "name": "test"}
    async_client = httpx.AsyncClient(
        transport=httpx.MockTransport(_create_mock_response(200, expected_data))
    )
    result = await make_request(
        "GET", "https://api.test.com/data", api_key="test-key", client=async_client
    )
    assert result == expected_data


@pytest.mark.asyncio
async def test_make_request_no_api_key():
    """Test request without API key."""
    with pytest.raises(HudAuthenticationError):
        await make_request("GET", "https://api.test.com/data", api_key=None)


@pytest.mark.asyncio
async def test_make_request_http_error():
    """Test HTTP error handling."""
    async_client = httpx.AsyncClient(
        transport=httpx.MockTransport(_create_mock_response(404, {"error": "Not found"}))
    )

    with pytest.raises(HudRequestError) as excinfo:
        await make_request(
            "GET", "https://api.test.com/data", api_key="test-key", client=async_client
        )

    assert "404" in str(excinfo.value)


@pytest.mark.asyncio
async def test_make_request_network_error():
    """Test network error handling with retry exhaustion."""
    request_error = httpx.RequestError(
        "Connection error", request=httpx.Request("GET", "https://api.test.com")
    )
    async_client = httpx.AsyncClient(
        transport=httpx.MockTransport(_create_mock_response(raise_exception=request_error))
    )

    # Replace handle_retry to avoid sleep
    with patch("hud.server.requests._handle_retry", AsyncMock()) as mock_retry:
        mock_retry.return_value = None

        with pytest.raises(HudNetworkError) as excinfo:
            await make_request(
                "GET",
                "https://api.test.com/data",
                api_key="test-key",
                max_retries=2,
                retry_delay=0.01,
                client=async_client,
            )

        assert "Connection error" in str(excinfo.value)


@pytest.mark.asyncio
async def test_make_request_timeout():
    """Test timeout error handling."""
    timeout_error = httpx.TimeoutException(
        "Request timed out", request=httpx.Request("GET", "https://api.test.com")
    )
    async_client = httpx.AsyncClient(
        transport=httpx.MockTransport(_create_mock_response(raise_exception=timeout_error))
    )

    with pytest.raises(HudTimeoutError) as excinfo:
        await make_request(
            "GET", "https://api.test.com/data", api_key="test-key", client=async_client
        )

    assert "timed out" in str(excinfo.value)


@pytest.mark.asyncio
async def test_make_request_unexpected_error():
    """Test handling of unexpected errors."""
    unexpected_error = ValueError("Unexpected error")
    async_client = httpx.AsyncClient(
        transport=httpx.MockTransport(_create_mock_response(raise_exception=unexpected_error))
    )
    with pytest.raises(HudRequestError) as excinfo:
        await make_request(
            "GET", "https://api.test.com/data", api_key="test-key", client=async_client
        )

    assert "Unexpected error" in str(excinfo.value)


@pytest.mark.asyncio
async def test_make_request_auto_client_creation(mocker):
    """Test automatic client creation when not provided."""
    mock_create_client = mocker.patch("hud.server.requests._create_default_async_client")
    mock_client = AsyncMock()
    mock_client.request.return_value = httpx.Response(
        200, json={"result": "success"}, request=httpx.Request("GET", "https://api.test.com")
    )
    mock_client.aclose = AsyncMock()
    mock_create_client.return_value = mock_client

    result = await make_request("GET", "https://api.test.com/data", api_key="test-key")

    assert result == {"result": "success"}
    mock_client.aclose.assert_awaited_once()


def test_make_request_sync_success():
    """Test successful sync request."""
    expected_data = {"id": "123", "name": "test"}
    sync_client = httpx.Client(
        transport=httpx.MockTransport(_create_mock_response(200, expected_data))
    )

    result = make_request_sync(
        "GET", "https://api.test.com/data", api_key="test-key", client=sync_client
    )

    assert result == expected_data


def test_make_request_sync_no_api_key():
    """Test sync request without API key."""
    with pytest.raises(HudAuthenticationError):
        make_request_sync("GET", "https://api.test.com/data", api_key=None)


def test_make_request_sync_http_error():
    """Test HTTP error handling."""
    sync_client = httpx.Client(
        transport=httpx.MockTransport(_create_mock_response(404, {"error": "Not found"}))
    )
    with pytest.raises(HudRequestError) as excinfo:
        make_request_sync(
            "GET", "https://api.test.com/data", api_key="test-key", client=sync_client
        )

    assert "404" in str(excinfo.value)


def test_make_request_sync_network_error():
    """Test network error handling with retry exhaustion."""
    request_error = httpx.RequestError(
        "Connection error", request=httpx.Request("GET", "https://api.test.com")
    )
    sync_client = httpx.Client(
        transport=httpx.MockTransport(_create_mock_response(raise_exception=request_error))
    )
    with patch("time.sleep", lambda _: None):
        with pytest.raises(HudNetworkError) as excinfo:
            make_request_sync(
                "GET",
                "https://api.test.com/data",
                api_key="test-key",
                max_retries=2,
                retry_delay=0.01,
                client=sync_client,
            )

        assert "Connection error" in str(excinfo.value)


def test_make_request_sync_timeout():
    """Test timeout error handling."""
    timeout_error = httpx.TimeoutException(
        "Request timed out", request=httpx.Request("GET", "https://api.test.com")
    )
    sync_client = httpx.Client(
        transport=httpx.MockTransport(_create_mock_response(raise_exception=timeout_error))
    )
    with pytest.raises(HudTimeoutError) as excinfo:
        make_request_sync(
            "GET", "https://api.test.com/data", api_key="test-key", client=sync_client
        )

    assert "timed out" in str(excinfo.value)


def test_make_request_sync_unexpected_error():
    """Test handling of unexpected errors."""
    unexpected_error = ValueError("Unexpected error")
    sync_client = httpx.Client(
        transport=httpx.MockTransport(_create_mock_response(raise_exception=unexpected_error))
    )

    with pytest.raises(HudRequestError) as excinfo:
        make_request_sync(
            "GET", "https://api.test.com/data", api_key="test-key", client=sync_client
        )

    assert "Unexpected error" in str(excinfo.value)


def test_make_request_sync_auto_client_creation():
    """Test automatic client creation when not provided."""
    with patch("hud.server.requests._create_default_sync_client") as mock_create_client:
        mock_client = Mock()
        mock_client.request.return_value = httpx.Response(
            200, json={"result": "success"}, request=httpx.Request("GET", "https://api.test.com")
        )
        mock_client.close = Mock()
        mock_create_client.return_value = mock_client

        result = make_request_sync("GET", "https://api.test.com/data", api_key="test-key")

        assert result == {"result": "success"}
        mock_client.close.assert_called_once()
