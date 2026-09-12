"""Tests for the HTTP request utilities in the HUD API."""

from __future__ import annotations

from http import HTTPStatus
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, Mock, patch

import httpx
import pytest

from hud.utils.exceptions import (
    HudAuthenticationError,
    HudNetworkError,
    HudRequestError,
    HudTimeoutError,
)
from hud.utils.requests import (
    make_request,
    make_request_sync,
)

if TYPE_CHECKING:
    from collections.abc import Callable


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_requests_retry_transient_responses_with_shared_backoff(asynchronous):
    calls = 0

    def handle(request):
        nonlocal calls
        calls += 1
        return httpx.Response(503 if calls < 3 else 200, json={"ok": True})

    transport = httpx.MockTransport(handle)
    with (
        patch("hud.utils.requests.time.sleep") as sync_sleep,
        patch("hud.utils.requests.asyncio.sleep", new_callable=AsyncMock) as async_sleep,
    ):
        if asynchronous:
            async with httpx.AsyncClient(transport=transport) as client:
                result = await make_request(
                    "GET", "https://test/data", api_key="key", client=client
                )
        else:
            with httpx.Client(transport=transport) as client:
                result = make_request_sync("GET", "https://test/data", api_key="key", client=client)
        sleep = async_sleep if asynchronous else sync_sleep
        assert [call.args[0] for call in sleep.call_args_list] == [2.0, 4.0]
    assert result == {"ok": True}
    assert calls == 3


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("status", [402, 429])
async def test_requests_preserve_status_hints_without_retry(asynchronous, status):
    from hud.utils.hints import CREDITS_EXHAUSTED, RATE_LIMIT_HIT

    transport = httpx.MockTransport(lambda request: httpx.Response(status, json={"detail": "stop"}))
    with pytest.raises(HudRequestError) as error:
        if asynchronous:
            async with httpx.AsyncClient(transport=transport) as client:
                await make_request("GET", "https://test/data", api_key="key", client=client)
        else:
            with httpx.Client(transport=transport) as client:
                make_request_sync("GET", "https://test/data", api_key="key", client=client)
    assert error.value.hints == [CREDITS_EXHAUSTED if status == 402 else RATE_LIMIT_HIT]


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

    with patch("hud.utils.requests.asyncio.sleep", AsyncMock()) as mock_retry:
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
async def test_make_request_auto_client_creation():
    """Test automatic client creation when not provided."""
    with patch("hud.utils.requests._create_default_async_client") as mock_create_client:
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
    with patch("hud.utils.requests._create_default_sync_client") as mock_create_client:
        mock_client = Mock()
        mock_client.request.return_value = httpx.Response(
            200, json={"result": "success"}, request=httpx.Request("GET", "https://api.test.com")
        )
        mock_client.close = Mock()
        mock_create_client.return_value = mock_client

        result = make_request_sync("GET", "https://api.test.com/data", api_key="test-key")

        assert result == {"result": "success"}
        mock_client.close.assert_called_once()


def test_make_request_sync_empty_204() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(204, request=request)

    sync_client = httpx.Client(transport=httpx.MockTransport(handler))
    assert (
        make_request_sync(
            "DELETE",
            "https://api.test.com/data",
            api_key="test-key",
            client=sync_client,
        )
        == {}
    )
