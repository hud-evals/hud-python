"""Tests for the HTTP request utilities in the HUD API."""

from __future__ import annotations

import uuid
from http import HTTPStatus
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, Mock, patch

import httpx
import pytest

from hud.utils.exceptions import (
    HudAuthenticationError,
    HudDeprecationWarning,
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
        patch("hud.utils.requests._rand", return_value=1.0),
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


def test_requests_warn_once_per_deprecation_notice():
    successor = f"https://api.test/v2/{uuid.uuid4()}"
    headers = {
        "Deprecation": "@1790985600",
        "Sunset": "Sat, 10 Oct 2026 00:00:00 GMT",
        "Link": f'<{successor}>; rel="successor-version"',
    }
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json={}, headers=headers))
    with httpx.Client(transport=transport) as client, pytest.warns(HudDeprecationWarning) as caught:
        for url in ("https://api.test/v2/old/1", "https://api.test/v2/old/2"):
            make_request_sync("GET", url, api_key="key", client=client)

    assert [str(warning.message) for warning in caught] == [
        "GET /v2/old/1 is deprecated by the HUD API and stops being served on "
        f"Sat, 10 Oct 2026 00:00:00 GMT; its replacement is {successor}. "
        "If a hud command printed this, upgrade hud."
    ]


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


async def _run(asynchronous: bool, handle, *, rand: float = 0.0, **kwargs: Any):
    """Run one request against ``handle``, returning (result or error, sleeps, calls)."""
    calls = 0

    def counting(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return handle(calls)

    transport = httpx.MockTransport(counting)
    outcome: Any
    with (
        patch("hud.utils.requests._rand", return_value=rand),
        patch("hud.utils.requests.time.sleep") as sync_sleep,
        patch("hud.utils.requests.asyncio.sleep", new_callable=AsyncMock) as async_sleep,
    ):
        try:
            if asynchronous:
                async with httpx.AsyncClient(transport=transport) as client:
                    outcome = await make_request(
                        "GET", "https://test/data", api_key="key", client=client, **kwargs
                    )
            else:
                with httpx.Client(transport=transport) as client:
                    outcome = make_request_sync(
                        "GET", "https://test/data", api_key="key", client=client, **kwargs
                    )
        except HudRequestError as error:
            outcome = error
        sleep = async_sleep if asynchronous else sync_sleep
        sleeps = [call.args[0] for call in sleep.call_args_list]
    return outcome, sleeps, calls


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize(("rand", "expected"), [(0.0, [1.0, 2.0]), (1.0, [2.0, 4.0])])
async def test_backoff_without_a_header_is_jittered_between_half_and_full(
    asynchronous, rand, expected
):
    outcome, sleeps, _ = await _run(
        asynchronous,
        lambda n: httpx.Response(503 if n < 3 else 200, json={"ok": True}),
        rand=rand,
    )
    assert outcome == {"ok": True}
    assert sleeps == expected


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("status", [429, 503])
@pytest.mark.parametrize(("rand", "expected"), [(0.0, 3.0), (1.0, 3.75)])
async def test_retry_after_is_honoured_and_jitter_only_adds(asynchronous, status, rand, expected):
    outcome, sleeps, calls = await _run(
        asynchronous,
        lambda n: httpx.Response(
            status if n == 1 else 200,
            headers={"Retry-After": "3"} if n == 1 else {},
            json={"ok": True},
        ),
        rand=rand,
    )
    assert outcome == {"ok": True}
    assert sleeps == [expected]
    assert calls == 2


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_a_429_asking_for_a_long_wait_is_raised_not_slept_through(asynchronous):
    from hud.utils.hints import RATE_LIMIT_HIT

    outcome, sleeps, calls = await _run(
        asynchronous,
        lambda n: httpx.Response(429, headers={"Retry-After": "60"}, json={"detail": "slow"}),
    )
    assert isinstance(outcome, HudRequestError)
    assert outcome.hints == [RATE_LIMIT_HIT]
    assert sleeps == []
    assert calls == 1


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_a_429_that_keeps_coming_stops_after_max_retries(asynchronous):
    outcome, sleeps, calls = await _run(
        asynchronous,
        lambda n: httpx.Response(429, headers={"Retry-After": "1"}, json={"detail": "slow"}),
        max_retries=2,
    )
    assert isinstance(outcome, HudRequestError)
    assert sleeps == [1.0, 2.0]  # Retry-After is a floor under the doubling backoff
    assert calls == 3


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_a_5xx_asking_for_more_than_the_cap_waits_the_cap(asynchronous):
    outcome, sleeps, _ = await _run(
        asynchronous,
        lambda n: httpx.Response(
            503 if n == 1 else 200,
            headers={"Retry-After": "120"} if n == 1 else {},
            json={"ok": True},
        ),
    )
    assert outcome == {"ok": True}
    assert sleeps == [30.0]


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_an_unreadable_retry_after_falls_back_to_the_backoff(asynchronous):
    outcome, sleeps, _ = await _run(
        asynchronous,
        lambda n: httpx.Response(
            503 if n == 1 else 200,
            headers={"Retry-After": "soon"} if n == 1 else {},
            json={"ok": True},
        ),
        rand=1.0,
    )
    assert outcome == {"ok": True}
    assert sleeps == [2.0]


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("5", 5.0),
        (" 2.5 ", 2.5),
        ("0", 0.0),
        ("-1", None),
        ("nan", None),
        ("soon", None),
        ("", None),
        (None, None),
    ],
)
def test_parse_retry_after_seconds(value, expected):
    from hud.utils.requests import _parse_retry_after

    assert _parse_retry_after(value) == expected


def test_parse_retry_after_http_date():
    from datetime import UTC, datetime, timedelta
    from email.utils import format_datetime

    from hud.utils.requests import _parse_retry_after

    future = datetime.now(UTC) + timedelta(seconds=10)
    past = datetime.now(UTC) - timedelta(seconds=10)
    assert 8.0 < _parse_retry_after(format_datetime(future, usegmt=True)) <= 10.0
    assert _parse_retry_after(format_datetime(past, usegmt=True)) is None
