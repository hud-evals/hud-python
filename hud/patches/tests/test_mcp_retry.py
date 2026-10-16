"""Tests for the MCP transport retry policy: Retry-After, jitter, and the deadline."""

from __future__ import annotations

import asyncio
import ssl
import time
from unittest.mock import patch

import httpx
import pytest

from hud.patches.mcp_patches import (
    _is_retryable_transport_error,
    _retry_transport_request,
    _transport_retry_wait,
)


def _status_error(status: int, retry_after: str | None = None) -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "https://test/mcp")
    headers = {"Retry-After": retry_after} if retry_after is not None else {}
    response = httpx.Response(status, headers=headers, request=request)
    return httpx.HTTPStatusError(f"status {status}", request=request, response=response)


@pytest.mark.parametrize(
    "error",
    [
        httpx.ConnectError("refused"),
        httpx.ReadTimeout("slow"),
        ssl.SSLError("handshake"),
        _status_error(502),
        _status_error(503),
        _status_error(504),
        _status_error(503, "3"),
        _status_error(429, "2"),
    ],
)
def test_transient_transport_errors_are_retried(error):
    assert _is_retryable_transport_error(error)


@pytest.mark.parametrize(
    "error",
    [
        ValueError("bad payload"),
        _status_error(400),
        _status_error(401),
        _status_error(429),  # no Retry-After: not retried, as for make_request
        _status_error(429, "60"),  # a wait longer than the cap is raised, not slept through
    ],
)
def test_other_errors_are_not_retried(error):
    assert not _is_retryable_transport_error(error)


@pytest.mark.parametrize(("rand", "expected"), [(0.0, 3.0), (1.0, 3.75)])
def test_retry_after_longer_than_the_backoff_is_the_wait_and_jitter_only_lengthens_it(
    rand, expected
):
    with patch("hud.utils.requests._rand", return_value=rand):
        assert _transport_retry_wait(_status_error(503, "3"), backoff=0.5) == expected


@pytest.mark.parametrize(("rand", "expected"), [(0.0, 4.0), (1.0, 8.0)])
def test_a_short_retry_after_never_shortens_the_backoff(rand, expected):
    with patch("hud.utils.requests._rand", return_value=rand):
        assert _transport_retry_wait(_status_error(503, "1"), backoff=8.0) == expected


@pytest.mark.parametrize(("rand", "expected"), [(0.0, 0.25), (1.0, 0.5)])
def test_without_retry_after_the_backoff_is_jittered_between_half_and_full(rand, expected):
    with patch("hud.utils.requests._rand", return_value=rand):
        assert _transport_retry_wait(httpx.ConnectError("refused"), backoff=0.5) == expected
        assert _transport_retry_wait(_status_error(503), backoff=0.5) == expected


class _Attempts:
    """An attempt callable that fails with the given errors, then succeeds."""

    def __init__(self, *errors: BaseException) -> None:
        self.errors = list(errors)
        self.calls = 0

    async def __call__(self) -> None:
        self.calls += 1
        if self.errors:
            raise self.errors.pop(0)


async def _run(attempts: _Attempts, *, remaining: float = 600.0, rand: float = 1.0):
    sleeps: list[float] = []
    errors: list[Exception] = []

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    async def send_error(exc: Exception) -> None:
        errors.append(exc)

    with (
        patch("hud.utils.requests._rand", return_value=rand),
        patch("hud.patches.mcp_patches.asyncio.sleep", side_effect=fake_sleep),
    ):
        await _retry_transport_request(
            attempts,
            deadline=time.monotonic() + remaining,
            global_timeout=remaining,
            send_error_response=send_error,
        )
    return sleeps, errors


async def test_transient_errors_are_retried_with_a_doubling_jittered_backoff():
    attempts = _Attempts(httpx.ConnectError("a"), httpx.ConnectError("b"))

    sleeps, errors = await _run(attempts)

    assert attempts.calls == 3
    assert sleeps == [0.5, 1.0]
    assert errors == []


async def test_a_server_requested_wait_longer_than_the_backoff_is_used():
    attempts = _Attempts(_status_error(503, "2"))

    sleeps, errors = await _run(attempts, rand=0.0)

    assert attempts.calls == 2
    assert sleeps == [2.0]
    assert errors == []


async def test_a_non_retryable_error_is_reported_at_once():
    failure = ValueError("bad payload")
    attempts = _Attempts(failure)

    sleeps, errors = await _run(attempts)

    assert attempts.calls == 1
    assert sleeps == []
    assert errors == [failure]


async def test_past_the_deadline_the_error_is_reported_without_another_retry():
    failure = httpx.ConnectError("down")
    attempts = _Attempts(failure)

    sleeps, errors = await _run(attempts, remaining=-1.0)

    assert attempts.calls == 1
    assert sleeps == []
    assert errors == [failure]


async def test_a_wait_never_runs_past_the_deadline():
    attempts = _Attempts(_status_error(503, "30"))

    sleeps, _ = await _run(attempts, remaining=1.0)

    assert len(sleeps) == 1
    assert sleeps[0] <= 1.0


async def test_cancellation_propagates_and_is_not_retried():
    attempts = _Attempts(asyncio.CancelledError())

    with pytest.raises(asyncio.CancelledError):
        await _run(attempts)

    assert attempts.calls == 1
