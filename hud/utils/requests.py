"""
HTTP request utilities for the HUD API.
"""

from __future__ import annotations

import asyncio
import logging
import random
import ssl
import time
import warnings
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Any

import httpx

from hud.utils.exceptions import (
    HudAuthenticationError,
    HudDeprecationWarning,
    HudNetworkError,
    HudRequestError,
    HudTimeoutError,
)

# Set up logger
logger = logging.getLogger("hud.http")
logger.setLevel(logging.INFO)


# Long running requests can take up to 10 minutes.
_DEFAULT_TIMEOUT = 600.0
_DEFAULT_LIMITS = httpx.Limits(
    max_connections=1000,
    max_keepalive_connections=1000,
    keepalive_expiry=10.0,
)


# Gateway and upstream failures are retried, with or without a Retry-After header.
_RETRY_STATUS_CODES = frozenset({502, 503, 504})
# A 429 is retried only when the server says how long to wait and the wait is short:
# the server then knows the exact time its limiter reopens, and a caller that
# sleeps that long (plus jitter) is admitted instead of being refused again.
_RATE_LIMIT_STATUS = 429
# Longest server-requested wait the client sleeps through. A longer wait is not
# slept: a 429 is raised to the caller as before, a 5xx falls back to this cap.
_MAX_RETRY_AFTER_SECONDS = 30.0
# Retry-After is a minimum wait under the client's own backoff, not a replacement for it.
# Extra wait added on top of a server-requested delay, as a fraction of it, so
# callers refused together do not all return in the same instant. Never negative:
# returning before the requested time would only be refused again.
_RETRY_AFTER_JITTER = 0.25


def _rand() -> float:
    """Uniform [0, 1) source for jitter; a seam for tests."""
    return random.random()  # noqa: S311


def _parse_retry_after(value: str | None) -> float | None:
    """Seconds named by a Retry-After header (delta-seconds or HTTP-date), else None."""
    if not value:
        return None
    value = value.strip()
    try:
        seconds = float(value)
    except ValueError:
        try:
            when = parsedate_to_datetime(value)
        except (TypeError, ValueError):
            return None
        if when.tzinfo is None:
            when = when.replace(tzinfo=UTC)
        seconds = (when - datetime.now(UTC)).total_seconds()
    if seconds != seconds or seconds < 0:  # NaN or in the past
        return None
    return seconds


def _retry_delay(
    attempt: int, max_retries: int, retry_delay: float, url: str, error_msg: str
) -> float:
    """Calculate and log the shared retry backoff (exponential, with equal jitter)."""
    retry_time = jittered_backoff(retry_delay * (2 ** (attempt - 1)))
    logger.debug(
        "%s from %s, retrying in %.2f seconds (attempt %d/%d)",
        error_msg,
        url,
        retry_time,
        attempt,
        max_retries,
    )
    return retry_time


def is_retryable_response(response: httpx.Response) -> bool:
    """Whether a response status is worth retrying.

    502, 503 and 504 always are. A 429 is only when it names a short wait.
    """
    status = response.status_code
    if status == _RATE_LIMIT_STATUS:
        retry_after = _parse_retry_after(response.headers.get("Retry-After"))
        return retry_after is not None and retry_after <= _MAX_RETRY_AFTER_SECONDS
    return status in _RETRY_STATUS_CODES


def retry_after_wait(response: httpx.Response) -> float | None:
    """Seconds the server asked the caller to wait, or None without a readable Retry-After.

    The wait is capped at ``_MAX_RETRY_AFTER_SECONDS`` and only ever lengthened by jitter.
    Callers use it as a minimum under their own backoff, never in place of it: a short
    Retry-After on every refusal would otherwise make a retry loop knock once a second.
    """
    retry_after = _parse_retry_after(response.headers.get("Retry-After"))
    if retry_after is None:
        return None
    wait = min(retry_after, _MAX_RETRY_AFTER_SECONDS)
    return wait + wait * _RETRY_AFTER_JITTER * _rand()


def jittered_backoff(base: float) -> float:
    """Equal jitter: a wait drawn from the upper half of ``base``."""
    return base * (0.5 + 0.5 * _rand())


def _response_retry_delay(
    response: httpx.Response, attempt: int, max_retries: int, retry_delay: float, url: str
) -> float | None:
    """Seconds to wait before retrying this response, or None when it is not retried.

    The wait is the exponential backoff, or the server's Retry-After (capped at
    ``_MAX_RETRY_AFTER_SECONDS``, with a little added jitter) when that is longer.
    """
    if not is_retryable_response(response):
        return None
    backoff = _retry_delay(
        attempt, max_retries, retry_delay, url, f"Received status {response.status_code}"
    )
    asked = retry_after_wait(response)
    return backoff if asked is None else max(asked, backoff)


def _create_default_async_client() -> httpx.AsyncClient:
    """Create a default httpx AsyncClient with standard configuration."""
    return httpx.AsyncClient(
        timeout=_DEFAULT_TIMEOUT,
        limits=_DEFAULT_LIMITS,
    )


def _create_default_sync_client() -> httpx.Client:
    """Create a default httpx Client with standard configuration."""
    return httpx.Client(
        timeout=_DEFAULT_TIMEOUT,
        limits=_DEFAULT_LIMITS,
    )


_announced_deprecations: set[tuple[str, str, str]] = set()


def _warn_if_deprecated(method: str, response: httpx.Response) -> None:
    """Warn once per process for each deprecation notice the HUD API sends."""
    if "Deprecation" not in response.headers:
        return
    notice = (
        response.headers["Deprecation"],
        response.headers.get("Sunset", ""),
        response.headers.get("Link", ""),
    )
    if notice in _announced_deprecations:
        return
    _announced_deprecations.add(notice)
    message = f"{method} {response.url.path} is deprecated by the HUD API"
    if sunset := response.headers.get("Sunset"):
        message += f" and stops being served on {sunset}"
    if successor := response.links.get("successor-version"):
        message += f"; its replacement is {successor['url']}"
    if docs := response.links.get("deprecation"):
        message += f" (see {docs['url']})"
    warnings.warn(
        f"{message}. If a hud command printed this, upgrade hud.",
        HudDeprecationWarning,
        stacklevel=3,
    )


def _response_payload(response: httpx.Response) -> Any:
    """Decode JSON, treating 204 / empty bodies as an empty object."""
    if response.status_code == 204 or not response.content:
        return {}
    return response.json()


async def make_request(
    method: str,
    url: str,
    json: Any | None = None,
    api_key: str | None = None,
    max_retries: int = 4,
    retry_delay: float = 2.0,
    client: httpx.AsyncClient | None = None,
) -> dict[str, Any]:
    """
    Make an asynchronous HTTP request to the HUD API.

    Args:
        method: HTTP method (GET, POST, etc.)
        url: Full URL for the request
        json: Optional JSON serializable data
        api_key: API key for authentication
        max_retries: Maximum number of retries
        retry_delay: Base delay between retries; it doubles each attempt and is jittered.
            A Retry-After header on a 429, 502, 503 or 504 response sets a minimum wait
            (capped at 30 seconds, with up to 25% added jitter) when it is longer. A 429
            is retried only when it carries a Retry-After of at most 30 seconds.
        *,
        client: Optional custom httpx.AsyncClient

    Returns:
        dict: JSON response from the server

    Raises:
        HudAuthenticationError: If API key is missing or invalid.
        HudRequestError: If the request fails with a non-retryable status code.
        HudNetworkError: If there are network-related issues.
        HudTimeoutError: If the request times out.
    """
    if not api_key:
        raise HudAuthenticationError("API key is required but not provided")

    headers = {"Authorization": f"Bearer {api_key}"}
    attempt = 0
    should_close_client = False

    if client is None:
        client = _create_default_async_client()
        should_close_client = True

    try:
        while attempt <= max_retries:
            attempt += 1

            try:
                response = await client.request(method=method, url=url, json=json, headers=headers)

                # Check if we got a retriable status code
                delay = (
                    _response_retry_delay(response, attempt, max_retries, retry_delay, url)
                    if attempt <= max_retries
                    else None
                )
                if delay is not None:
                    await asyncio.sleep(delay)
                    continue

                _warn_if_deprecated(method, response)
                response.raise_for_status()
                result = _response_payload(response)
                return result
            except httpx.TimeoutException as e:
                raise HudTimeoutError(f"Request timed out: {e!s}") from None
            except httpx.HTTPStatusError as e:
                raise HudRequestError.from_httpx_error(e) from None
            except (httpx.RequestError, ssl.SSLError) as e:
                kind = "SSL error" if isinstance(e, ssl.SSLError) else "Network error"
                if attempt > max_retries:
                    raise HudNetworkError(f"{kind}: {e}") from None
                await asyncio.sleep(
                    _retry_delay(attempt, max_retries, retry_delay, url, f"{kind}: {e}")
                )
            except Exception as e:
                raise HudRequestError(f"Unexpected error: {e!s}") from None
        raise HudRequestError(f"Request failed after {max_retries} retries with unknown error")
    finally:
        if should_close_client:
            await client.aclose()


def make_request_sync(
    method: str,
    url: str,
    json: Any | None = None,
    api_key: str | None = None,
    max_retries: int = 4,
    retry_delay: float = 2.0,
    *,
    client: httpx.Client | None = None,
) -> dict[str, Any]:
    """
    Make a synchronous HTTP request to the HUD API.

    Args:
        method: HTTP method (GET, POST, etc.)
        url: Full URL for the request
        json: Optional JSON serializable data
        api_key: API key for authentication
        max_retries: Maximum number of retries
        retry_delay: Base delay between retries; it doubles each attempt and is jittered.
            A Retry-After header on a 429, 502, 503 or 504 response sets a minimum wait
            (capped at 30 seconds, with up to 25% added jitter) when it is longer. A 429
            is retried only when it carries a Retry-After of at most 30 seconds.
        client: Optional custom httpx.Client

    Returns:
        dict: JSON response from the server

    Raises:
        HudAuthenticationError: If API key is missing or invalid.
        HudRequestError: If the request fails with a non-retryable status code.
        HudNetworkError: If there are network-related issues.
        HudTimeoutError: If the request times out.
    """
    if not api_key:
        raise HudAuthenticationError("API key is required but not provided")

    headers = {"Authorization": f"Bearer {api_key}"}
    attempt = 0
    should_close_client = False

    if client is None:
        client = _create_default_sync_client()
        should_close_client = True

    try:
        while attempt <= max_retries:
            attempt += 1

            try:
                response = client.request(method=method, url=url, json=json, headers=headers)

                # Check if we got a retriable status code
                delay = (
                    _response_retry_delay(response, attempt, max_retries, retry_delay, url)
                    if attempt <= max_retries
                    else None
                )
                if delay is not None:
                    time.sleep(delay)
                    continue

                _warn_if_deprecated(method, response)
                response.raise_for_status()
                result = _response_payload(response)
                return result
            except httpx.TimeoutException as e:
                raise HudTimeoutError(f"Request timed out: {e!s}") from None
            except httpx.HTTPStatusError as e:
                raise HudRequestError.from_httpx_error(e) from None
            except (httpx.RequestError, ssl.SSLError) as e:
                kind = "SSL error" if isinstance(e, ssl.SSLError) else "Network error"
                if attempt > max_retries:
                    raise HudNetworkError(f"{kind}: {e}") from None
                time.sleep(_retry_delay(attempt, max_retries, retry_delay, url, f"{kind}: {e}"))
            except Exception as e:
                raise HudRequestError(f"Unexpected error: {e!s}") from None
        raise HudRequestError(f"Request failed after {max_retries} retries with unknown error")
    finally:
        if should_close_client:
            client.close()
