"""
HTTP request utilities for the HUD API.
"""

from __future__ import annotations

import asyncio
import logging
import ssl
import time
from typing import Any

import httpx

from hud.shared.exceptions import (
    HudAuthenticationError,
    HudNetworkError,
    HudRequestError,
    HudTimeoutError,
)
from hud.shared.hints import (
    CREDITS_EXHAUSTED,
    HUD_API_KEY_MISSING,
    RATE_LIMIT_HIT,
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


async def _handle_retry(
    attempt: int, max_retries: int, retry_delay: float, url: str, error_msg: str
) -> None:
    """Helper function to handle retry logic and logging."""
    retry_time = retry_delay * (2 ** (attempt - 1))  # Exponential backoff
    logger.debug(
        "%s from %s, retrying in %.2f seconds (attempt %d/%d)",
        error_msg,
        url,
        retry_time,
        attempt,
        max_retries,
    )
    await asyncio.sleep(retry_time)


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
        retry_delay: Delay between retries
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
        raise HudAuthenticationError(
            "API key is required but not provided",
            hints=[HUD_API_KEY_MISSING],
        )

    headers = {"Authorization": f"Bearer {api_key}"}
    retry_status_codes = [502, 503, 504]
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
                if response.status_code in retry_status_codes and attempt <= max_retries:
                    await _handle_retry(
                        attempt,
                        max_retries,
                        retry_delay,
                        url,
                        f"Received status {response.status_code}",
                    )
                    continue

                response.raise_for_status()
                result = response.json()
                return result
            except httpx.TimeoutException as e:
                raise HudTimeoutError(f"Request timed out: {e!s}") from None
            except httpx.HTTPStatusError as e:
                err = HudRequestError.from_httpx_error(e)
                code = getattr(err, "status_code", None)
                if code == 429 and RATE_LIMIT_HIT not in err.hints:
                    logger.debug("Attaching RATE_LIMIT hint to 429 error")
                    err.hints.append(RATE_LIMIT_HIT)
                elif code == 402 and CREDITS_EXHAUSTED not in err.hints:
                    logger.debug("Attaching CREDITS_EXHAUSTED hint to 402 error")
                    err.hints.append(CREDITS_EXHAUSTED)
                raise err from None
            except httpx.RequestError as e:
                if attempt <= max_retries:
                    await _handle_retry(
                        attempt, max_retries, retry_delay, url, f"Network error: {e}"
                    )
                    continue
                else:
                    raise HudNetworkError(f"Network error: {e!s}") from None
            except ssl.SSLError as e:
                if attempt <= max_retries:
                    await _handle_retry(attempt, max_retries, retry_delay, url, f"SSL error: {e}")
                    continue
                else:
                    raise HudNetworkError(f"SSL error: {e!s}") from None
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
        retry_delay: Delay between retries
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
    retry_status_codes = [502, 503, 504]
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
                if response.status_code in retry_status_codes and attempt <= max_retries:
                    retry_time = retry_delay * (2 ** (attempt - 1))  # Exponential backoff
                    logger.debug(
                        "Received status %d from %s, retrying in %.2f seconds (attempt %d/%d)",
                        response.status_code,
                        url,
                        retry_time,
                        attempt,
                        max_retries,
                    )
                    time.sleep(retry_time)
                    continue

                response.raise_for_status()
                result = response.json()
                return result
            except httpx.TimeoutException as e:
                raise HudTimeoutError(f"Request timed out: {e!s}") from None
            except httpx.HTTPStatusError as e:
                err = HudRequestError.from_httpx_error(e)
                code = getattr(err, "status_code", None)
                if code == 429 and RATE_LIMIT_HIT not in err.hints:
                    logger.debug("Attaching RATE_LIMIT hint to 429 error")
                    err.hints.append(RATE_LIMIT_HIT)
                elif code == 402 and CREDITS_EXHAUSTED not in err.hints:
                    logger.debug("Attaching CREDITS_EXHAUSTED hint to 402 error")
                    err.hints.append(CREDITS_EXHAUSTED)
                raise err from None
            except httpx.RequestError as e:
                if attempt <= max_retries:
                    retry_time = retry_delay * (2 ** (attempt - 1))
                    logger.debug(
                        "Network error %s from %s, retrying in %.2f seconds (attempt %d/%d)",
                        str(e),
                        url,
                        retry_time,
                        attempt,
                        max_retries,
                    )
                    time.sleep(retry_time)
                    continue
                else:
                    raise HudNetworkError(f"Network error: {e!s}") from None
            except ssl.SSLError as e:
                if attempt <= max_retries:
                    retry_time = retry_delay * (2 ** (attempt - 1))  # Exponential backoff
                    logger.debug(
                        "SSL error %s from %s, retrying in %.2f seconds (attempt %d/%d)",
                        str(e),
                        url,
                        retry_time,
                        attempt,
                        max_retries,
                    )
                    time.sleep(retry_time)
                    continue
                else:
                    raise HudNetworkError(f"SSL error: {e!s}") from None
            except Exception as e:
                raise HudRequestError(f"Unexpected error: {e!s}") from None
        raise HudRequestError(f"Request failed after {max_retries} retries with unknown error")
    finally:
        if should_close_client:
            client.close()
