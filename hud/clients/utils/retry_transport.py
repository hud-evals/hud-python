"""Custom HTTPX transport with retry logic for HTTP errors."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

import httpx
from httpx._transports.default import AsyncHTTPTransport

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from httpx._models import Request, Response


class RetryTransport(AsyncHTTPTransport):
    """
    Custom HTTPX transport that retries on specific HTTP status codes.

    This transport wraps the standard AsyncHTTPTransport and adds
    retry logic with exponential backoff for gateway errors (502, 503, 504).
    """

    def __init__(
        self,
        *args: Any,
        max_retries: int = 3,
        retry_status_codes: set[int] | None = None,
        retry_delay: float = 1.0,
        backoff_factor: float = 2.0,
        **kwargs: Any,
    ) -> None:
        """
        Initialize retry transport.

        Args:
            max_retries: Maximum number of retry attempts
            retry_status_codes: HTTP status codes to retry (default: 502, 503, 504)
            retry_delay: Initial delay between retries in seconds
            backoff_factor: Multiplier for exponential backoff
            *args, **kwargs: Passed to AsyncHTTPTransport
        """
        super().__init__(*args, **kwargs)
        self.max_retries = max_retries
        self.retry_status_codes = retry_status_codes or {502, 503, 504}
        self.retry_delay = retry_delay
        self.backoff_factor = backoff_factor

    async def handle_async_request(self, request: Request) -> Response:
        """
        Handle request with retry logic.

        Retries the request if it fails with a retryable status code,
        using exponential backoff between attempts.
        """
        last_exception = None

        for attempt in range(self.max_retries + 1):
            try:
                response = await super().handle_async_request(request)

                # Check if we should retry based on status code
                if response.status_code in self.retry_status_codes and attempt < self.max_retries:
                    delay = self.retry_delay * (self.backoff_factor**attempt)
                    logger.warning(
                        "Got %d from %s, retrying in %.1fs (attempt %d/%d)",
                        response.status_code,
                        request.url,
                        delay,
                        attempt + 1,
                        self.max_retries,
                    )
                    # Important: Close the response to free resources
                    await response.aclose()
                    await asyncio.sleep(delay)
                    continue

                return response

            except (httpx.ConnectError, httpx.TimeoutException) as e:
                last_exception = e
                if attempt < self.max_retries:
                    delay = self.retry_delay * (self.backoff_factor**attempt)
                    logger.warning(
                        "%s for %s, retrying in %.1fs (attempt %d/%d)",
                        type(e).__name__,
                        request.url,
                        delay,
                        attempt + 1,
                        self.max_retries,
                    )
                    await asyncio.sleep(delay)
                    continue
                raise

        # If we get here, we've exhausted retries
        if last_exception:
            raise last_exception
        else:
            # This shouldn't happen, but just in case
            raise httpx.HTTPStatusError(
                "Max retries exceeded",
                request=request,
                response=response,
            )


def create_retry_httpx_client(
    headers: dict[str, str] | None = None,
    timeout: httpx.Timeout | None = None,
    auth: httpx.Auth | None = None,
    max_retries: int = 3,
    retry_status_codes: set[int] | None = None,
) -> httpx.AsyncClient:
    """
    Create an HTTPX AsyncClient with HTTP error retry support.

    This factory creates an HTTPX client with a custom transport that
    retries on specific HTTP status codes (502, 503, 504 by default).

    Args:
        headers: Optional headers to include with all requests
        timeout: Request timeout (defaults to 600s)
        auth: Optional authentication handler
        max_retries: Maximum retry attempts (default: 3)
        retry_status_codes: Status codes to retry (default: {502, 503, 504})

    Returns:
        Configured httpx.AsyncClient with retry transport
    """
    if timeout is None:
        timeout = httpx.Timeout(600.0)  # 10 minutes

    # Use higher connection limits for concurrent operations
    # These match HUD server's configuration for consistency
    limits = httpx.Limits(
        max_connections=1000,
        max_keepalive_connections=1000,
        keepalive_expiry=20.0,
    )

    # Create our custom retry transport
    transport = RetryTransport(
        max_retries=max_retries,
        retry_status_codes=retry_status_codes,
        # Connection-level retries (in addition to HTTP retries)
        retries=3,
        limits=limits,
    )

    return httpx.AsyncClient(
        transport=transport,
        headers=headers,
        timeout=timeout,
        auth=auth,
        follow_redirects=True,
        limits=limits,
    )
