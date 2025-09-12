"""Shared retry utilities for MCP client operations."""

from __future__ import annotations

import asyncio
import logging
from functools import wraps
from typing import TYPE_CHECKING, Any, TypeVar

if TYPE_CHECKING:
    from collections.abc import Callable

from httpx import HTTPStatusError
from mcp.shared.exceptions import McpError

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Default retry configuration matching requests.py
DEFAULT_MAX_RETRIES = 4
DEFAULT_RETRY_DELAY = 2.0
DEFAULT_RETRY_STATUS_CODES = {502, 503, 504}
DEFAULT_BACKOFF_FACTOR = 2.0


def is_retryable_error(error: Exception, retry_status_codes: set[int]) -> bool:
    """
    Check if an error is retryable based on status codes.

    Args:
        error: The exception to check
        retry_status_codes: Set of HTTP status codes to retry on

    Returns:
        True if the error is retryable, False otherwise
    """
    # Check for HTTP status errors with retryable status codes
    if isinstance(error, HTTPStatusError):
        return error.response.status_code in retry_status_codes

    # Check for MCP errors that might wrap HTTP errors
    if isinstance(error, McpError):
        error_msg = str(error).lower()
        # Check for common gateway error patterns in the message
        for code in retry_status_codes:
            if str(code) in error_msg:
                return True
        # Check for gateway error keywords
        if any(
            keyword in error_msg
            for keyword in ["bad gateway", "service unavailable", "gateway timeout"]
        ):
            return True

    # Check for generic errors with status codes in the message
    error_msg = str(error)
    for code in retry_status_codes:
        if f"{code}" in error_msg or f"status {code}" in error_msg.lower():
            return True

    return False


async def retry_with_backoff(
    func: Callable[..., Any],
    *args: Any,
    max_retries: int = DEFAULT_MAX_RETRIES,
    retry_delay: float = DEFAULT_RETRY_DELAY,
    retry_status_codes: set[int] | None = None,
    backoff_factor: float = DEFAULT_BACKOFF_FACTOR,
    operation_name: str | None = None,
    **kwargs: Any,
) -> Any:
    """
    Execute an async function with retry logic and exponential backoff.

    This matches the retry behavior in requests.py but can be applied
    to any async function, particularly MCP client operations.

    Args:
        func: The async function to retry
        *args: Positional arguments for the function
        max_retries: Maximum number of retry attempts
        retry_delay: Initial delay between retries in seconds
        retry_status_codes: HTTP status codes to retry on
        backoff_factor: Multiplier for exponential backoff
        operation_name: Name of the operation for logging
        **kwargs: Keyword arguments for the function

    Returns:
        The result of the function call

    Raises:
        The last exception if all retries are exhausted
    """
    if retry_status_codes is None:
        retry_status_codes = DEFAULT_RETRY_STATUS_CODES

    operation = operation_name or func.__name__
    last_error = None

    for attempt in range(max_retries + 1):
        try:
            result = await func(*args, **kwargs)
            return result
        except Exception as e:
            last_error = e

            # Check if this is a retryable error
            if not is_retryable_error(e, retry_status_codes):
                # Not retryable, raise immediately
                raise

            # Don't retry if we've exhausted attempts
            if attempt >= max_retries:
                logger.debug(
                    "Operation '%s' failed after %d retries: %s",
                    operation,
                    max_retries,
                    e,
                )
                raise

            # Calculate backoff delay (exponential backoff)
            delay = retry_delay * (backoff_factor**attempt)

            logger.warning(
                "Operation '%s' failed with retryable error, "
                "retrying in %.2f seconds (attempt %d/%d): %s",
                operation,
                delay,
                attempt + 1,
                max_retries,
                e,
            )

            await asyncio.sleep(delay)

    # This should never be reached, but just in case
    if last_error:
        raise last_error
    raise RuntimeError(f"Unexpected retry loop exit for operation '{operation}'")


def with_retry(
    max_retries: int = DEFAULT_MAX_RETRIES,
    retry_delay: float = DEFAULT_RETRY_DELAY,
    retry_status_codes: set[int] | None = None,
    backoff_factor: float = DEFAULT_BACKOFF_FACTOR,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    Decorator to add retry logic to async methods.

    Usage:
        @with_retry(max_retries=3)
        async def my_method(self, ...):
            ...

    Args:
        max_retries: Maximum number of retry attempts
        retry_delay: Initial delay between retries
        retry_status_codes: HTTP status codes to retry on
        backoff_factor: Multiplier for exponential backoff

    Returns:
        Decorated function with retry logic
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            return await retry_with_backoff(
                func,
                *args,
                max_retries=max_retries,
                retry_delay=retry_delay,
                retry_status_codes=retry_status_codes,
                backoff_factor=backoff_factor,
                operation_name=func.__name__,
                **kwargs,
            )

        return wrapper

    return decorator
