"""Retry wrapper for MCP-use HTTP transport.

This module provides a transport-level retry mechanism for MCP-use,
similar to the approach used in FastMCP.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any, TypeVar

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)

T = TypeVar("T")


def create_retry_session(
    max_retries: int = 3,
    retry_status_codes: tuple[int, ...] = (502, 503, 504),
    retry_delay: float = 1.0,
    backoff_factor: float = 2.0,
) -> requests.Session:
    """
    Create a requests session with retry logic.

    Args:
        max_retries: Maximum number of retry attempts
        retry_status_codes: HTTP status codes to retry
        retry_delay: Initial delay between retries in seconds
        backoff_factor: Multiplier for exponential backoff

    Returns:
        Configured requests.Session with retry logic
    """
    session = requests.Session()

    # Configure retry strategy
    retry = Retry(
        total=max_retries,
        backoff_factor=backoff_factor,
        status_forcelist=list(retry_status_codes),
        # Allow retries on all methods
        allowed_methods=["HEAD", "GET", "OPTIONS", "POST", "PUT", "DELETE", "PATCH"],
        # Respect Retry-After header if present
        respect_retry_after_header=True,
    )

    # Create adapter with retry strategy
    adapter = HTTPAdapter(max_retries=retry)

    # Mount adapter for both HTTP and HTTPS
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    logger.debug(
        "Created retry session with max_retries=%d, status_codes=%s, backoff_factor=%.1f",
        max_retries,
        retry_status_codes,
        backoff_factor,
    )

    return session


def patch_mcp_session_http_client(session: Any) -> None:
    """
    Patch an MCP-use session to use HTTP retry logic.

    This function attempts to replace the HTTP client used by an MCP session
    with one that has retry logic enabled.

    Args:
        session: MCP-use session to patch
    """
    try:
        # Check if session has a connector with an HTTP client
        if hasattr(session, "connector"):
            connector = session.connector

            # For HTTP connectors, patch the underlying HTTP client
            if hasattr(connector, "_connection_manager"):
                manager = connector._connection_manager

                # If it's using requests, replace the session
                if hasattr(manager, "_session") or hasattr(manager, "session"):
                    retry_session = create_retry_session()

                    # Try different attribute names
                    if hasattr(manager, "_session"):
                        manager._session = retry_session
                        logger.debug("Patched connection manager's _session with retry logic")
                    elif hasattr(manager, "session"):
                        manager.session = retry_session
                        logger.debug("Patched connection manager's session with retry logic")

            # Also check for client_session (async variant)
            if hasattr(connector, "client_session") and connector.client_session:
                client = connector.client_session

                # Wrap the async HTTP methods with retry logic
                if hasattr(client, "_send_request"):
                    original_send = client._send_request
                    client._send_request = create_async_retry_wrapper(original_send)
                    logger.debug("Wrapped client_session._send_request with retry logic")

    except Exception as e:
        logger.warning("Could not patch MCP session with retry logic: %s", e)


def create_async_retry_wrapper(
    func: Callable[..., Any],
    max_retries: int = 3,
    retry_status_codes: tuple[int, ...] = (502, 503, 504),
    retry_delay: float = 1.0,
    backoff_factor: float = 2.0,
) -> Callable[..., Any]:
    """
    Create an async wrapper that adds retry logic to a function.

    Args:
        func: The async function to wrap
        max_retries: Maximum number of retry attempts
        retry_status_codes: HTTP status codes to retry
        retry_delay: Initial delay between retries
        backoff_factor: Multiplier for exponential backoff

    Returns:
        Wrapped function with retry logic
    """

    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        last_exception = None
        delay = retry_delay

        for attempt in range(max_retries + 1):
            try:
                result = await func(*args, **kwargs)

                # Check if result has a status code that should trigger retry
                if (
                    hasattr(result, "status_code")
                    and result.status_code in retry_status_codes
                    and attempt < max_retries
                ):
                    logger.warning(
                        "HTTP %d error (attempt %d/%d), retrying in %.1fs",
                        result.status_code,
                        attempt + 1,
                        max_retries + 1,
                        delay,
                    )
                    await asyncio.sleep(delay)
                    delay *= backoff_factor
                    continue

                return result

            except Exception as e:
                # Check if it's an HTTP error that should be retried
                error_str = str(e)
                should_retry = any(str(code) in error_str for code in retry_status_codes)

                if should_retry and attempt < max_retries:
                    logger.warning(
                        "Error '%s' (attempt %d/%d), retrying in %.1fs",
                        e,
                        attempt + 1,
                        max_retries + 1,
                        delay,
                    )
                    await asyncio.sleep(delay)
                    delay *= backoff_factor
                    last_exception = e
                else:
                    raise

        # If we exhausted retries, raise the last exception
        if last_exception:
            raise last_exception

    return wrapper


def patch_all_sessions(sessions: dict[str, Any]) -> None:
    """
    Apply retry logic to all MCP sessions.

    Args:
        sessions: Dictionary of session name to session object
    """
    for name, session in sessions.items():
        logger.debug("Patching session '%s' with retry logic", name)
        patch_mcp_session_http_client(session)
