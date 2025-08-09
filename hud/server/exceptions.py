from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from typing import Self

    import httpx

logger = logging.getLogger(__name__)


class HudException(Exception):
    """Base exception class for all HUD SDK errors.

    This is the parent class for all exceptions raised by the HUD SDK.
    Consumers should be able to catch this exception to handle any HUD-related error.
    """


class HudRequestError(Exception):
    """Any request to the HUD API can raise this exception."""

    def __init__(
        self,
        message: str,
        status_code: int | None = None,
        response_text: str | None = None,
        response_json: dict[str, Any] | None = None,
        response_headers: dict[str, str] | None = None,
    ) -> None:
        self.message = message
        self.status_code = status_code
        self.response_text = response_text
        self.response_json = response_json
        self.response_headers = response_headers
        super().__init__(message)

    def __str__(self) -> str:
        parts = [self.message]

        if self.status_code:
            parts.append(f"Status: {self.status_code}")

        if self.response_text:
            parts.append(f"Response Text: {self.response_text}")

        if self.response_json:
            parts.append(f"Response JSON: {self.response_json}")

        if self.response_headers:
            parts.append(f"Headers: {self.response_headers}")

        return " | ".join(parts)

    @classmethod
    def from_httpx_error(cls, error: httpx.HTTPStatusError, context: str = "") -> Self:
        """Create a RequestError from an HTTPx error response.

        Args:
            error: The HTTPx error response.
            context: Additional context to include in the error message.

        Returns:
            A RequestError instance.
        """
        response = error.response
        status_code = response.status_code
        response_text = response.text
        response_headers = dict(response.headers)

        # Try to get detailed error info from JSON if available
        response_json = None
        try:
            response_json = response.json()
            detail = response_json.get("detail")
            if detail:
                message = f"Request failed: {detail}"
            else:
                # If no detail field but we have JSON, include a summary
                message = f"Request failed with status {status_code}"
                if len(response_json) <= 5:  # If it's a small object, include it in the message
                    message += f" - JSON response: {response_json}"
        except Exception:
            # Fallback to simple message if JSON parsing fails
            message = f"Request failed with status {status_code}"

        # Add context if provided
        if context:
            message = f"{context}: {message}"

        # Log the error details
        logger.error(
            "HTTP error from HUD SDK: %s | URL: %s | Status: %s | Response: %s%s",
            message,
            response.url,
            status_code,
            response_text[:500],
            "..." if len(response_text) > 500 else "",
        )
        return cls(
            message=message,
            status_code=status_code,
            response_text=response_text,
            response_json=response_json,
            response_headers=response_headers,
        )


class HudResponseError(HudException):
    """Raised when an API response is invalid or missing required data.

    This exception is raised when we receive a successful response (e.g. 200)
    but the response data is invalid, missing required fields, or otherwise
    cannot be processed.

    Attributes:
        message: A human-readable error message
        response_json: The invalid response data
    """

    def __init__(
        self,
        message: str,
        response_json: dict[str, Any] | None = None,
    ) -> None:
        self.message = message
        self.response_json = response_json
        super().__init__(message)

    def __str__(self) -> str:
        parts = [self.message]
        if self.response_json:
            parts.append(f"Response: {self.response_json}")
        return " | ".join(parts)


class HudAuthenticationError(HudException):
    """Raised when authentication with the HUD API fails.

    This exception is raised when an API key is missing, invalid, or
    has insufficient permissions for the requested operation.
    """


class HudRateLimitError(HudException):
    """Raised when the rate limit for the HUD API is exceeded.

    This exception is raised when too many requests are made in a
    short period of time.
    """


class HudTimeoutError(HudException):
    """Raised when a request to the HUD API times out.

    This exception is raised when a request takes longer than the
    configured timeout period.
    """


class HudNetworkError(HudException):
    """Raised when there is a network-related error.

    This exception is raised when there are issues with the network
    connection, DNS resolution, or other network-related problems.
    """


class GymMakeException(HudException):
    """Raised when environment creation or setup fails, includes context data."""

    def __init__(self, message: str, data: dict[str, Any]) -> None:
        super().__init__(message)
        self.data = data

    def __str__(self) -> str:
        base = super().__str__()
        return f"{base} | Data: {self.data}"
