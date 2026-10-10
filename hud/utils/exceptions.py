"""HUD SDK exceptions: a small typed hierarchy rooted at :class:`HudException`."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from typing import Self

    import httpx

logger = logging.getLogger(__name__)


class HudException(Exception):
    """Base exception class for all HUD SDK errors."""

    def __init__(self, message: str = "", response_json: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.response_json = response_json

    def __str__(self) -> str:
        if self.response_json:
            prefix = f"{self.message} | " if self.message else ""
            return f"{prefix}Response: {self.response_json}"
        return self.message


class HudRequestError(HudException):
    """Any request to the HUD API can raise this exception."""

    def __init__(
        self,
        message: str,
        status_code: int | None = None,
        response_text: str | None = None,
        response_json: dict[str, Any] | None = None,
        response_headers: dict[str, str] | None = None,
    ) -> None:
        self.status_code = status_code
        self.response_text = response_text
        self.response_headers = response_headers
        super().__init__(message, response_json)

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


class HudAuthenticationError(HudException):
    """Missing or invalid HUD API key."""


class HudRateLimitError(HudException):
    """Too many requests to the API."""


class HudTimeoutError(HudException):
    """Request timed out."""


class HudNetworkError(HudException):
    """Network connection issue."""


class HudClientError(HudException):
    """MCP client not initialized."""


class HudConfigError(HudException):
    """Invalid or missing configuration."""


class HudEnvVarError(HudException):
    """Missing required environment variables."""


class HudToolNotFoundError(HudException):
    """Requested tool not found."""


class HudMCPError(HudException):
    """MCP protocol or server error."""


class HudDeprecationWarning(FutureWarning):
    """The HUD API deprecated a route this client calls."""
