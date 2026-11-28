"""HUD SDK exceptions."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from typing import Self

    import httpx2


class HudException(Exception):
    """Base class for HUD API and credential errors."""

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
    """A failed HUD API request."""

    def __init__(
        self,
        message: str,
        status_code: int | None = None,
        response_text: str | None = None,
        response_json: dict[str, Any] | None = None,
        response_headers: dict[str, str] | None = None,
        *,
        url: str | None = None,
    ) -> None:
        self.status_code = status_code
        self.url = url
        self.response_text = response_text
        self.response_headers = response_headers
        super().__init__(message, response_json)

    def __str__(self) -> str:
        parts = [self.message]

        if self.url:
            parts.append(f"URL: {self.url}")

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
    def from_httpx_error(cls, error: httpx2.HTTPStatusError, context: str = "") -> Self:
        """Build from an httpx status error, the message prefixed with ``context``."""
        response = error.response
        status_code = response.status_code
        response_text = response.text
        response_headers = dict(response.headers)

        response_json = None
        try:
            response_json = response.json()
            detail = response_json.get("detail")
            if detail:
                message = f"Request failed: {detail}"
            else:
                message = f"Request failed with status {status_code}"
                if len(response_json) <= 5:
                    message += f" - JSON response: {response_json}"
        except Exception:
            message = f"Request failed with status {status_code}"

        if context:
            message = f"{context}: {message}"
        return cls(
            message=message,
            status_code=status_code,
            response_text=response_text,
            response_json=response_json,
            response_headers=response_headers,
            url=str(response.url),
        )


class HudReplyError(HudRequestError):
    """The HUD API answered a request with a body this client cannot read."""


class HudAuthenticationError(HudException):
    """Missing or invalid HUD API key."""


class HudTimeoutError(HudException):
    """Request timed out."""


class HudNetworkError(HudException):
    """Network connection issue."""


class HudDeprecationWarning(FutureWarning):
    """The HUD API deprecated a route this client calls."""
