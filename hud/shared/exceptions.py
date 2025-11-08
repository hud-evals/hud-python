"""HUD SDK Exception System.

This module provides intelligent exception handling with automatic error
classification and helpful hints for users.

Key Features:
- Auto-converts generic exceptions to specific HUD exceptions
- Attaches contextual hints based on error type
- Clean chaining syntax: raise HudException() from e

Example:
    try:
        client.call_tool("missing")
    except Exception as e:
        raise HudException() from e  # Becomes HudToolNotFoundError with hints
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import TYPE_CHECKING, Any, ClassVar, TypeVar

if TYPE_CHECKING:
    from typing import Self

    import httpx

from hud.shared.hints import (
    CLIENT_NOT_INITIALIZED,
    ENV_VAR_MISSING,
    HUD_API_KEY_MISSING,
    INVALID_CONFIG,
    MCP_SERVER_ERROR,
    RATE_LIMIT_HIT,
    TOOL_NOT_FOUND,
    Hint,
)

T = TypeVar("T", bound="HudException")

logger = logging.getLogger(__name__)


class HudException(Exception):
    """Base exception class for all HUD SDK errors.

    Usage:
        raise HudException() from e  # Auto-converts to appropriate subclass
        raise HudException("Custom message") from e  # With custom message
    """

    def __new__(cls, message: str = "", *args: Any, **kwargs: Any) -> Any:
        """Auto-convert generic exceptions to specific HUD exceptions when chained."""
        import sys

        # Only intercept for base HudException, not subclasses
        if cls is not HudException:
            return super().__new__(cls)

        # Check if we're in a 'raise...from' context
        exc_type, exc_value, _ = sys.exc_info()
        if exc_type and exc_value:
            # If it's already a HudException, return it as-is
            if isinstance(exc_value, HudException):
                return exc_value
            # Otherwise analyze if it's a regular Exception
            elif isinstance(exc_value, Exception):
                # Try to convert to a specific HudException
                result = cls._analyze_exception(exc_value, message or str(exc_value))
                return result

        # Normal creation
        return super().__new__(cls)

    # Subclasses can override this class attribute
    default_hints: ClassVar[list[Hint]] = []

    def __init__(
        self,
        message: str = "",
        response_json: dict[str, Any] | None = None,
        *,
        hints: list[Hint] | None = None,
    ) -> None:
        # If we already have args set (from _analyze_exception), don't override them
        if not self.args:
            # Pass the message to the base Exception class
            super().__init__(message)
        self.message = message or (self.args[0] if self.args else "")
        self.response_json = response_json
        # If hints not provided, use defaults defined by subclass
        self.hints: list[Hint] = hints if hints is not None else list(self.default_hints)

    def __str__(self) -> str:
        # Get the message from the exception
        # First check if we have args (standard Exception message storage)
        msg = str(self.args[0]) if self.args and self.args[0] else ""

        # Add response JSON if available
        if self.response_json:
            if msg:
                return f"{msg} | Response: {self.response_json}"
            else:
                return f"Response: {self.response_json}"

        return msg

    @classmethod
    def _analyze_exception(cls, e: Exception, message: str = "") -> HudException:
        """Convert generic exceptions to specific HUD exceptions based on content."""
        error_msg = str(e).lower()
        final_msg = message or str(e)

        # Map error patterns to exception types
        patterns = [
            # (condition_func, exception_class)
            (
                lambda: "not initialized" in error_msg or "not connected" in error_msg,
                HudClientError,
            ),
            (
                lambda: "invalid json" in error_msg or "config" in error_msg or "json" in error_msg,
                HudConfigError,
            ),
            (
                lambda: "tool" in error_msg
                and ("not found" in error_msg or "not exist" in error_msg),
                HudToolNotFoundError,
            ),
            (
                lambda: ("api key" in error_msg or "authorization" in error_msg)
                and ("hud" in error_msg or "mcp.hud.ai" in error_msg),
                HudAuthenticationError,
            ),
            (
                lambda: "rate limit" in error_msg or "too many request" in error_msg,
                HudRateLimitError,
            ),
            (lambda: isinstance(e, (TimeoutError | asyncio.TimeoutError)), HudTimeoutError),
            (lambda: isinstance(e, json.JSONDecodeError), HudConfigError),
            (
                lambda: "environment variable" in error_msg and "required" in error_msg,
                HudEnvVarError,
            ),
            (lambda: "event loop" in error_msg and "closed" in error_msg, HudClientError),
            (
                lambda: type(e).__name__ == "McpError",  # Check by name to avoid import issues
                HudMCPError,
            ),
        ]

        # Find first matching pattern
        for condition, exception_class in patterns:
            if condition():
                # Create instance directly using Exception.__new__ to bypass our custom __new__
                instance = Exception.__new__(exception_class)
                # Manually set args before calling __init__ to ensure proper Exception behavior
                instance.args = (final_msg,)
                instance.__init__(final_msg)
                return instance

        # No pattern matched - return base exception instance
        instance = Exception.__new__(HudException)
        instance.args = (final_msg,)
        instance.__init__(final_msg)
        return instance


class HudRequestError(HudException):
    """Any request to the HUD API can raise this exception."""

    def __init__(
        self,
        message: str,
        status_code: int | None = None,
        response_text: str | None = None,
        response_json: dict[str, Any] | None = None,
        response_headers: dict[str, str] | None = None,
        *,
        hints: list[Hint] | None = None,
    ) -> None:
        self.status_code = status_code
        self.response_text = response_text
        self.response_headers = response_headers
        # Compute default hints from status code if none provided
        if hints is None and status_code in (401, 402, 403, 429):
            try:
                from hud.shared.hints import (  # type: ignore
                    CREDITS_EXHAUSTED,
                    HUD_API_KEY_MISSING,
                    PRO_PLAN_REQUIRED,
                    RATE_LIMIT_HIT,
                )

                if status_code == 402:
                    hints = [CREDITS_EXHAUSTED]
                elif status_code == 403:
                    # Default 403 to auth unless the message clearly indicates Pro plan
                    combined_text = (message or "").lower()
                    try:
                        if response_text:
                            combined_text += "\n" + str(response_text).lower()
                    except Exception:  # noqa: S110
                        pass
                    try:
                        if response_json and isinstance(response_json, dict):
                            detail = response_json.get("detail")
                            if isinstance(detail, str):
                                combined_text += "\n" + detail.lower()
                    except Exception:  # noqa: S110
                        pass

                    mentions_pro = (
                        "pro plan" in combined_text
                        or "requires pro" in combined_text
                        or "pro mode" in combined_text
                        or combined_text.strip().startswith("pro ")
                    )

                    hints = [PRO_PLAN_REQUIRED] if mentions_pro else [HUD_API_KEY_MISSING]
                elif status_code == 401:
                    hints = [HUD_API_KEY_MISSING]
                elif status_code == 429:
                    hints = [RATE_LIMIT_HIT]
            except Exception as import_error:
                logger.debug("Failed to attach structured hints: %s", import_error)
        super().__init__(message, response_json, hints=hints)

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
        inst = cls(
            message=message,
            status_code=status_code,
            response_text=response_text,
            response_json=response_json,
            response_headers=response_headers,
        )
        return inst


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
    """Missing or invalid HUD API key."""

    default_hints: ClassVar[list[Hint]] = [HUD_API_KEY_MISSING]


class HudRateLimitError(HudException):
    """Too many requests to the API."""

    default_hints: ClassVar[list[Hint]] = [RATE_LIMIT_HIT]


class HudTimeoutError(HudException):
    """Request timed out."""


class HudNetworkError(HudException):
    """Network connection issue."""


class HudClientError(HudException):
    """MCP client not initialized."""

    default_hints: ClassVar[list[Hint]] = [CLIENT_NOT_INITIALIZED]


class HudConfigError(HudException):
    """Invalid or missing configuration."""

    default_hints: ClassVar[list[Hint]] = [INVALID_CONFIG]


class HudEnvVarError(HudException):
    """Missing required environment variables."""

    default_hints: ClassVar[list[Hint]] = [ENV_VAR_MISSING]


class HudToolNotFoundError(HudException):
    """Requested tool not found."""

    default_hints: ClassVar[list[Hint]] = [TOOL_NOT_FOUND]


class HudMCPError(HudException):
    """MCP protocol or server error."""

    default_hints: ClassVar[list[Hint]] = [MCP_SERVER_ERROR]


class GymMakeException(HudException):
    """Raised when environment creation or setup fails, includes context data."""

    def __init__(self, message: str, data: dict[str, Any]) -> None:
        super().__init__(message)
        self.data = data

    def __str__(self) -> str:
        base = super().__str__()
        return f"{base} | Data: {self.data}"
