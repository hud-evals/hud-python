"""AnchorBrowser provider implementation."""

import os
import logging
from typing import Optional, Dict, Any
import httpx

from .base import BrowserProvider

logger = logging.getLogger(__name__)


class AnchorBrowserProvider(BrowserProvider):
    """AnchorBrowser provider for remote browser control.

    AnchorBrowser provides cloud-based browser instances with features like:
    - Proxy support
    - CAPTCHA solving
    - Ad blocking
    - Popup blocking
    """

    def __init__(self, config: Dict[str, Any] | None = None):
        super().__init__(config)
        self.api_key = config.get("api_key") if config else os.getenv("ANCHOR_API_KEY")
        self.base_url = (
            config.get("base_url", "https://api.anchorbrowser.io")
            if config
            else "https://api.anchorbrowser.io"
        )
        self._session_data: Dict[str, Any] | None = None  # Initialize session data storage

        if not self.api_key:
            raise ValueError("AnchorBrowser API key not provided")

    async def launch(self, **kwargs) -> str:
        """Launch an AnchorBrowser instance.

        Args:
            **kwargs: Launch options including:
                - max_duration: Maximum session duration in seconds (default: 120)
                - idle_timeout: Idle timeout in seconds (default: 30)
                - proxy: Proxy configuration dict with:
                    - type: "custom" or "anchor_residential"
                    - server: Proxy server address (for custom)
                    - username: Proxy username (for custom)
                    - password: Proxy password (for custom)
                    - country_code: Country code (for anchor_residential)
                - headless: Whether to run headless
                - viewport: Viewport size
                - captcha_solver: Enable CAPTCHA solving
                - adblock: Enable ad blocking
                - popup_blocker: Enable popup blocking

        Returns:
            CDP URL for connecting to the browser
        """
        # Build request payload
        request_data = {
            "session": {
                "timeout": {
                    "max_duration": kwargs.get("max_duration", 120),
                    "idle_timeout": kwargs.get("idle_timeout", 30),
                },
            },
            "browser": {
                "adblock": {"active": True},
                "popup_blocker": {"active": True},
                "captcha_solver": {"active": True},
            },
        }

        # Add proxy configuration if provided
        if "proxy" in kwargs:
            request_data["session"]["proxy"] = kwargs["proxy"]
        else:
            # Default to anchor residential proxy
            request_data["session"]["proxy"] = {
                "type": "anchor_residential",
                "active": True,
                "country_code": "us",
            }

        # Make API request
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/v1/sessions",
                json=request_data,
                headers={"anchor-api-key": str(self.api_key), "Content-Type": "application/json"},
                timeout=30.0,
            )
            response.raise_for_status()

        # Extract session data
        data = response.json()
        session_data = data.get("data", {})
        self._instance_id = session_data.get("id")
        self._session_data = session_data  # Store full session data
        self._cdp_url = session_data.get("cdp_url")

        if not self._instance_id:
            raise Exception("Failed to get session ID from AnchorBrowser response")
        if not self._cdp_url:
            raise Exception("Failed to get CDP URL from AnchorBrowser response")

        self._is_running = True

        logger.info(f"Launched AnchorBrowser session: {self._instance_id}")
        return self._cdp_url

    async def close(self) -> None:
        """Terminate the AnchorBrowser session."""
        if not self._instance_id:
            return

        try:
            async with httpx.AsyncClient() as client:
                response = await client.delete(
                    f"{self.base_url}/v1/sessions/{self._instance_id}",
                    headers={
                        "anchor-api-key": str(self.api_key),
                        "Content-Type": "application/json",
                    },
                    timeout=30.0,
                )
                response.raise_for_status()

            logger.info(f"Terminated AnchorBrowser session: {self._instance_id}")
        except Exception as e:
            logger.error(f"Error terminating session {self._instance_id}: {e}")
        finally:
            self._is_running = False
            self._cdp_url = None
            self._instance_id = None

    async def get_status(self) -> Dict[str, Any]:
        """Get status including AnchorBrowser-specific info."""
        status = await super().get_status()

        # Add AnchorBrowser-specific status
        if self._instance_id and self._is_running:
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.get(
                        f"{self.base_url}/v1/sessions/{self._instance_id}/status",
                        headers={
                            "anchor-api-key": str(self.api_key),
                            "Content-Type": "application/json",
                        },
                        timeout=10.0,
                    )
                    if response.status_code == 200:
                        session_status = response.json().get("data", {})
                        status["session_status"] = session_status
            except Exception as e:
                logger.warning(f"Failed to get session status: {e}")

        return status

    def get_live_view_url(self) -> Optional[str]:
        """Get the live view URL for the AnchorBrowser instance."""
        if self._session_data:
            return self._session_data.get("live_view_url")
        return None
