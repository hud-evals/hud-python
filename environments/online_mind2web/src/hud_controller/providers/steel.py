"""Steel provider implementation."""

import os
import logging
from typing import Optional, Dict, Any
import httpx

from .base import BrowserProvider

logger = logging.getLogger(__name__)


class SteelProvider(BrowserProvider):
    """Steel provider for remote browser control.

    Steel is an open-source browser API that provides cloud browser instances with features like:
    - CAPTCHA solving
    - Proxy support
    - Session management
    - Context persistence (cookies, local storage)
    - Live view and recordings
    - Anti-detection measures
    - Up to 24-hour sessions

    API Documentation: https://docs.steel.dev/api-reference
    """

    def __init__(self, config: Dict[str, Any] | None = None):
        super().__init__(config)
        self.api_key = config.get("api_key") if config else os.getenv("STEEL_API_KEY")
        self.base_url = (
            config.get("base_url", "https://api.steel.dev") if config else "https://api.steel.dev"
        )
        self._session_data: Dict[str, Any] | None = None

        if not self.api_key:
            raise ValueError("Steel API key not provided")

    async def launch(self, **kwargs) -> str:
        """Launch a Steel browser instance.

        Args:
            **kwargs: Launch options including:
                - sessionTimeout: Session timeout in milliseconds (max 24 hours)
                - proxy: Proxy configuration (user:pass@host:port)
                - blockAds: Block ads (default: False)
                - stealth: Enable stealth mode
                - isSelenium: Create Selenium-compatible session
                - loadExtensions: Load Chrome extensions
                - solveCaptchas: Enable CAPTCHA solving
                - context: Saved context (cookies, localStorage)

        Returns:
            CDP WebSocket URL for connecting to the browser
        """
        # Build request payload using Steel's format
        request_data = {
            "sessionId": kwargs.get("sessionId", ""),
            "userAgent": kwargs.get("userAgent", ""),
            "useProxy": kwargs.get("useProxy", False),
            "proxyUrl": kwargs.get("proxyUrl", ""),
            "blockAds": kwargs.get("blockAds", False),
            "solveCaptcha": kwargs.get("solveCaptcha", False),
            "timeout": kwargs.get("timeout", 1800000),  # 30 minutes default
            "concurrency": kwargs.get("concurrency", 1),
            "isSelenium": kwargs.get("isSelenium", False),
            "region": kwargs.get("region", "lax"),
        }

        # Add dimensions if specified
        if "dimensions" in kwargs:
            request_data["dimensions"] = kwargs["dimensions"]
        else:
            request_data["dimensions"] = {"width": 1920, "height": 1080}

        # Add session context if provided
        if "sessionContext" in kwargs:
            request_data["sessionContext"] = kwargs["sessionContext"]

        # Add stealth config
        if "stealthConfig" in kwargs:
            request_data["stealthConfig"] = kwargs["stealthConfig"]

        # Make API request
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/v1/sessions",
                json=request_data,
                headers={"Content-Type": "application/json", "Steel-Api-Key": str(self.api_key)},
                timeout=30.0,
            )
            response.raise_for_status()

        # Extract session data
        data = response.json()
        self._session_data = data
        self._instance_id = data.get("id")

        if not self._instance_id:
            raise Exception("Failed to get session ID from Steel response")

        # Get WebSocket URL - Steel returns wsUrl
        self._cdp_url = data.get("wsUrl")
        if not self._cdp_url:
            # Fallback to constructing URL if not provided
            self._cdp_url = f"wss://api.steel.dev/sessions/{self._instance_id}"

        self._is_running = True

        logger.info(f"Launched Steel session: {self._instance_id}")
        logger.info(f"CDP URL: {self._cdp_url}")

        # Store additional URLs for reference
        self._debug_url = data.get("debugUrl")
        self._live_view_url = data.get("liveViewUrl")

        return self._cdp_url

    def close(self) -> None:
        """Terminate the Steel session."""
        if not self._instance_id:
            return

        try:
            with httpx.Client() as client:
                response = client.delete(
                    f"{self.base_url}/v1/sessions/{self._instance_id}",
                    headers={
                        "Content-Type": "application/json",
                        "Steel-Api-Key": str(self.api_key),
                    },
                    timeout=30.0,
                )
                # Steel may return 404 if session already ended
                if response.status_code != 404:
                    response.raise_for_status()

            logger.info(f"Terminated Steel session: {self._instance_id}")
        except Exception as e:
            logger.error(f"Error terminating session {self._instance_id}: {e}")
        finally:
            self._is_running = False
            self._cdp_url = None
            self._instance_id = None

    async def get_status(self) -> Dict[str, Any]:
        """Get status including Steel-specific info."""
        status = await super().get_status()

        # Add Steel-specific status
        if self._instance_id and self._is_running:
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.get(
                        f"{self.base_url}/v1/sessions/{self._instance_id}",
                        headers={
                            "steel_api_key": str(self.api_key),
                            "Content-Type": "application/json",
                        },
                        timeout=10.0,
                    )
                    if response.status_code == 200:
                        session_data = response.json()
                        status["session_data"] = session_data
                        status["status"] = session_data.get("status", "UNKNOWN")
                        status["context"] = session_data.get("context")
            except Exception as e:
                logger.warning(f"Failed to get session status: {e}")

        return status

    def get_debug_url(self) -> Optional[str]:
        """Get the debug URL for the Steel instance."""
        return self._debug_url if hasattr(self, "_debug_url") else None

    def get_live_view_url(self) -> Optional[str]:
        """Get the live view URL for the Steel instance."""
        return self._live_view_url if hasattr(self, "_live_view_url") else None

    async def save_context(self) -> Optional[Dict[str, Any]]:
        """Save the current browser context (cookies, localStorage).

        Returns:
            Context data that can be passed to launch() to restore state
        """
        if not self._instance_id:
            return None

        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.base_url}/v1/sessions/{self._instance_id}/context",
                    headers={
                        "Content-Type": "application/json",
                        "Steel-Api-Key": str(self.api_key),
                    },
                    timeout=10.0,
                )
                response.raise_for_status()
                return response.json()
        except Exception as e:
            logger.error(f"Failed to save context: {e}")
            return None
