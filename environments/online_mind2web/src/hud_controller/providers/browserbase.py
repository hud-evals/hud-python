"""BrowserBase provider implementation."""

import os
import logging
from typing import Optional, Dict, Any
import httpx

from .base import BrowserProvider

logger = logging.getLogger(__name__)


class BrowserBaseProvider(BrowserProvider):
    """BrowserBase provider for remote browser control.

    BrowserBase provides cloud browser instances with features like:
    - Multiple regions support
    - Context persistence
    - Live view URLs
    - Session recordings
    - Proxy support

    API Documentation: https://docs.browserbase.com/reference/api/create-a-session
    """

    def __init__(self, config: Dict[str, Any] | None = None):
        super().__init__(config)
        self.api_key = config.get("api_key") if config else os.getenv("BROWSERBASE_API_KEY")
        self.base_url = (
            config.get("base_url", "https://api.browserbase.com")
            if config
            else "https://api.browserbase.com"
        )
        self.project_id = (
            config.get("project_id") if config else os.getenv("BROWSERBASE_PROJECT_ID")
        )
        self._session_data: Dict[str, Any] | None = None

        if not self.api_key:
            raise ValueError("BrowserBase API key not provided")

    async def launch(self, **kwargs) -> str:
        """Launch a BrowserBase instance.

        Args:
            **kwargs: Launch options including:
                - projectId: Project ID (required if not set in config)
                - region: Browser region (e.g., "us-west-2")
                - keepAlive: Keep session alive after disconnect
                - contextId: Reuse browser context
                - browserSettings: Additional browser settings
                - proxies: Enable proxy support

        Returns:
            CDP URL for connecting to the browser
        """
        # Build request payload
        request_data = {"projectId": kwargs.get("projectId", self.project_id)}

        # Add optional parameters
        if "region" in kwargs:
            request_data["region"] = kwargs["region"]

        if "keepAlive" in kwargs:
            request_data["keepAlive"] = kwargs["keepAlive"]

        if "contextId" in kwargs:
            request_data["contextId"] = kwargs["contextId"]

        if "browserSettings" in kwargs:
            request_data["browserSettings"] = kwargs["browserSettings"]

        if "proxies" in kwargs:
            request_data["proxies"] = kwargs["proxies"]

        # Ensure we have a project ID
        if not request_data.get("projectId"):
            raise ValueError("BrowserBase project ID not provided")

        # Make API request
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/v1/sessions",
                json=request_data,
                headers={"X-BB-API-Key": str(self.api_key), "Content-Type": "application/json"},
                timeout=30.0,
            )
            response.raise_for_status()

        # Extract session data
        data = response.json()
        self._session_data = data
        self._instance_id = data.get("id")

        if not self._instance_id:
            raise Exception("Failed to get session ID from BrowserBase response")

        # Get CDP URL - BrowserBase returns connectUrl directly
        self._cdp_url = data.get("connectUrl")
        if not self._cdp_url:
            raise Exception("Failed to get connect URL from BrowserBase response")

        self._is_running = True

        logger.info(f"Launched BrowserBase session: {self._instance_id}")
        logger.info(f"CDP URL: {self._cdp_url}")

        # Store additional URLs for reference
        self._live_view_url = data.get("liveViewUrl")
        self._selenium_remote_url = data.get("seleniumRemoteUrl")

        return self._cdp_url

    def close(self) -> None:
        """Terminate the BrowserBase session."""
        if not self._instance_id:
            return

        try:
            # BrowserBase sessions automatically close after disconnect unless keepAlive is true
            # We can explicitly update the session to mark it as ended
            with httpx.Client() as client:
                response = client.post(
                    f"{self.base_url}/v1/sessions/{self._instance_id}",
                    json={"status": "COMPLETED"},
                    headers={"X-BB-API-Key": str(self.api_key), "Content-Type": "application/json"},
                    timeout=30.0,
                )
                # BrowserBase may return 404 if session already ended
                if response.status_code != 404:
                    response.raise_for_status()

            logger.info(f"Terminated BrowserBase session: {self._instance_id}")
        except Exception as e:
            logger.error(f"Error terminating session {self._instance_id}: {e}")
        finally:
            self._is_running = False
            self._cdp_url = None
            self._instance_id = None

    async def get_status(self) -> Dict[str, Any]:
        """Get status including BrowserBase-specific info."""
        status = await super().get_status()

        # Add BrowserBase-specific status
        if self._instance_id and self._is_running:
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.get(
                        f"{self.base_url}/v1/sessions/{self._instance_id}",
                        headers={
                            "X-BB-API-Key": str(self.api_key),
                            "Content-Type": "application/json",
                        },
                        timeout=10.0,
                    )
                    if response.status_code == 200:
                        session_data = response.json()
                        status["session_data"] = session_data
                        status["status"] = session_data.get("status", "UNKNOWN")
                        status["region"] = session_data.get("region")
                        status["proxy_bytes"] = session_data.get("proxyBytes")
                        status["cpu_usage"] = session_data.get("avgCpuUsage")
                        status["memory_usage"] = session_data.get("memoryUsage")
            except Exception as e:
                logger.warning(f"Failed to get session status: {e}")

        return status

    def get_live_view_url(self) -> Optional[str]:
        """Get the live view URL for the BrowserBase instance."""
        return self._live_view_url if hasattr(self, "_live_view_url") else None

    def get_selenium_remote_url(self) -> Optional[str]:
        """Get the Selenium remote URL for the BrowserBase instance."""
        return self._selenium_remote_url if hasattr(self, "_selenium_remote_url") else None
