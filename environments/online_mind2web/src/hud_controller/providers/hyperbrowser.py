"""HyperBrowser provider implementation."""

import os
import logging
from typing import Optional, Dict, Any, List
import httpx

from .base import BrowserProvider

logger = logging.getLogger(__name__)


class HyperBrowserProvider(BrowserProvider):
    """HyperBrowser provider for remote browser control.

    HyperBrowser provides cloud browser instances with advanced features like:
    - Stealth mode with fingerprinting
    - Advanced proxy configuration (country/state/city)
    - Profile management and persistence
    - Web recording (video and screenshots)
    - CAPTCHA solving
    - Ad blocking and tracker blocking
    - Browser fingerprinting customization

    API Documentation: https://docs.hyperbrowser.ai/reference/api-reference/sessions
    """

    def __init__(self, config: Dict[str, Any] | None = None):
        super().__init__(config)
        self.api_key = config.get("api_key") if config else os.getenv("HYPERBROWSER_API_KEY")
        self.base_url = (
            config.get("base_url", "https://api.hyperbrowser.ai")
            if config
            else "https://api.hyperbrowser.ai"
        )
        self._session_data: Dict[str, Any] | None = None

        if not self.api_key:
            raise ValueError("HyperBrowser API key not provided")

    async def launch(self, **kwargs) -> str:
        """Launch a HyperBrowser instance.

        Args:
            **kwargs: Launch options including:
                - useStealth: Enable stealth mode (default: False)
                - useProxy: Enable proxy (default: False)
                - proxyCountry: Country code for proxy
                - proxyState: State code for US proxies
                - proxyCity: City name for proxy
                - proxyServer: Custom proxy server
                - proxyServerUsername: Proxy username
                - proxyServerPassword: Proxy password
                - solveCaptchas: Enable CAPTCHA solving
                - adblock: Enable ad blocking
                - trackers: Enable tracker blocking
                - annoyances: Enable annoyances blocking
                - enableWebRecording: Enable session recording
                - enableVideoWebRecording: Enable video recording
                - profile: Profile configuration dict with id and persistChanges
                - acceptCookies: Auto-accept cookies
                - extensionIds: List of extension IDs to load
                - browserArgs: Additional browser arguments
                - timeoutMinutes: Session timeout (1-720 minutes)

        Returns:
            CDP URL for connecting to the browser
        """
        # Build request payload with defaults
        request_data = {
            "useStealth": kwargs.get("useStealth", False),
            "useProxy": kwargs.get("useProxy", False),
        }

        # Add proxy configuration
        if request_data["useProxy"]:
            if "proxyServer" in kwargs:
                request_data["proxyServer"] = kwargs["proxyServer"]
                request_data["proxyServerUsername"] = kwargs.get("proxyServerUsername")
                request_data["proxyServerPassword"] = kwargs.get("proxyServerPassword")
            else:
                # Use HyperBrowser's residential proxy
                request_data["proxyCountry"] = kwargs.get("proxyCountry", "US")
                if "proxyState" in kwargs:
                    request_data["proxyState"] = kwargs["proxyState"]
                if "proxyCity" in kwargs:
                    request_data["proxyCity"] = kwargs["proxyCity"]

        # Add optional features
        if "solveCaptchas" in kwargs:
            request_data["solveCaptchas"] = kwargs["solveCaptchas"]

        if "adblock" in kwargs:
            request_data["adblock"] = kwargs["adblock"]

        if "trackers" in kwargs:
            request_data["trackers"] = kwargs["trackers"]

        if "annoyances" in kwargs:
            request_data["annoyances"] = kwargs["annoyances"]

        # Recording options
        if "enableWebRecording" in kwargs:
            request_data["enableWebRecording"] = kwargs["enableWebRecording"]

        if "enableVideoWebRecording" in kwargs:
            request_data["enableVideoWebRecording"] = kwargs["enableVideoWebRecording"]

        # Profile management
        if "profile" in kwargs:
            request_data["profile"] = kwargs["profile"]

        if "acceptCookies" in kwargs:
            request_data["acceptCookies"] = kwargs["acceptCookies"]

        # Extensions and browser args
        if "extensionIds" in kwargs:
            request_data["extensionIds"] = kwargs["extensionIds"]

        if "browserArgs" in kwargs:
            request_data["browserArgs"] = kwargs["browserArgs"]

        # Timeout
        if "timeoutMinutes" in kwargs:
            request_data["timeoutMinutes"] = kwargs["timeoutMinutes"]

        # Make API request
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/api/session",
                json=request_data,
                headers={"x-api-key": str(self.api_key), "Content-Type": "application/json"},
                timeout=30.0,
            )
            response.raise_for_status()

        # Extract session data
        data = response.json()
        self._session_data = data
        self._instance_id = data.get("id")

        if not self._instance_id:
            raise Exception("Failed to get session ID from HyperBrowser response")

        # Get WebSocket endpoint - HyperBrowser returns wsEndpoint
        self._cdp_url = data.get("wsEndpoint")
        if not self._cdp_url:
            raise Exception("Failed to get WebSocket endpoint from HyperBrowser response")

        self._is_running = True

        logger.info(f"Launched HyperBrowser session: {self._instance_id}")
        logger.info(f"CDP URL: {self._cdp_url}")

        # Store additional URLs for reference
        self._session_url = data.get("sessionUrl")
        self._live_url = data.get("liveUrl")
        self._token = data.get("token")

        return self._cdp_url

    def close(self) -> None:
        """Terminate the HyperBrowser session."""
        if not self._instance_id:
            return

        try:
            with httpx.Client() as client:
                response = client.put(
                    f"{self.base_url}/api/session/{self._instance_id}/stop",
                    headers={"x-api-key": str(self.api_key), "Content-Type": "application/json"},
                    timeout=30.0,
                )
                response.raise_for_status()

            logger.info(f"Terminated HyperBrowser session: {self._instance_id}")
        except Exception as e:
            logger.error(f"Error terminating session {self._instance_id}: {e}")
        finally:
            self._is_running = False
            self._cdp_url = None
            self._instance_id = None

    async def get_status(self) -> Dict[str, Any]:
        """Get status including HyperBrowser-specific info."""
        status = await super().get_status()

        # Add HyperBrowser-specific status
        if self._instance_id and self._is_running:
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.get(
                        f"{self.base_url}/api/session/{self._instance_id}",
                        headers={
                            "x-api-key": str(self.api_key),
                            "Content-Type": "application/json",
                        },
                        timeout=10.0,
                    )
                    if response.status_code == 200:
                        session_data = response.json()
                        status["session_data"] = session_data
                        status["status"] = session_data.get("status", "UNKNOWN")
                        status["start_time"] = session_data.get("startTime")
                        status["end_time"] = session_data.get("endTime")
            except Exception as e:
                logger.warning(f"Failed to get session status: {e}")

        return status

    def get_live_view_url(self) -> Optional[str]:
        """Get the live view URL for the HyperBrowser instance."""
        return self._live_url if hasattr(self, "_live_url") else None

    def get_session_url(self) -> Optional[str]:
        """Get the session URL for the HyperBrowser instance."""
        return self._session_url if hasattr(self, "_session_url") else None

    async def get_sessions_list(
        self, page: int = 1, status: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get list of sessions.

        Args:
            page: Page number (default: 1)
            status: Filter by status ("active", "closed", "error")

        Returns:
            Dict with sessions list and pagination info
        """
        params = {"page": page}
        if status:
            params["status"] = status

        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.base_url}/api/sessions",
                params=params,
                headers={"x-api-key": str(self.api_key), "Content-Type": "application/json"},
                timeout=10.0,
            )
            response.raise_for_status()

        return response.json()
