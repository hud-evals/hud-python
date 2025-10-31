"""Base class for browser providers."""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class BrowserProvider(ABC):
    """Abstract base class for browser providers.

    Each provider manages the lifecycle of a remote browser instance
    and provides access to its Chrome DevTools Protocol (CDP) endpoint.
    """

    def __init__(self, config: Dict[str, Any] | None = None):
        """Initialize the provider with optional configuration.

        Args:
            config: Provider-specific configuration
        """
        self.config = config or {}
        self._cdp_url: Optional[str] = None
        self._instance_id: Optional[str] = None
        self._is_running = False

    @abstractmethod
    async def launch(self, **kwargs) -> str:
        """Launch a browser instance and return its CDP URL.

        Args:
            **kwargs: Provider-specific launch options

        Returns:
            CDP URL (e.g., "ws://localhost:9222/devtools/browser/xxx")

        Raises:
            Exception: If launch fails
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """Close the browser instance and cleanup resources.

        Raises:
            Exception: If close fails
        """
        pass

    async def get_status(self) -> Dict[str, Any]:
        """Get the current status of the browser instance.

        Returns:
            Dictionary with status information including:
            - is_running: Whether the browser is running
            - cdp_url: The CDP URL if available
            - instance_id: Provider-specific instance identifier
            - additional provider-specific status info
        """
        return {
            "is_running": self._is_running,
            "cdp_url": self._cdp_url,
            "instance_id": self._instance_id,
            "provider": self.__class__.__name__,
        }

    def get_live_view_url(self) -> Optional[str]:
        """Get the live view URL for the browser instance.

        Returns:
            Live view URL if available, None otherwise
        """
        # Default implementation returns None
        # Providers should override this method
        return None

    @property
    def cdp_url(self) -> Optional[str]:
        """Get the CDP URL of the running browser instance."""
        return self._cdp_url

    @property
    def is_running(self) -> bool:
        """Check if the browser instance is running."""
        return self._is_running

    async def __aenter__(self):
        """Async context manager entry - launch the browser."""
        await self.launch()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit - close the browser."""
        self.close()
