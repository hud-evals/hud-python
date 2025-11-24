"""
Context server for remote browser environment that persists state across hot-reloads.

Run this as a separate process to maintain browser session state during development:
    python -m hud_controller.context
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, Any, Optional
from hud.server.context import run_context_server

logger = logging.getLogger(__name__)


class RemoteBrowserContext:
    """Context that holds remote browser state across reloads."""

    def __init__(self):
        """Initialize the remote browser context."""
        self.browser_provider = None
        self.is_initialized = False
        self.provider_config: Optional[Dict[str, Any]] = None
        self.launch_options: Optional[Dict[str, Any]] = None
        self._startup_complete = False
        self.playwright_tool = None  # Store the playwright tool
        self._telemetry: Optional[Dict[str, Any]] = None  # Store full telemetry data

        logger.info("[RemoteBrowserContext] Created new remote browser context")

    def startup(self):
        """One-time startup when context server starts."""
        if self._startup_complete:
            logger.info("[RemoteBrowserContext] Startup already complete, skipping")
            return

        logger.info("[RemoteBrowserContext] Performing one-time startup")
        self._startup_complete = True

    # === Proxy-friendly methods for multiprocessing.Manager ===
    # Note: These are needed because direct attribute access doesn't always
    # work correctly through the multiprocessing proxy

    def get_browser_provider(self):
        """Get the browser provider instance."""
        return self.browser_provider

    def set_browser_provider(self, provider) -> None:
        """Set the browser provider instance."""
        self.browser_provider = provider
        if provider:
            self.provider_name = provider.__class__.__name__.replace("Provider", "").lower()
            logger.info(f"[RemoteBrowserContext] Set browser provider: {self.provider_name}")

    def get_cdp_url(self) -> Optional[str]:
        """Get the CDP URL from telemetry."""
        return self._telemetry.get("cdp_url") if self._telemetry else None

    def get_is_initialized(self) -> bool:
        """Check if environment is initialized."""
        return self.is_initialized

    def set_initialized(self, value: bool) -> None:
        """Set initialization status."""
        self.is_initialized = value
        logger.info(f"[RemoteBrowserContext] Initialization status: {value}")

    def get_provider_config(self) -> Optional[Dict[str, Any]]:
        """Get provider configuration."""
        return self.provider_config

    def set_provider_config(self, config: Dict[str, Any]) -> None:
        """Set provider configuration."""
        self.provider_config = config
        logger.info(f"[RemoteBrowserContext] Set provider config")

    def get_launch_options(self) -> Optional[Dict[str, Any]]:
        """Get launch options."""
        return self.launch_options

    def set_launch_options(self, options: Dict[str, Any]) -> None:
        """Set launch options."""
        self.launch_options = options
        logger.info(f"[RemoteBrowserContext] Set launch options")

    def get_playwright_tool(self):
        """Get the playwright tool instance."""
        return self.playwright_tool

    def set_playwright_tool(self, tool) -> None:
        """Set the playwright tool instance."""
        self.playwright_tool = tool
        logger.info(f"[RemoteBrowserContext] Set playwright tool")

    def set_telemetry(self, telemetry: Dict[str, Any]) -> None:
        """Set the full telemetry data."""
        self._telemetry = telemetry
        logger.info(f"[RemoteBrowserContext] Set telemetry: {telemetry}")

    def get_state_summary(self) -> Dict[str, Any]:
        """Get a summary of the current state."""
        return {
            "is_initialized": self.is_initialized,
            "startup_complete": self._startup_complete,
            "provider_name": self._telemetry.get("provider") if self._telemetry else None,
            "has_cdp_url": self.get_cdp_url() is not None,
            "has_browser_provider": self.browser_provider is not None,
            "has_playwright_tool": self.playwright_tool is not None,
        }

    def get_telemetry(self) -> Dict[str, Any]:
        """Get telemetry data from the browser provider."""
        # If we have stored telemetry, return it
        if self._telemetry:
            return self._telemetry

        # Otherwise return basic telemetry data
        return {
            "provider": "unknown",
            "status": "not_initialized",
            "live_url": None,
            "cdp_url": None,
            "instance_id": None,
            "timestamp": datetime.now().isoformat(),
        }


if __name__ == "__main__":
    # Run the context server with RemoteBrowserContext
    context = RemoteBrowserContext()
    context.startup()

    # Log initial state
    logger.info(f"[Context] Starting remote browser context server")
    logger.info(f"[Context] Initial state: {context.get_state_summary()}")

    # Run the context server
    asyncio.run(run_context_server(context, "/tmp/hud_remote_browser_ctx.sock"))
