"""PlaywrightTool with memory/history tracking for remote browser environment."""

import logging
from typing import Any, Dict, List, Optional, Literal
from datetime import datetime
from pydantic import Field
from hud.tools.playwright import PlaywrightTool
from mcp.types import ContentBlock, ImageContent, TextContent

logger = logging.getLogger(__name__)


class PlaywrightToolWithMemory(PlaywrightTool):
    """Extended PlaywrightTool that tracks navigation and action history.

    This tool extends the base PlaywrightTool to add:
    - Navigation history tracking
    - Action history tracking
    - Selector history for debugging
    """

    def __init__(self, context: Any = None, cdp_url: str | None = None) -> None:
        """Initialize with history tracking capabilities.

        Args:
            context: Optional context (not used, for compatibility)
            cdp_url: Chrome DevTools Protocol URL for connecting to browser
        """
        # Initialize base tool with CDP URL as context
        super().__init__(cdp_url=cdp_url)

        # Initialize history tracking
        self.navigation_history: List[Dict[str, Any]] = []
        self.action_history: List[Dict[str, Any]] = []
        self.selector_history: List[str] = []

    async def _ensure_browser(self) -> None:
        """Ensure browser is launched and setup event listeners."""
        await super()._ensure_browser()

        # Setup event listeners for tracking
        if self.page:
            self._setup_event_listeners()

    def _setup_event_listeners(self) -> None:
        """Setup event listeners to track navigation history."""
        if not self.page:
            return

        try:
            # Track frame navigations (includes main frame)
            def on_frame_navigated(frame):
                if self.page is None:
                    return
                if frame == self.page.main_frame:
                    self.navigation_history.append(
                        {"url": frame.url, "timestamp": datetime.now().isoformat()}
                    )
                    logger.debug(f"Navigation tracked: {frame.url}")

            self.page.on("framenavigated", on_frame_navigated)

            # Track page loads
            def on_load():
                if self.page:
                    logger.debug(f"Page loaded: {self.page.url}")

            self.page.on("load", on_load)

        except Exception as e:
            logger.warning(f"Failed to setup event listeners: {e}")

    def _record_action(self, action_type: str, details: Dict[str, Any], result: Any = None) -> None:
        """Record an action in the history.

        Args:
            action_type: Type of action performed
            details: Details about the action
            result: Result of the action
        """
        action_record = {
            "type": action_type,
            "timestamp": datetime.now().isoformat(),
            "details": details,
            "result": result,
        }
        self.action_history.append(action_record)
        logger.debug(f"Action recorded: {action_type} - {details}")

    async def navigate(
        self,
        url: str = Field(..., description="URL to navigate to"),
        wait_for_load_state: Literal["load", "domcontentloaded", "networkidle"] = Field(
            "networkidle", description="Wait condition after navigation"
        ),
    ) -> dict:
        """Navigate to a URL with history tracking.

        Args:
            url: URL to navigate to
            wait_for_load_state: State to wait for after navigation

        Returns:
            Navigation result dictionary
        """
        # Record the navigation action
        self._record_action("navigate", {"url": url, "wait_for_load_state": wait_for_load_state})

        # Perform the navigation using parent class
        result = await super().navigate(url, wait_for_load_state)

        # Update action record with result
        if self.action_history:
            self.action_history[-1]["result"] = result

        return result

    async def click(
        self,
        selector: str = Field(..., description="CSS selector to click"),
        button: Literal["left", "right", "middle"] = Field("left", description="Mouse button"),
        count: int = Field(1, description="Number of clicks"),
        wait_for_navigation: bool = Field(False, description="Wait for navigation after click"),
    ) -> dict:
        """Click an element with history tracking.

        Args:
            selector: CSS selector to click
            button: Mouse button to use
            count: Number of clicks
            wait_for_navigation: Whether to wait for navigation

        Returns:
            Click result dictionary
        """
        # Track selector
        self.selector_history.append(selector)

        # Record the action
        self._record_action(
            "click",
            {
                "selector": selector,
                "button": button,
                "count": count,
                "wait_for_navigation": wait_for_navigation,
            },
        )

        # Perform the click using parent class
        result = await super().click(selector, button, count, wait_for_navigation)

        # Update action record with result
        if self.action_history:
            self.action_history[-1]["result"] = result

        return result

    def get_history_summary(self) -> Dict[str, Any]:
        """Get a summary of the browsing history.

        Returns:
            Dictionary with history statistics
        """
        return {
            "navigation_count": len(self.navigation_history),
            "action_count": len(self.action_history),
            "unique_selectors": len(set(self.selector_history)),
            "last_navigation": self.navigation_history[-1] if self.navigation_history else None,
            "last_action": self.action_history[-1] if self.action_history else None,
        }
