"""PlaywrightTool with memory/history for deep research environment."""

import logging
from datetime import datetime
from typing import Any, Dict, List, Literal

from pydantic import Field

from hud.tools.playwright import PlaywrightTool

logger = logging.getLogger(__name__)


class PlaywrightToolWithMemory(PlaywrightTool):
    """Extended PlaywrightTool that tracks navigation and actions."""

    def __init__(self, context: Any = None, cdp_url: str | None = None) -> None:
        super().__init__(cdp_url=cdp_url)
        self.navigation_history: List[Dict[str, Any]] = []
        self.action_history: List[Dict[str, Any]] = []

    async def _ensure_browser(self) -> None:
        await super()._ensure_browser()
        if self.page:
            try:
                def on_frame_navigated(frame):
                    try:
                        if self.page and frame == self.page.main_frame:
                            self.navigation_history.append(
                                {"url": frame.url, "timestamp": datetime.now().isoformat()}
                            )
                    except Exception:
                        pass

                self.page.on("framenavigated", on_frame_navigated)
            except Exception as e:
                logger.debug(f"Failed to attach navigation listener: {e}")

    def _record_action(self, action_type: str, details: Dict[str, Any]) -> None:
        self.action_history.append(
            {"type": action_type, "details": details, "timestamp": datetime.now().isoformat()}
        )

    async def navigate(
        self,
        url: str = Field(..., description="URL to navigate to"),
        wait_for_load_state: Literal["load", "domcontentloaded", "networkidle"] = Field(
            "networkidle", description="Wait condition after navigation"
        ),
    ) -> dict:
        self._record_action("navigate", {"url": url, "state": wait_for_load_state})
        return await super().navigate(url, wait_for_load_state)

