"""Browser executor reusing Playwright for deep_research environment."""

import base64
import logging
from typing import Literal

from hud.tools.executors.base import BaseExecutor
from hud.tools.types import ContentResult

logger = logging.getLogger(__name__)


class BrowserExecutor(BaseExecutor):
    def __init__(self, playwright_tool, display_num: int | None = None):
        super().__init__(display_num)
        self.playwright_tool = playwright_tool

    async def _ensure_page(self):
        await self.playwright_tool._ensure_browser()
        if not self.playwright_tool.page:
            raise RuntimeError("No browser page available")
        return self.playwright_tool.page

    async def screenshot(self) -> str | None:
        try:
            page = await self._ensure_page()
            buf = await page.screenshot(full_page=False)
            return base64.b64encode(buf).decode()
        except Exception as e:
            logger.error(f"Screenshot failed: {e}")
            return None

    async def click(
        self,
        x: int | None = None,
        y: int | None = None,
        button: Literal["left", "right", "middle", "back", "forward"] = "left",
        pattern: list[int] | None = None,
        hold_keys: list[str] | None = None,
        take_screenshot: bool = True,
    ) -> ContentResult:
        try:
            page = await self._ensure_page()
            if x is None or y is None:
                return ContentResult(error="Coordinates required for click")

            # Handle modifier keys
            if hold_keys:
                for key in hold_keys:
                    await page.keyboard.down(key)

            if pattern:
                for delay in pattern:
                    await page.mouse.click(x, y, button=button if button in ["left", "right", "middle"] else "left")
                    if delay > 0:
                        await page.wait_for_timeout(delay)
            else:
                await page.mouse.click(x, y, button=button if button in ["left", "right", "middle"] else "left")

            if hold_keys:
                for key in hold_keys:
                    await page.keyboard.up(key)

            result = ContentResult(output=f"Clicked at ({x}, {y})")
            if take_screenshot:
                result = result + ContentResult(base64_image=await self.screenshot())
            return result
        except Exception as e:
            return ContentResult(error=str(e))

