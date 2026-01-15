"""Browser executor for PDF computer use."""

import logging
from typing import Literal

from hud.tools.executors.base import BaseExecutor
from hud.tools.types import ContentResult

from .browser import pdf_browser

logger = logging.getLogger(__name__)


class PDFBrowserExecutor(BaseExecutor):
    """Executor that performs actions in the PDF browser viewer."""

    def __init__(self, display_num: int | None = None):
        super().__init__(display_num)
        logger.info("PDFBrowserExecutor initialized")

    async def screenshot(self) -> str | None:
        """Take a screenshot."""
        return await pdf_browser.screenshot()

    async def click(
        self,
        x: int | None = None,
        y: int | None = None,
        button: Literal["left", "right", "middle", "back", "forward"] = "left",
        pattern: list[int] | None = None,
        hold_keys: list[str] | None = None,
        take_screenshot: bool = True,
    ) -> ContentResult:
        """Click at coordinates."""
        if x is None or y is None:
            return ContentResult(error="Coordinates required for click")

        try:
            page = pdf_browser.page
            if not page:
                return ContentResult(error="Browser not started")

            # Handle multi-click patterns (double-click, triple-click)
            if pattern:
                for delay in pattern:
                    await page.mouse.click(x, y, button=button)
                    if delay > 0:
                        import asyncio
                        await asyncio.sleep(delay / 1000)
            else:
                await page.mouse.click(x, y, button=button)

            logger.debug(f"Clicked at ({x}, {y})")

            result = ContentResult(output=f"Clicked at ({x}, {y})")
            if take_screenshot:
                screenshot = await self.screenshot()
                if screenshot:
                    result = ContentResult(output=result.output, base64_image=screenshot)
            return result

        except Exception as e:
            logger.error(f"Click failed: {e}")
            return ContentResult(error=str(e))

    async def write(
        self,
        text: str,
        enter_after: bool = False,
        hold_keys: list[str] | None = None,
        take_screenshot: bool = True,
    ) -> ContentResult:
        """Type text."""
        try:
            page = pdf_browser.page
            if not page:
                return ContentResult(error="Browser not started")

            await page.keyboard.type(text)

            if enter_after:
                await page.keyboard.press("Enter")

            logger.debug(f"Typed: {text[:50]}...")

            result = ContentResult(output=f"Typed: {text}")
            if take_screenshot:
                screenshot = await self.screenshot()
                if screenshot:
                    result = ContentResult(output=result.output, base64_image=screenshot)
            return result

        except Exception as e:
            logger.error(f"Type failed: {e}")
            return ContentResult(error=str(e))

    async def press(
        self,
        keys: list[str],
        take_screenshot: bool = True,
    ) -> ContentResult:
        """Press keyboard keys."""
        try:
            page = pdf_browser.page
            if not page:
                return ContentResult(error="Browser not started")

            # Press keys as combination
            key_combo = "+".join(keys)
            await page.keyboard.press(key_combo)

            logger.debug(f"Pressed: {key_combo}")

            result = ContentResult(output=f"Pressed: {key_combo}")
            if take_screenshot:
                screenshot = await self.screenshot()
                if screenshot:
                    result = ContentResult(output=result.output, base64_image=screenshot)
            return result

        except Exception as e:
            logger.error(f"Key press failed: {e}")
            return ContentResult(error=str(e))

    async def scroll(
        self,
        x: int | None = None,
        y: int | None = None,
        scroll_x: int | None = None,
        scroll_y: int | None = None,
        hold_keys: list[str] | None = None,
        take_screenshot: bool = True,
    ) -> ContentResult:
        """Scroll the page."""
        try:
            page = pdf_browser.page
            if not page:
                return ContentResult(error="Browser not started")

            # Move to position if provided
            if x is not None and y is not None:
                await page.mouse.move(x, y)

            # Scroll
            delta_x = scroll_x or 0
            delta_y = scroll_y or 0
            await page.mouse.wheel(delta_x, delta_y)

            logger.debug(f"Scrolled by ({delta_x}, {delta_y})")

            result = ContentResult(output=f"Scrolled by ({delta_x}, {delta_y})")
            if take_screenshot:
                screenshot = await self.screenshot()
                if screenshot:
                    result = ContentResult(output=result.output, base64_image=screenshot)
            return result

        except Exception as e:
            logger.error(f"Scroll failed: {e}")
            return ContentResult(error=str(e))

    async def move(
        self,
        x: int | None = None,
        y: int | None = None,
        take_screenshot: bool = True,
    ) -> ContentResult:
        """Move mouse to coordinates."""
        if x is None or y is None:
            return ContentResult(error="Coordinates required for move")

        try:
            page = pdf_browser.page
            if not page:
                return ContentResult(error="Browser not started")

            await page.mouse.move(x, y)

            result = ContentResult(output=f"Moved to ({x}, {y})")
            if take_screenshot:
                screenshot = await self.screenshot()
                if screenshot:
                    result = ContentResult(output=result.output, base64_image=screenshot)
            return result

        except Exception as e:
            logger.error(f"Move failed: {e}")
            return ContentResult(error=str(e))

    async def drag(
        self,
        path: list[tuple[int, int]],
        button: Literal["left", "right", "middle"] = "left",
        hold_keys: list[str] | None = None,
        take_screenshot: bool = True,
    ) -> ContentResult:
        """Drag along a path."""
        if not path or len(path) < 2:
            return ContentResult(error="Path must have at least 2 points")

        try:
            page = pdf_browser.page
            if not page:
                return ContentResult(error="Browser not started")

            # Start drag
            start_x, start_y = path[0]
            await page.mouse.move(start_x, start_y)
            await page.mouse.down(button=button)

            # Move through path
            for x, y in path[1:]:
                await page.mouse.move(x, y)

            # End drag
            await page.mouse.up(button=button)

            result = ContentResult(output=f"Dragged through {len(path)} points")
            if take_screenshot:
                screenshot = await self.screenshot()
                if screenshot:
                    result = ContentResult(output=result.output, base64_image=screenshot)
            return result

        except Exception as e:
            logger.error(f"Drag failed: {e}")
            return ContentResult(error=str(e))
