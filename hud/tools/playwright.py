"""Playwright web automation tool for HUD."""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any, Literal

from mcp import ErrorData, McpError
from mcp.types import INVALID_PARAMS, ContentBlock
from pydantic import Field

from .base import BaseTool
from .types import ContentResult

if TYPE_CHECKING:
    from playwright.async_api import Browser, BrowserContext, Page

logger = logging.getLogger(__name__)


class PlaywrightTool(BaseTool):
    """Playwright tool for web automation."""

    def __init__(self, page: Page | None = None, cdp_url: str | None = None) -> None:
        """Initialize PlaywrightTool.

        Args:
            page: Optional existing Playwright Page to use as context
            cdp_url: Optional Chrome DevTools Protocol URL for connecting to existing browser
        """
        super().__init__(
            env=page,
            name="playwright",
            title="Playwright Browser",
            description="Web automation tool using Playwright",
        )
        self._cdp_url = cdp_url
        self._playwright = None
        # Internal browser management - not exposed as context
        self._browser: Browser | None = None
        self._browser_context: BrowserContext | None = None

    @property
    def page(self) -> Page | None:
        """Get the current page."""
        return self.env

    @page.setter
    def page(self, value: Page | None) -> None:
        """Set the page."""
        self.env = value

    async def __call__(
        self,
        action: str = Field(
            ...,
            description="The action to perform (navigate, screenshot, click, type, get_page_info, wait_for_element)",  # noqa: E501
        ),
        url: str | None = Field(None, description="URL to navigate to (for navigate action)"),
        selector: str | None = Field(
            None, description="CSS selector for element (for click, type, wait_for_element actions)"
        ),
        text: str | None = Field(None, description="Text to type (for type action)"),
        wait_for_load_state: Literal["commit", "domcontentloaded", "load", "networkidle"]
        | None = Field(
            None,
            description="State to wait for: commit, domcontentloaded, load, networkidle (default: networkidle)",  # noqa: E501
        ),
    ) -> list[ContentBlock]:
        """
        Execute a Playwright web automation action.

        Returns:
            List of MCP content blocks
        """
        logger.info("PlaywrightTool executing action: %s", action)

        try:
            if action == "navigate":
                if url is None:
                    raise McpError(
                        ErrorData(
                            code=INVALID_PARAMS, message="url parameter is required for navigate"
                        )
                    )
                # Guard against pydantic FieldInfo default leaking through
                if not isinstance(wait_for_load_state, str):
                    wait_for_load_state = None
                result = await self.navigate(url, wait_for_load_state or "networkidle")

            elif action == "screenshot":
                result = await self.screenshot()

            elif action == "click":
                if selector is None:
                    raise McpError(
                        ErrorData(
                            code=INVALID_PARAMS, message="selector parameter is required for click"
                        )
                    )
                result = await self.click(selector)

            elif action == "type":
                if selector is None:
                    raise McpError(
                        ErrorData(
                            code=INVALID_PARAMS, message="selector parameter is required for type"
                        )
                    )
                if text is None:
                    raise McpError(
                        ErrorData(
                            code=INVALID_PARAMS, message="text parameter is required for type"
                        )
                    )
                result = await self.type_text(selector, text)

            elif action == "get_page_info":
                result = await self.get_page_info()

            elif action == "wait_for_element":
                if selector is None:
                    raise McpError(
                        ErrorData(
                            code=INVALID_PARAMS,
                            message="selector parameter is required for wait_for_element",
                        )
                    )
                result = await self.wait_for_element(selector)

            else:
                raise McpError(ErrorData(code=INVALID_PARAMS, message=f"Unknown action: {action}"))

            # Convert dict result to ToolResult
            if isinstance(result, dict):
                if result.get("success"):
                    tool_result = ContentResult(output=result.get("message", ""))
                else:
                    tool_result = ContentResult(error=result.get("error", "Unknown error"))
            elif isinstance(result, ContentResult):
                tool_result = result
            else:
                tool_result = ContentResult(output=str(result))

            # Convert result to content blocks
            return tool_result.to_content_blocks()

        except McpError:
            raise
        except Exception as e:
            logger.error("PlaywrightTool error: %s", e)
            raise McpError(ErrorData(code=INVALID_PARAMS, message=f"Playwright error: {e}")) from e

    async def _ensure_browser(self) -> None:
        """Ensure browser is launched and ready."""
        if self._browser is None or not self._browser.is_connected():
            if self._cdp_url:
                logger.info("Connecting to remote browser via CDP")
            else:
                logger.info("Launching Playwright browser...")

            # Ensure DISPLAY is set (only needed for local browser)
            if not self._cdp_url:
                os.environ["DISPLAY"] = os.environ.get("DISPLAY", ":1")

            if self._playwright is None:
                try:
                    from playwright.async_api import async_playwright

                    self._playwright = await async_playwright().start()
                except ImportError:
                    raise ImportError(
                        "Playwright is not installed. Please install with: pip install playwright"
                    ) from None

            # Connect via CDP URL or launch local browser
            if self._cdp_url:
                # Connect to remote browser via CDP
                self._browser = await self._playwright.chromium.connect_over_cdp(self._cdp_url)

                if self._browser is None:
                    raise RuntimeError("Failed to connect to remote browser")

                # Reuse existing context and page where possible to avoid spawning new windows
                contexts = self._browser.contexts
                if contexts:
                    self._browser_context = contexts[0]
                    # Prefer the first existing page to keep using the already visible window/tab
                    existing_pages = self._browser_context.pages
                    if existing_pages:
                        self.page = existing_pages[0]
                else:
                    # As a fallback, create a new context
                    self._browser_context = await self._browser.new_context(
                        viewport={"width": 1920, "height": 1080},
                        ignore_https_errors=True,
                    )
            else:
                # Launch local browser
                self._browser = await self._playwright.chromium.launch(
                    headless=False,
                    args=[
                        "--no-sandbox",
                        "--disable-dev-shm-usage",
                        "--disable-gpu",
                        "--disable-web-security",
                        "--disable-features=IsolateOrigins,site-per-process",
                        "--disable-blink-features=AutomationControlled",
                        "--window-size=1920,1080",
                        "--window-position=0,0",
                        "--start-maximized",
                        "--disable-background-timer-throttling",
                        "--disable-backgrounding-occluded-windows",
                        "--disable-renderer-backgrounding",
                        "--disable-features=TranslateUI",
                        "--disable-ipc-flooding-protection",
                        "--disable-default-apps",
                        "--no-first-run",
                        "--disable-sync",
                        "--no-default-browser-check",
                    ],
                )

                if self._browser is None:
                    raise RuntimeError("Browser failed to initialize")

                self._browser_context = await self._browser.new_context(
                    viewport={"width": 1920, "height": 1080},
                    ignore_https_errors=True,
                )

            if self._browser_context is None:
                raise RuntimeError("Browser context failed to initialize")

            # Reuse existing page if available (for CDP connections), otherwise create new one
            pages = self._browser_context.pages
            if pages:
                self.page = pages[0]
                logger.info("Reusing existing browser page")
            else:
                self.page = await self._browser_context.new_page()
                logger.info("Created new browser page")
            logger.info("Playwright browser launched successfully")

    async def navigate(
        self,
        url: str,
        wait_for_load_state: Literal[
            "commit", "domcontentloaded", "load", "networkidle"
        ] = "networkidle",
    ) -> dict[str, Any]:
        """Navigate to a URL.

        Args:
            url: URL to navigate to
            wait_for_load_state: Load state to wait for (load, domcontentloaded, networkidle)

        Returns:
            Dict with navigation result
        """
        await self._ensure_browser()
        if self.page is None:
            raise RuntimeError("Page not initialized after _ensure_browser")

        logger.info("Navigating to %s", url)
        try:
            await self.page.goto(url, wait_until=wait_for_load_state)
            current_url = self.page.url
            title = await self.page.title()

            return {
                "success": True,
                "url": current_url,
                "title": title,
                "message": f"Successfully navigated to {url}",
            }
        except Exception as e:
            logger.error("Navigation failed: %s", e)
            return {
                "success": False,
                "error": str(e),
                "message": f"Failed to navigate to {url}: {e}",
            }

    async def screenshot(self) -> ContentResult:
        """Take a screenshot of the current page.

        Returns:
            ToolResult with base64_image
        """
        await self._ensure_browser()
        if self.page is None:
            raise RuntimeError("Page not initialized after _ensure_browser")

        try:
            # Always return base64 encoded screenshot as ToolResult
            screenshot_bytes = await self.page.screenshot(full_page=False)
            import base64

            screenshot_b64 = base64.b64encode(screenshot_bytes).decode()
            return ContentResult(base64_image=screenshot_b64)
        except Exception as e:
            logger.error("Screenshot failed: %s", e)
            return ContentResult(error=f"Failed to take screenshot: {e}")

    async def click(
        self,
        selector: str,
        button: Literal["left", "right", "middle"] = "left",
        count: int = 1,
        wait_for_navigation: bool = True,
    ) -> dict[str, Any]:
        """Click an element by selector.

        Args:
            selector: CSS selector for element to click

        Returns:
            Dict with click result
        """
        await self._ensure_browser()
        if self.page is None:
            raise RuntimeError("Page not initialized after _ensure_browser")

        try:
            await self.page.click(selector, button=button, click_count=count)
            return {"success": True, "message": f"Clicked element: {selector}"}
        except Exception as e:
            logger.error("Click failed: %s", e)
            return {
                "success": False,
                "error": str(e),
                "message": f"Failed to click {selector}: {e}",
            }

    async def type_text(self, selector: str, text: str) -> dict[str, Any]:
        """Type text into an element.

        Args:
            selector: CSS selector for input element
            text: Text to type

        Returns:
            Dict with type result
        """
        await self._ensure_browser()
        if self.page is None:
            raise RuntimeError("Page not initialized after _ensure_browser")

        try:
            await self.page.fill(selector, text)
            return {"success": True, "message": f"Typed '{text}' into {selector}"}
        except Exception as e:
            logger.error("Type failed: %s", e)
            return {
                "success": False,
                "error": str(e),
                "message": f"Failed to type into {selector}: {e}",
            }

    async def get_page_info(self) -> dict[str, Any]:
        """Get current page information.

        Returns:
            Dict with page info
        """
        await self._ensure_browser()
        if self.page is None:
            raise RuntimeError("Page not initialized after _ensure_browser")

        try:
            url = self.page.url
            title = await self.page.title()
            return {
                "success": True,
                "url": url,
                "title": title,
                "message": f"Current page: {title} ({url})",
            }
        except Exception as e:
            logger.error("Get page info failed: %s", e)
            return {"success": False, "error": str(e), "message": f"Failed to get page info: {e}"}

    async def wait_for_element(self, selector: str) -> dict[str, Any]:
        """Wait for an element to appear.

        Args:
            selector: CSS selector for element

        Returns:
            Dict with wait result
        """
        await self._ensure_browser()
        if self.page is None:
            raise RuntimeError("Page not initialized after _ensure_browser")

        try:
            await self.page.wait_for_selector(selector, timeout=30000)
            return {"success": True, "message": f"Element {selector} appeared"}
        except Exception as e:
            logger.error("Wait for element failed: %s", e)
            return {
                "success": False,
                "error": str(e),
                "message": f"Element {selector} did not appear within 30000ms: {e}",
            }

    async def close(self) -> None:
        """Close browser and cleanup."""
        if self._browser:
            try:
                await self._browser.close()
                logger.info("Browser closed")
            except Exception as e:
                logger.error("Error closing browser: %s", e)

        if self._playwright:
            try:
                await self._playwright.stop()
            except Exception as e:
                logger.error("Error stopping playwright: %s", e)

        self._browser = None
        self._browser_context = None
        self.env = None  # Clear the page
        self._playwright = None
