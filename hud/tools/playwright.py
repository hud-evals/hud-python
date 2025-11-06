"""Playwright web automation tool for HUD."""

from __future__ import annotations

import json
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
            description=(
                "The action to perform (navigate, screenshot, click, type, hover, press, focus, "
                "select_option, check, uncheck, clear, set_input_files, scroll_into_view, "
                "wait_for_element, wait_for_load_state, evaluate, get_page_info, get_iframes)"
            ),
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
        key: str | None = Field(None, description="Single key to press (for press action)"),
        keys: list[str] | None = Field(None, description="Sequence of keys to press (press action)"),
        values: list[str] | str | None = Field(
            None, description="Option value(s) to select (for select_option action)"
        ),
        files: list[str] | None = Field(
            None, description="File paths to upload (for set_input_files action)"
        ),
        expression: str | None = Field(None, description="JavaScript expression for evaluate action"),
        argument: Any | None = Field(
            None, description="Optional JSON-serializable argument for evaluate action"
        ),
        state: Literal["commit", "domcontentloaded", "load", "networkidle"] | None = Field(
            None, description="Load state to wait for (for wait_for_load_state action)"
        ),
        force: bool = Field(False, description="Whether to force the action when applicable"),
        timeout_ms: int | None = Field(
            None, description="Override default timeout in milliseconds for the action"
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

            elif action == "get_iframes":
                result = await self.get_iframes()

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

    def _get_locator(self, selector: str):
        """Get a locator that handles iframe traversal.

        Args:
            selector: CSS selector, potentially with iframe >> syntax
                     Examples:
                     - "button" - regular selector
                     - "iframe#myframe >> button" - iframe traversal
                     - "iframe >> iframe >> button" - nested iframes

        Returns:
            Playwright locator or FrameLocator
        """
        if self.page is None:
            raise RuntimeError("Page not initialized")

        # Split on >> to handle iframe traversal
        parts = [part.strip() for part in selector.split(">>")]

        if len(parts) == 1:
            return self.page.locator(parts[0])

        # Detect iframe traversal. Only treat the chain specially if every intermediate
        # segment explicitly targets an iframe/frame element (e.g., iframe#foo, frame[name="x"]).
        iframe_segments = parts[:-1]
        if not iframe_segments or any(
            not seg.lower().startswith(("iframe", "frame")) for seg in iframe_segments
        ):
            return self.page.locator(" >> ".join(parts))

        locator = self.page.frame_locator(iframe_segments[0])

        for segment in iframe_segments[1:]:
            locator = locator.frame_locator(segment)

        return locator.locator(parts[-1].strip())

    async def click(
        self,
        selector: str,
        button: Literal["left", "right", "middle"] = "left",
        count: int = 1,
        wait_for_navigation: bool = True,
        force: bool = False,
        timeout_ms: int | None = None,
        navigation_state: Literal["commit", "domcontentloaded", "load", "networkidle"] | None = None,
    ) -> dict[str, Any]:
        """Click an element by selector.

        Args:
            selector: CSS selector for element to click (supports iframe >> selector syntax)
                     For iframes, use: "iframe#id >> button"
                     For nested iframes: "iframe.outer >> iframe.inner >> button"

        Returns:
            Dict with click result
        """
        await self._ensure_browser()
        if self.page is None:
            raise RuntimeError("Page not initialized after _ensure_browser")

        try:
            locator = self._get_locator(selector)
            timeout = timeout_ms if timeout_ms is not None else 30000
            click_kwargs: dict[str, Any] = {"button": button, "click_count": count, "timeout": timeout}
            if force:
                click_kwargs["force"] = True
            await locator.click(**click_kwargs)

            if wait_for_navigation:
                await self.page.wait_for_load_state(
                    navigation_state or "load", timeout=timeout_ms if timeout_ms is not None else 30000
                )
            return {"success": True, "message": f"Clicked element: {selector}"}
        except Exception as e:
            logger.error("Click failed: %s", e)
            return {
                "success": False,
                "error": str(e),
                "message": f"Failed to click {selector}: {e}",
            }

    async def type_text(
        self, selector: str, text: str, force: bool = False, timeout_ms: int | None = None
    ) -> dict[str, Any]:
        """Type text into an element.

        Args:
            selector: CSS selector for input element (supports iframe >> selector syntax)
            text: Text to type

        Returns:
            Dict with type result
        """
        await self._ensure_browser()
        if self.page is None:
            raise RuntimeError("Page not initialized after _ensure_browser")

        try:
            locator = self._get_locator(selector)
            timeout = timeout_ms if timeout_ms is not None else 30000
            fill_kwargs: dict[str, Any] = {"timeout": timeout}
            if force:
                fill_kwargs["force"] = True
            await locator.fill(text, **fill_kwargs)
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

    async def wait_for_element(self, selector: str, timeout_ms: int | None = None) -> dict[str, Any]:
        """Wait for an element to appear.

        Args:
            selector: CSS selector for element (supports iframe >> selector syntax)

        Returns:
            Dict with wait result
        """
        await self._ensure_browser()
        if self.page is None:
            raise RuntimeError("Page not initialized after _ensure_browser")

        try:
            locator = self._get_locator(selector)
            timeout = timeout_ms if timeout_ms is not None else 30000
            await locator.wait_for(state="visible", timeout=timeout)
            return {"success": True, "message": f"Element {selector} appeared"}
        except Exception as e:
            logger.error("Wait for element failed: %s", e)
            return {
                "success": False,
                "error": str(e),
                "message": f"Element {selector} did not appear within 30000ms: {e}",
            }

    async def get_iframes(self) -> dict[str, Any]:
        """Get information about all iframes on the page.

        Returns:
            Dict with iframe structure information
        """
        await self._ensure_browser()
        if self.page is None:
            raise RuntimeError("Page not initialized after _ensure_browser")

        try:
            # Get all iframe elements via selector
            iframe_locator = self.page.locator("iframe")
            iframe_count = await iframe_locator.count()
            
            iframes = []
            for i in range(min(iframe_count, 20)):  # Limit to first 20 to avoid overwhelming
                try:
                    iframe_elem = iframe_locator.nth(i)
                    
                    # Get iframe attributes
                    iframe_id = await iframe_elem.get_attribute("id") or f"(no id)"
                    iframe_name = await iframe_elem.get_attribute("name") or "(no name)"
                    iframe_src = await iframe_elem.get_attribute("src") or "(no src)"
                    
                    # Try to get element counts inside the iframe
                    frame_loc = self.page.frame_locator(f"iframe").nth(i)
                    try:
                        # Count basic elements inside the iframe
                        body_count = await frame_loc.locator("body").count()
                        link_count = await frame_loc.locator("a").count()
                        button_count = await frame_loc.locator("button").count()
                        iframe_nested_count = await frame_loc.locator("iframe").count()
                        
                        content_info = f"{link_count} links, {button_count} buttons"
                        if iframe_nested_count > 0:
                            content_info += f", {iframe_nested_count} nested iframes"
                        if body_count == 0:
                            content_info = "(empty or not loaded)"
                    except Exception:
                        content_info = "(unable to inspect - may be cross-origin or not loaded)"
                    
                    # Build selector hint
                    if iframe_id and iframe_id != "(no id)":
                        selector = f"iframe#{iframe_id}"
                    elif iframe_name and iframe_name != "(no name)":
                        selector = f"iframe[name='{iframe_name}']"
                    else:
                        selector = f"iframe (index {i})"
                    
                    iframes.append({
                        "index": i,
                        "selector": selector,
                        "id": iframe_id,
                        "name": iframe_name,
                        "src": iframe_src[:80] if len(iframe_src) > 80 else iframe_src,
                        "content": content_info
                    })
                except Exception as e:
                    iframes.append({
                        "index": i,
                        "error": f"Could not inspect: {e}"
                    })
            
            # Format output message
            if iframe_count == 0:
                message = "No iframes found on this page."
            else:
                message = f"Found {iframe_count} iframe(s) on page:\n\n"
                for iframe in iframes:
                    if "error" in iframe:
                        message += f"  [{iframe['index']}] {iframe['error']}\n"
                    else:
                        message += f"  [{iframe['index']}] {iframe['selector']}\n"
                        message += f"      ID: {iframe['id']}, Name: {iframe['name']}\n"
                        message += f"      Src: {iframe['src']}\n"
                        message += f"      Content: {iframe['content']}\n"
                        message += "\n"
                
                if iframe_count > 20:
                    message += f"\n(Showing first 20 of {iframe_count} iframes)\n"
                
                message += "\nTo interact with iframe content, use: 'iframe#id >> selector' or 'iframe[name=\"name\"] >> selector'"
            
            return {"success": True, "message": message}
        
        except Exception as e:
            logger.error("Get iframes failed: %s", e)
            return {
                "success": False,
                "error": str(e),
                "message": f"Failed to get iframe information: {e}",
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
