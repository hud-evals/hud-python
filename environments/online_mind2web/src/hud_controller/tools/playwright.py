"""Playwright web automation tool for HUD."""

from __future__ import annotations

import logging
import os
import base64
from datetime import datetime
from typing import TYPE_CHECKING, Any, Literal

from mcp import ErrorData, McpError
from mcp.types import INVALID_PARAMS, ContentBlock
from pydantic import Field
from hud.tools.playwright import PlaywrightTool, ContentResult
from playwright.sync_api import expect

if TYPE_CHECKING:
    from playwright.async_api import Browser, BrowserContext, Page

logger = logging.getLogger(__name__)


class OnlineMind2Web_PlaywrightTool(PlaywrightTool):
    """Enhanced Playwright tool with screenshot and action recording for Mind2Web."""

    def __init__(self, page=None, cdp_url=None):
        super().__init__(page=page, cdp_url=cdp_url)
        # Register callbacks for recording
        self.add_callback("on_screenshot_action", self._on_screenshot_action)
        self.add_callback("on_recorded_action", self._on_recorded_action)

    async def _on_screenshot_action(self, **_) -> bytes | None:
        """Callback to take and save screenshots to /screenshot directory."""
        try:
            # Ensure browser connection is alive before taking screenshot
            await self._ensure_browser()
            if self.page is None:
                logger.debug("Page not initialized, skipping screenshot")
                return

            # Take screenshot with animations disabled to avoid font loading delays
            screenshot_bytes = await self.page.screenshot(full_page=False, animations="disabled")
            screenshot_base64 = base64.b64encode(screenshot_bytes).decode()

            # Create screenshot directory
            screenshot_dir = "/screenshot"
            os.makedirs(screenshot_dir, exist_ok=True)

            # Generate timestamp-based filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            filename = f"screenshot_{timestamp}.png"
            filepath = os.path.join(screenshot_dir, filename)

            # Decode and save
            image_data = base64.b64decode(screenshot_base64)
            with open(filepath, "wb") as f:
                f.write(image_data)

            logger.info(f"Saved screenshot to {filepath}")
            return image_data

        except Exception as e:
            logger.debug(f"Screenshot callback failed: {e}")

    async def _on_recorded_action(
        self,
        action=None,
        selector=None,
        text=None,
        value=None,
        label=None,
        **_,
    ):
        """Record action in unified representation format."""
        if not action:
            return

        try:
            # Create action representation
            action_repr = self._to_action_repr(action, selector, text, value, label)

            # Save to action history
            action_history_dir = "/action_history"
            os.makedirs(action_history_dir, exist_ok=True)
            action_file = os.path.join(action_history_dir, "action_history.txt")

            with open(action_file, "a", encoding="utf-8") as f:
                f.write(f"{action_repr}\n")

            logger.info(f"Recorded action: {action_repr}")

        except Exception as e:
            logger.warning(f"Failed to record action: {e}")

    def _to_action_repr(self, action, selector=None, text=None, value=None, label=None):
        """Create unified action representation.

        Format examples:
        - <selector=button.submit> -> CLICK
        - <selector=input#email> -> TYPE user@example.com
        - <selector=select#country, value=US> -> SELECT
        """
        # Normalize action name
        action_name = action.upper()

        # Build element attributes
        attributes = []
        if selector:
            attributes.append(f"selector={selector}")
        if value:
            attributes.append(f"value={value}")
        if label:
            attributes.append(f"label={label}")

        # Create element part
        element_part = f"<{', '.join(attributes)}>" if attributes else "<>"

        # Create action part
        if text and action_name == "TYPE":
            action_part = f"TYPE {text}"
        elif action_name == "SELECT_OPTION":
            action_part = "SELECT"
        else:
            action_part = action_name

        return f"{element_part} -> {action_part}"

    async def get_elements(self, element_type: str | None = None) -> dict[str, Any]:
        """Get interactive elements on the page with their selectors.

        Args:
            element_type: Optional filter for element type (e.g., 'button', 'a', 'input')

        Returns:
            Dict with list of elements and their properties
        """
        await self._ensure_browser()
        if self.page is None:
            raise RuntimeError("Page not initialized")

        try:
            # JavaScript to extract interactive elements
            js_code = """
            (elementType) => {
                const elements = [];
                let selector = elementType || 'a, button, input, select, textarea, [role="button"], [onclick]';

                document.querySelectorAll(selector).forEach((el, idx) => {
                    if (!el.offsetParent && el.tagName !== 'INPUT') return; // Skip hidden elements

                    const rect = el.getBoundingClientRect();
                    if (rect.width === 0 || rect.height === 0) return; // Skip zero-size elements

                    const info = {
                        tag: el.tagName.toLowerCase(),
                        text: el.textContent?.trim().substring(0, 100) || '',
                        type: el.type || '',
                        id: el.id || '',
                        classes: Array.from(el.classList).join('.'),
                        name: el.name || '',
                        placeholder: el.placeholder || '',
                        href: el.href || '',
                        value: el.value || '',
                        role: el.getAttribute('role') || '',
                        ariaLabel: el.getAttribute('aria-label') || '',
                    };

                    // Generate suggested selector
                    let selector = '';
                    if (info.id) {
                        selector = `#${info.id}`;
                    } else if (info.name) {
                        selector = `${info.tag}[name="${info.name}"]`;
                    } else if (info.classes) {
                        selector = `${info.tag}.${info.classes}`;
                    } else if (info.text && info.text.length > 0 && info.text.length < 30) {
                        selector = `text=${info.text}`;
                    } else {
                        selector = info.tag;
                    }

                    elements.push({
                        selector: selector,
                        ...info
                    });
                });

                return elements;
            }
            """

            elements = await self.page.evaluate(js_code, element_type)

            if not elements:
                return {
                    "success": True,
                    "elements": [],
                    "message": f"No interactive elements found{f' of type {element_type}' if element_type else ''}",
                }

            # Format output for better readability
            formatted_elements = []
            for i, elem in enumerate(elements[:50], 1):  # Limit to 50 elements
                elem_desc = f"{i}. {elem['selector']}"
                if elem["text"]:
                    elem_desc += f" - '{elem['text'][:50]}'"
                if elem["placeholder"]:
                    elem_desc += f" (placeholder: {elem['placeholder']})"
                if elem["type"]:
                    elem_desc += f" [type={elem['type']}]"
                formatted_elements.append(elem_desc)

            message = f"Found {len(elements)} interactive elements:\n" + "\n".join(
                formatted_elements
            )
            if len(elements) > 50:
                message += f"\n... and {len(elements) - 50} more elements"

            return {"success": True, "elements": elements, "message": message}

        except Exception as e:
            logger.error("Get elements failed: %s", e)
            return {
                "success": False,
                "error": str(e),
                "message": f"Failed to get elements: {e}",
            }

    async def get_page_content(self) -> dict[str, Any]:
        """Get simplified page content including HTML structure and text.

        Returns:
            Dict with page content information
        """
        await self._ensure_browser()
        if self.page is None:
            raise RuntimeError("Page not initialized")

        try:
            # Get basic page info
            url = self.page.url
            title = await self.page.title()

            # Get simplified DOM structure focusing on semantic and interactive elements
            js_code = """
            () => {
                function getSimplifiedDOM(element, depth = 0, maxDepth = 4) {
                    if (depth > maxDepth) return null;

                    // Skip script, style, and hidden elements
                    if (['SCRIPT', 'STYLE', 'NOSCRIPT'].includes(element.tagName)) return null;
                    if (element.offsetParent === null && element.tagName !== 'BODY') return null;

                    const node = {
                        tag: element.tagName.toLowerCase(),
                    };

                    // Add relevant attributes
                    if (element.id) node.id = element.id;
                    if (element.className && typeof element.className === 'string') {
                        node.class = element.className.split(' ').filter(c => c).slice(0, 3).join(' ');
                    }
                    if (element.href) node.href = element.href;
                    if (element.type) node.type = element.type;
                    if (element.placeholder) node.placeholder = element.placeholder;
                    if (element.name) node.name = element.name;

                    // Get text content for leaf nodes or small elements
                    const text = element.textContent?.trim();
                    if (element.children.length === 0 && text && text.length < 100) {
                        node.text = text;
                    }

                    // Recursively process children
                    const children = [];
                    for (const child of element.children) {
                        const childNode = getSimplifiedDOM(child, depth + 1, maxDepth);
                        if (childNode) children.push(childNode);
                    }

                    if (children.length > 0) {
                        node.children = children;
                    }

                    return node;
                }

                const body = document.body;
                return {
                    structure: getSimplifiedDOM(body),
                    mainText: document.body.innerText?.substring(0, 2000) || '',
                };
            }
            """

            content = await self.page.evaluate(js_code)

            # Format structure as readable text
            def format_structure(node, indent=0):
                if not node:
                    return ""

                lines = []
                prefix = "  " * indent

                # Format node
                tag_str = f"{prefix}<{node['tag']}"
                if node.get("id"):
                    tag_str += f' id="{node["id"]}"'
                if node.get("class"):
                    tag_str += f' class="{node["class"]}"'
                if node.get("href"):
                    tag_str += f' href="{node["href"][:50]}"'
                if node.get("type"):
                    tag_str += f' type="{node["type"]}"'
                if node.get("placeholder"):
                    tag_str += f' placeholder="{node["placeholder"]}"'
                if node.get("name"):
                    tag_str += f' name="{node["name"]}"'
                tag_str += ">"

                if node.get("text"):
                    tag_str += f" {node['text']}"

                lines.append(tag_str)

                # Process children
                if node.get("children"):
                    for child in node["children"][:20]:  # Limit children shown
                        lines.append(format_structure(child, indent + 1))

                return "\n".join(lines)

            structure_text = format_structure(content.get("structure", {}))
            main_text = content.get("mainText", "")

            message = f"""Page: {title}
URL: {url}

Main Text Content (first 2000 chars):
{main_text}

Page Structure:
{structure_text[:3000]}
"""

            return {"success": True, "content": content, "message": message}

        except Exception as e:
            logger.error("Get page content failed: %s", e)
            return {
                "success": False,
                "error": str(e),
                "message": f"Failed to get page content: {e}",
            }

    async def __call__(
        self,
        action: str = Field(
            ...,
            description="The action to perform (navigate, screenshot, click, type, select_option, wait_for_element, get_elements, get_page_content)",  # noqa: E501
        ),
        url: str | None = Field(None, description="URL to navigate to (for navigate action)"),
        selector: str | None = Field(
            None,
            description="""CSS selector, Playwright locator syntax, or XPath for element selection. Examples:
- CSS: 'button.submit', '#email', 'a[href="/api"]', 'div.container > p'
- Playwright text: 'text=Click here', 'text=/API.*/i' (case-insensitive regex)
- Playwright role: 'role=button[name="Submit"]'
- XPath: '//button[contains(text(), "Submit")]'
- Attribute: '[data-testid="login-button"]'
If selector matches multiple elements, Playwright will use the first visible one. For stricter matching, make selector more specific.""",  # noqa: E501
        ),
        text: str | None = Field(None, description="Text to type (for type action)"),
        value: str | None = Field(
            None, description="Option value to select (for select_option action)"
        ),
        label: str | None = Field(
            None, description="Option label to select (for select_option action)"
        ),
        index: int | None = Field(
            None, description="Option index to select (for select_option action)"
        ),
        element_type: str | None = Field(
            None,
            description="Element type filter for get_elements (e.g., 'button', 'a', 'input', 'select'). Leave empty to get all interactive elements.",  # noqa: E501
        ),
        wait_for_load_state: Literal["commit", "domcontentloaded", "load", "networkidle"]
        | None = Field(
            None,
            description="State to wait for: commit, domcontentloaded, load, networkidle (default: load)",  # noqa: E501
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
                result = await self.navigate(url, wait_for_load_state or "load")

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

            elif action == "select_option":
                if selector is None:
                    raise McpError(
                        ErrorData(
                            code=INVALID_PARAMS,
                            message="selector parameter is required for select_option",
                        )
                    )
                # Implement select_option using page API
                await self._ensure_browser()
                if self.page is None:
                    raise RuntimeError("Page not initialized")
                locator = self.page.locator(selector)
                if value:
                    await locator.select_option(value=value)
                    result = {"success": True, "message": f"Selected option by value: {value}"}
                elif label:
                    await locator.select_option(label=label)
                    result = {"success": True, "message": f"Selected option by label: {label}"}
                elif index is not None:
                    await locator.select_option(index=index)
                    result = {"success": True, "message": f"Selected option by index: {index}"}
                else:
                    result = {"success": False, "error": "Must provide value, label, or index"}

            elif action == "wait_for_element":
                if selector is None:
                    raise McpError(
                        ErrorData(
                            code=INVALID_PARAMS,
                            message="selector parameter is required for wait_for_element",
                        )
                    )
                result = await self.wait_for_element(selector)

            elif action == "get_elements":
                result = await self.get_elements(element_type)

            elif action == "get_page_content":
                result = await self.get_page_content()

            else:
                raise McpError(ErrorData(code=INVALID_PARAMS, message=f"Unknown action: {action}"))

            # Trigger callbacks for screenshot and action recording
            screenshot_actions = {
                "navigate",
                "click",
                "type",
                "select_option",
            }
            if action in screenshot_actions and action != "screenshot":
                try:
                    os.makedirs("/screenshot", exist_ok=True)
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                    screenshot_path = f"/screenshot/screenshot_{timestamp}.png"
                    # Take screenshot and save to file
                    screenshot_result = await self.screenshot(path=screenshot_path)
                    # Add screenshot to result if successful
                    if screenshot_result.base64_image and isinstance(result, dict):
                        result = ContentResult(
                            output=result.get("message", ""),
                            base64_image=screenshot_result.base64_image,
                        )
                    elif screenshot_result.base64_image and isinstance(result, ContentResult):
                        result = ContentResult(
                            output=result.output,
                            error=result.error,
                            base64_image=screenshot_result.base64_image,
                        )

                    if screenshot_result.base64_image:
                        logger.info(f"Saved screenshot to {screenshot_path}")
                except Exception as e:
                    logger.warning(f"Screenshot after action failed: {e}")

            recorded_actions = {
                "navigate",
                "click",
                "type",
                "select_option",
                "screenshot",
            }
            if action in recorded_actions:
                await self._trigger_callbacks(
                    "on_recorded_action",
                    action=action,
                    selector=selector,
                    text=text,
                    value=value,
                    label=label,
                )

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

    async def screenshot(self, path=None) -> ContentResult:
        """Take a screenshot of the current page.

        Returns:
            ToolResult with base64_image
        """
        await self._ensure_browser()
        if self.page is None:
            raise RuntimeError("Page not initialized after _ensure_browser")

        try:
            # Always return base64 encoded screenshot as ToolResult
            screenshot_bytes = await self.page.screenshot(
                full_page=False, animations="disabled", path=path
            )
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
    ) -> dict[str, Any]:
        """Click an element by selector.

        Args:
            selector: CSS selector for element to click
            button: Mouse button to use (left, right, middle)
            count: Number of clicks

        Returns:
            Dict with click result
        """
        await self._ensure_browser()
        if self.page is None:
            raise RuntimeError("Page not initialized after _ensure_browser")

        try:
            await self.page.click(selector, button=button, click_count=count, timeout=10000)
            return {"success": True, "message": f"Clicked element: {selector}"}
        except Exception as e:
            logger.error("Click failed: %s", e)
            return {
                "success": False,
                "error": str(e),
                "message": f"Failed to click {selector}: {e}",
            }
