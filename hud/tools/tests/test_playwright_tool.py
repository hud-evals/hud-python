"""Tests for Playwright tool."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from mcp.shared.exceptions import McpError
from mcp.types import INVALID_PARAMS, ImageContent, TextContent

from hud.tools.playwright import PlaywrightTool


class TestPlaywrightTool:
    """Tests for PlaywrightTool."""

    @pytest.mark.asyncio
    async def test_playwright_tool_init(self):
        """Test tool initialization."""
        tool = PlaywrightTool()
        assert tool._browser is None
        assert tool._browser_context is None
        assert tool.page is None

    @pytest.mark.asyncio
    async def test_playwright_tool_invalid_action(self):
        """Test that invalid action raises error."""
        tool = PlaywrightTool()

        with pytest.raises(McpError) as exc_info:
            await tool(action="invalid_action")

        assert exc_info.value.error.code == INVALID_PARAMS
        assert "Unknown action" in exc_info.value.error.message

    @pytest.mark.asyncio
    async def test_playwright_tool_navigate_with_mocked_browser(self):
        """Test navigate action with mocked browser."""
        tool = PlaywrightTool()

        # Mock the browser components
        mock_page = AsyncMock()
        mock_page.goto = AsyncMock()

        with patch.object(tool, "_ensure_browser", new_callable=AsyncMock) as mock_ensure:
            # Set up the tool with mocked page
            tool.page = mock_page

            blocks = await tool(action="navigate", url="https://example.com")

            assert blocks is not None
            assert any(isinstance(b, TextContent) for b in blocks)
            # The actual call includes wait_until parameter with a Field object
            mock_page.goto.assert_called_once()
            args, _kwargs = mock_page.goto.call_args
            assert args[0] == "https://example.com"
            mock_ensure.assert_called_once()

    @pytest.mark.asyncio
    async def test_playwright_tool_click_with_mocked_browser(self):
        """Test click action with mocked browser."""
        tool = PlaywrightTool()

        # Mock the browser components
        mock_page = AsyncMock()
        mock_page.click = AsyncMock()

        with patch.object(tool, "_ensure_browser", new_callable=AsyncMock):
            # Set up the tool with mocked page
            tool.page = mock_page

            blocks = await tool(action="click", selector="button#submit")

            assert blocks is not None
            assert any(isinstance(b, TextContent) for b in blocks)
            mock_page.click.assert_called_once_with("button#submit", button="left", click_count=1)

    @pytest.mark.asyncio
    async def test_playwright_tool_type_with_mocked_browser(self):
        """Test type action with mocked browser."""
        tool = PlaywrightTool()

        # Mock the browser components
        mock_page = AsyncMock()
        mock_page.fill = AsyncMock()  # Playwright uses fill, not type

        with patch.object(tool, "_ensure_browser", new_callable=AsyncMock):
            # Set up the tool with mocked page
            tool.page = mock_page

            blocks = await tool(action="type", selector="input#name", text="John Doe")

            assert blocks is not None
            assert any(isinstance(b, TextContent) for b in blocks)
            mock_page.fill.assert_called_once_with("input#name", "John Doe")

    @pytest.mark.asyncio
    async def test_playwright_tool_screenshot_with_mocked_browser(self):
        """Test screenshot action with mocked browser."""
        tool = PlaywrightTool()

        # Mock the browser components
        mock_page = AsyncMock()
        mock_page.screenshot = AsyncMock(return_value=b"fake_screenshot_data")

        with patch.object(tool, "_ensure_browser", new_callable=AsyncMock):
            # Set up the tool with mocked page
            tool.page = mock_page

            blocks = await tool(action="screenshot")

            assert blocks is not None
            assert len(blocks) > 0
            assert any(isinstance(b, ImageContent | TextContent) for b in blocks)
            mock_page.screenshot.assert_called_once()

    @pytest.mark.asyncio
    async def test_playwright_tool_get_page_info_with_mocked_browser(self):
        """Test get_page_info action with mocked browser."""
        tool = PlaywrightTool()

        # Mock the browser components
        mock_page = AsyncMock()
        mock_page.url = "https://example.com"
        mock_page.title = AsyncMock(return_value="Example Page")
        mock_page.evaluate = AsyncMock(return_value={"height": 1000})

        with patch.object(tool, "_ensure_browser", new_callable=AsyncMock):
            # Set up the tool with mocked page
            tool.page = mock_page

            blocks = await tool(action="get_page_info")

            assert blocks is not None
            assert any(isinstance(b, TextContent) for b in blocks)
            # Check that the text contains expected info
            text_blocks = [b.text for b in blocks if isinstance(b, TextContent)]
            combined_text = " ".join(text_blocks)
            assert "https://example.com" in combined_text
            assert "Example Page" in combined_text

    @pytest.mark.asyncio
    async def test_playwright_tool_wait_for_element_with_mocked_browser(self):
        """Test wait_for_element action with mocked browser."""
        tool = PlaywrightTool()

        # Mock the browser components
        mock_page = AsyncMock()
        mock_page.wait_for_selector = AsyncMock()

        with patch.object(tool, "_ensure_browser", new_callable=AsyncMock):
            # Set up the tool with mocked page
            tool.page = mock_page

            # wait_for_element doesn't accept timeout parameter directly
            blocks = await tool(action="wait_for_element", selector="div#loaded")

            assert blocks is not None
            assert any(isinstance(b, TextContent) for b in blocks)
            # Default timeout is used
            mock_page.wait_for_selector.assert_called_once()

    @pytest.mark.asyncio
    async def test_playwright_tool_cleanup(self):
        """Test cleanup functionality."""
        tool = PlaywrightTool()

        # Mock browser and context
        mock_browser = AsyncMock()
        mock_context = AsyncMock()
        mock_page = AsyncMock()

        tool._browser = mock_browser
        tool._browser_context = mock_context
        tool.page = mock_page

        # Call the cleanup method directly (tool is not a context manager)
        await tool.close()

        mock_browser.close.assert_called_once()
        assert tool._browser is None
        assert tool._browser_context is None
        assert tool.page is None
