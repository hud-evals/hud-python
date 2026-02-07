"""Tests for GLMComputerTool."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from mcp import McpError
from mcp.types import ImageContent, TextContent

from hud.tools.computer.glm import GLM_COORDINATE_SPACE, GLMComputerTool
from hud.tools.executors.base import BaseExecutor
from hud.tools.types import ContentResult


@pytest.fixture
def base_executor() -> BaseExecutor:
    """Create a BaseExecutor for testing."""
    return BaseExecutor()


@pytest.fixture
def glm_tool(base_executor: BaseExecutor) -> GLMComputerTool:
    """Create a GLMComputerTool with a base executor."""
    return GLMComputerTool(executor=base_executor)


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


class TestGLMComputerToolInit:
    """Test GLMComputerTool initialization and configuration."""

    def test_default_init(self, base_executor: BaseExecutor) -> None:
        tool = GLMComputerTool(executor=base_executor)
        assert tool.name == "glm_computer"
        assert tool.title == "GLM Computer Tool"
        assert tool.executor is base_executor

    def test_custom_dimensions(self, base_executor: BaseExecutor) -> None:
        tool = GLMComputerTool(executor=base_executor, width=1024, height=768)
        assert tool.width == 1024
        assert tool.height == 768

    def test_rescale_flag(self, base_executor: BaseExecutor) -> None:
        tool = GLMComputerTool(executor=base_executor, rescale_images=True)
        assert tool.rescale_images is True
        tool2 = GLMComputerTool(executor=base_executor, rescale_images=False)
        assert tool2.rescale_images is False

    def test_custom_name(self, base_executor: BaseExecutor) -> None:
        tool = GLMComputerTool(
            executor=base_executor,
            name="my_glm",
            title="My GLM",
            description="Custom GLM",
        )
        assert tool.name == "my_glm"
        assert tool.title == "My GLM"
        assert tool.description == "Custom GLM"


# ---------------------------------------------------------------------------
# _parse_box
# ---------------------------------------------------------------------------


class TestParseBox:
    """Test _parse_box parsing logic."""

    def test_string_format(self, glm_tool: GLMComputerTool) -> None:
        assert glm_tool._parse_box("[500, 300]") == (500, 300)

    def test_string_no_brackets(self, glm_tool: GLMComputerTool) -> None:
        assert glm_tool._parse_box("500, 300") == (500, 300)

    def test_string_tight(self, glm_tool: GLMComputerTool) -> None:
        assert glm_tool._parse_box("[500,300]") == (500, 300)

    def test_list_format(self, glm_tool: GLMComputerTool) -> None:
        assert glm_tool._parse_box([500, 300]) == (500, 300)

    def test_nested_list(self, glm_tool: GLMComputerTool) -> None:
        assert glm_tool._parse_box([[500, 300]]) == (500, 300)

    def test_none(self, glm_tool: GLMComputerTool) -> None:
        assert glm_tool._parse_box(None) is None

    def test_invalid_string(self, glm_tool: GLMComputerTool) -> None:
        assert glm_tool._parse_box("invalid") is None

    def test_empty_list(self, glm_tool: GLMComputerTool) -> None:
        assert glm_tool._parse_box([]) is None


# ---------------------------------------------------------------------------
# _scale_coord
# ---------------------------------------------------------------------------


class TestScaleCoord:
    """Test coordinate scaling from 0-999 to screen pixels."""

    def test_origin(self, glm_tool: GLMComputerTool) -> None:
        assert glm_tool._scale_coord(0, is_x=True) == 0
        assert glm_tool._scale_coord(0, is_x=False) == 0

    def test_max_coord(self, glm_tool: GLMComputerTool) -> None:
        # 999 should map to dimension-1 (last valid pixel index)
        x = glm_tool._scale_coord(999, is_x=True)
        y = glm_tool._scale_coord(999, is_x=False)
        assert x == int(999 * (glm_tool.environment_width - 1) / GLM_COORDINATE_SPACE)
        assert y == int(999 * (glm_tool.environment_height - 1) / GLM_COORDINATE_SPACE)
        # Must never exceed dimension-1
        assert x <= glm_tool.environment_width - 1
        assert y <= glm_tool.environment_height - 1

    def test_midpoint(self, glm_tool: GLMComputerTool) -> None:
        x = glm_tool._scale_coord(500, is_x=True)
        expected = int(500 * (glm_tool.environment_width - 1) / GLM_COORDINATE_SPACE)
        assert x == expected

    def test_custom_env_dimensions(self, base_executor: BaseExecutor) -> None:
        """Scaling should use environment_width/height, not agent width/height."""
        tool = GLMComputerTool(executor=base_executor, width=1024, height=768)
        x = tool._scale_coord(500, is_x=True)
        y = tool._scale_coord(500, is_x=False)
        # Environment dimensions default to 1920x1080
        assert x == int(500 * (tool.environment_width - 1) / GLM_COORDINATE_SPACE)
        assert y == int(500 * (tool.environment_height - 1) / GLM_COORDINATE_SPACE)


# ---------------------------------------------------------------------------
# _parse_keys
# ---------------------------------------------------------------------------


class TestParseKeys:
    """Test _parse_keys helper."""

    def test_string_combo(self, glm_tool: GLMComputerTool) -> None:
        assert glm_tool._parse_keys("ctrl+c") == ["ctrl", "c"]

    def test_single_key(self, glm_tool: GLMComputerTool) -> None:
        assert glm_tool._parse_keys("enter") == ["enter"]

    def test_list_input(self, glm_tool: GLMComputerTool) -> None:
        assert glm_tool._parse_keys(["Ctrl", "A"]) == ["ctrl", "a"]

    def test_none(self, glm_tool: GLMComputerTool) -> None:
        assert glm_tool._parse_keys(None) == []

    def test_empty_string(self, glm_tool: GLMComputerTool) -> None:
        assert glm_tool._parse_keys("") == []


# ---------------------------------------------------------------------------
# __call__ - click actions
# ---------------------------------------------------------------------------


class TestGLMClickActions:
    """Test click-related actions."""

    @pytest.mark.asyncio
    async def test_left_click(self, glm_tool: GLMComputerTool) -> None:
        blocks = await glm_tool(action="left_click", start_box="[500, 300]")
        assert blocks
        assert all(isinstance(b, (ImageContent, TextContent)) for b in blocks)

    @pytest.mark.asyncio
    async def test_click_alias(self, glm_tool: GLMComputerTool) -> None:
        blocks = await glm_tool(action="click", start_box="[500, 300]")
        assert blocks

    @pytest.mark.asyncio
    async def test_right_click(self, glm_tool: GLMComputerTool) -> None:
        blocks = await glm_tool(action="right_click", start_box="[500, 300]")
        assert blocks

    @pytest.mark.asyncio
    async def test_middle_click(self, glm_tool: GLMComputerTool) -> None:
        blocks = await glm_tool(action="middle_click", start_box="[500, 300]")
        assert blocks

    @pytest.mark.asyncio
    async def test_left_double_click(self, glm_tool: GLMComputerTool) -> None:
        blocks = await glm_tool(action="left_double_click", start_box="[500, 300]")
        assert blocks

    @pytest.mark.asyncio
    async def test_left_click_missing_start_box(self, glm_tool: GLMComputerTool) -> None:
        with pytest.raises(McpError):
            await glm_tool(action="left_click")

    @pytest.mark.asyncio
    async def test_right_click_missing_start_box(self, glm_tool: GLMComputerTool) -> None:
        with pytest.raises(McpError):
            await glm_tool(action="right_click")


# ---------------------------------------------------------------------------
# __call__ - hover and drag
# ---------------------------------------------------------------------------


class TestGLMHoverDrag:
    """Test hover and drag actions."""

    @pytest.mark.asyncio
    async def test_hover(self, glm_tool: GLMComputerTool) -> None:
        blocks = await glm_tool(action="hover", start_box="[100, 200]")
        assert blocks

    @pytest.mark.asyncio
    async def test_hover_missing_start_box(self, glm_tool: GLMComputerTool) -> None:
        with pytest.raises(McpError):
            await glm_tool(action="hover")

    @pytest.mark.asyncio
    async def test_left_drag(self, glm_tool: GLMComputerTool) -> None:
        blocks = await glm_tool(
            action="left_drag",
            start_box="[100, 100]",
            end_box="[500, 500]",
        )
        assert blocks

    @pytest.mark.asyncio
    async def test_left_drag_missing_end_box(self, glm_tool: GLMComputerTool) -> None:
        with pytest.raises(McpError):
            await glm_tool(action="left_drag", start_box="[100, 100]")

    @pytest.mark.asyncio
    async def test_left_drag_missing_start_box(self, glm_tool: GLMComputerTool) -> None:
        with pytest.raises(McpError):
            await glm_tool(action="left_drag", end_box="[500, 500]")


# ---------------------------------------------------------------------------
# __call__ - keyboard actions
# ---------------------------------------------------------------------------


class TestGLMKeyboardActions:
    """Test keyboard-related actions."""

    @pytest.mark.asyncio
    async def test_key_combo(self, glm_tool: GLMComputerTool) -> None:
        blocks = await glm_tool(action="key", keys="ctrl+c")
        assert blocks

    @pytest.mark.asyncio
    async def test_key_single(self, glm_tool: GLMComputerTool) -> None:
        blocks = await glm_tool(action="key", keys="enter")
        assert blocks

    @pytest.mark.asyncio
    async def test_key_missing_keys(self, glm_tool: GLMComputerTool) -> None:
        with pytest.raises(McpError):
            await glm_tool(action="key", keys=None)

    @pytest.mark.asyncio
    async def test_type_text(self, glm_tool: GLMComputerTool) -> None:
        blocks = await glm_tool(action="type", content="Hello World")
        assert blocks

    @pytest.mark.asyncio
    async def test_type_missing_content(self, glm_tool: GLMComputerTool) -> None:
        with pytest.raises(McpError):
            await glm_tool(action="type", content=None)


# ---------------------------------------------------------------------------
# __call__ - scroll
# ---------------------------------------------------------------------------


class TestGLMScrollAction:
    """Test scroll action."""

    @pytest.mark.asyncio
    async def test_scroll_down(self, glm_tool: GLMComputerTool) -> None:
        blocks = await glm_tool(action="scroll", start_box="[500, 500]", direction="down", step=3)
        assert blocks

    @pytest.mark.asyncio
    async def test_scroll_up(self, glm_tool: GLMComputerTool) -> None:
        blocks = await glm_tool(action="scroll", start_box="[500, 500]", direction="up", step=5)
        assert blocks

    @pytest.mark.asyncio
    async def test_scroll_no_start_box_defaults_to_center(self, glm_tool: GLMComputerTool) -> None:
        """Scroll without start_box should use screen center."""
        # Must pass step explicitly because Field(...) default doesn't resolve
        # when __call__ is invoked directly (not through MCP).
        blocks = await glm_tool(action="scroll", direction="down", step=5)
        assert blocks

    @pytest.mark.asyncio
    async def test_scroll_missing_direction(self, glm_tool: GLMComputerTool) -> None:
        with pytest.raises(McpError):
            await glm_tool(action="scroll", start_box="[500, 500]", direction=None, step=5)


# ---------------------------------------------------------------------------
# __call__ - screenshot
# ---------------------------------------------------------------------------


class TestGLMScreenshotAction:
    """Test screenshot action."""

    @pytest.mark.asyncio
    async def test_screenshot(self, base_executor: BaseExecutor) -> None:
        tool = GLMComputerTool(executor=base_executor)
        base_executor.screenshot = AsyncMock(return_value="fake_base64_data")

        blocks = await tool(action="screenshot")
        assert blocks
        assert any(isinstance(b, ImageContent) for b in blocks)

    @pytest.mark.asyncio
    async def test_screenshot_failure(self, base_executor: BaseExecutor) -> None:
        tool = GLMComputerTool(executor=base_executor)
        base_executor.screenshot = AsyncMock(return_value=None)

        blocks = await tool(action="screenshot")
        assert blocks
        assert any(isinstance(b, TextContent) and "Failed" in b.text for b in blocks)

    @pytest.mark.asyncio
    async def test_screenshot_rescaling(self, base_executor: BaseExecutor) -> None:
        tool = GLMComputerTool(executor=base_executor, width=1024, height=768, rescale_images=True)
        base_executor.screenshot = AsyncMock(return_value="fake_base64_data")
        tool._rescale_screenshot = AsyncMock(return_value="rescaled_base64_data")

        blocks = await tool(action="screenshot")
        assert blocks
        tool._rescale_screenshot.assert_called_with("fake_base64_data")


# ---------------------------------------------------------------------------
# __call__ - control actions
# ---------------------------------------------------------------------------


class TestGLMControlActions:
    """Test WAIT, DONE, FAIL actions."""

    @pytest.mark.asyncio
    async def test_wait(self, glm_tool: GLMComputerTool) -> None:
        blocks = await glm_tool(action="WAIT")
        assert blocks

    @pytest.mark.asyncio
    async def test_done(self, base_executor: BaseExecutor) -> None:
        tool = GLMComputerTool(executor=base_executor)
        base_executor.screenshot = AsyncMock(return_value="fake_base64")

        blocks = await tool(action="DONE")
        assert blocks
        assert any(isinstance(b, TextContent) and "completed" in b.text.lower() for b in blocks)

    @pytest.mark.asyncio
    async def test_fail(self, base_executor: BaseExecutor) -> None:
        tool = GLMComputerTool(executor=base_executor)
        base_executor.screenshot = AsyncMock(return_value="fake_base64")

        blocks = await tool(action="FAIL")
        assert blocks
        assert any(isinstance(b, TextContent) and "failed" in b.text.lower() for b in blocks)


# ---------------------------------------------------------------------------
# __call__ - unknown action
# ---------------------------------------------------------------------------


class TestGLMUnknownAction:
    """Test error handling for unknown actions."""

    @pytest.mark.asyncio
    async def test_unknown_action(self, glm_tool: GLMComputerTool) -> None:
        with pytest.raises(McpError):
            await glm_tool(action="nonexistent_action")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Auto-screenshot for interactive actions
# ---------------------------------------------------------------------------


class TestGLMAutoScreenshot:
    """Test that interactive actions include a screenshot in the result."""

    @pytest.mark.asyncio
    async def test_interactive_action_includes_screenshot(
        self, base_executor: BaseExecutor
    ) -> None:
        tool = GLMComputerTool(executor=base_executor)
        # Mock executor.click to return a result without a screenshot
        base_executor.click = AsyncMock(return_value=ContentResult(output="Clicked"))
        # Mock screenshot so the auto-screenshot fallback works
        base_executor.screenshot = AsyncMock(return_value="auto_screenshot_base64")

        blocks = await tool(action="left_click", start_box="[500, 300]")
        assert blocks
        assert any(isinstance(b, ImageContent) for b in blocks)

    @pytest.mark.asyncio
    async def test_interactive_action_with_existing_screenshot(
        self, base_executor: BaseExecutor
    ) -> None:
        """If executor already returns a screenshot, auto-screenshot should not override."""
        tool = GLMComputerTool(executor=base_executor)
        base_executor.click = AsyncMock(
            return_value=ContentResult(base64_image="existing_screenshot")
        )

        blocks = await tool(action="left_click", start_box="[500, 300]")
        assert blocks
        # Should have an image block
        assert any(isinstance(b, ImageContent) for b in blocks)
