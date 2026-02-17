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
# _fix_xml_args (moved from GLMCUAAgent)
# ---------------------------------------------------------------------------


class TestFixXMLArgs:
    """Test _fix_xml_args static method for handling GLM's XML-style output."""

    def test_clean_json_passthrough(self) -> None:
        """Clean JSON args should pass through unchanged."""
        args = {"action": "left_click", "start_box": "[500, 300]"}
        assert GLMComputerTool._fix_xml_args(args) == args

    def test_non_string_passthrough(self) -> None:
        """Non-string values should pass through unchanged."""
        args = {"action": "scroll", "step": 5}
        assert GLMComputerTool._fix_xml_args(args) == args

    def test_mixed_json_xml(self) -> None:
        """Mixed JSON/XML format: action value contains XML tags."""
        args = {"action": "left_click\n<arg_key>start_box</arg_key>\n<arg_value>[114, 167]"}
        result = GLMComputerTool._fix_xml_args(args)
        assert result["action"] == "left_click"
        assert result["start_box"] == "[114, 167]"

    def test_pure_xml_no_prefix(self) -> None:
        """Value starts directly with XML tag (no plain text prefix)."""
        args = {"action": "<arg_key>action</arg_key><arg_value>left_click"}
        result = GLMComputerTool._fix_xml_args(args)
        assert result["action"] == "left_click"

    def test_preserves_key_when_no_xml_match(self) -> None:
        """Original key preserved when no XML content found."""
        args = {"action": "<arg_key>unknown</arg_key>"}
        result = GLMComputerTool._fix_xml_args(args)
        # Original key should be preserved
        assert "action" in result

    def test_multiple_xml_pairs(self) -> None:
        """Multiple XML key-value pairs extracted correctly."""
        args = {
            "action": "left_click\n"
            "<arg_key>start_box</arg_key>\n<arg_value>[100, 200]\n"
            "<arg_key>element_info</arg_key>\n<arg_value>button"
        }
        result = GLMComputerTool._fix_xml_args(args)
        assert result["action"] == "left_click"
        assert result["start_box"] == "[100, 200]"
        assert result["element_info"] == "button"


# ---------------------------------------------------------------------------
# __call__ - XML arg fixing in __call__
# ---------------------------------------------------------------------------


class TestGLMXMLArgFixingInCall:
    """Test that __call__ fixes XML-mangled arguments inline."""

    @pytest.mark.asyncio
    async def test_xml_action_is_fixed(self, glm_tool: GLMComputerTool) -> None:
        """XML-mangled action string should be fixed and executed."""
        blocks = await glm_tool(
            action="left_click\n<arg_key>start_box</arg_key>\n<arg_value>[500, 300]",  # type: ignore[arg-type]
        )
        assert blocks
        assert all(isinstance(b, ImageContent | TextContent) for b in blocks)


# ---------------------------------------------------------------------------
# __call__ - validation
# ---------------------------------------------------------------------------


class TestGLMCallValidation:
    """Test __call__ parameter validation."""

    @pytest.mark.asyncio
    async def test_missing_action(self, glm_tool: GLMComputerTool) -> None:
        with pytest.raises(McpError):
            await glm_tool(action=None)

    @pytest.mark.asyncio
    async def test_unknown_action(self, glm_tool: GLMComputerTool) -> None:
        with pytest.raises(McpError):
            await glm_tool(action="nonexistent_action")  # type: ignore[arg-type]

    @pytest.mark.asyncio
    async def test_click_missing_start_box(self, glm_tool: GLMComputerTool) -> None:
        with pytest.raises(McpError):
            await glm_tool(action="left_click")

    @pytest.mark.asyncio
    async def test_drag_missing_end_box(self, glm_tool: GLMComputerTool) -> None:
        with pytest.raises(McpError):
            await glm_tool(action="left_drag", start_box="[100, 100]")

    @pytest.mark.asyncio
    async def test_key_missing_keys(self, glm_tool: GLMComputerTool) -> None:
        with pytest.raises(McpError):
            await glm_tool(action="key", keys=None)

    @pytest.mark.asyncio
    async def test_type_missing_content(self, glm_tool: GLMComputerTool) -> None:
        with pytest.raises(McpError):
            await glm_tool(action="type", content=None)

    @pytest.mark.asyncio
    async def test_scroll_missing_direction(self, glm_tool: GLMComputerTool) -> None:
        with pytest.raises(McpError):
            await glm_tool(action="scroll", start_box="[500, 500]", direction=None, step=5)

    @pytest.mark.asyncio
    async def test_done_raises_mcp_error(self, glm_tool: GLMComputerTool) -> None:
        with pytest.raises(McpError, match="DONE action is not supported"):
            await glm_tool(action="DONE")

    @pytest.mark.asyncio
    async def test_fail_raises_mcp_error(self, glm_tool: GLMComputerTool) -> None:
        with pytest.raises(McpError, match="FAIL action is not supported"):
            await glm_tool(action="FAIL")


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
