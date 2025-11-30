from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from mcp.types import ImageContent, TextContent

from hud.tools.computer.anthropic import AnthropicComputerTool
from hud.tools.computer.hud import HudComputerTool
from hud.tools.computer.openai import OpenAIComputerTool
from hud.tools.executors.base import BaseExecutor


@pytest.mark.asyncio
async def test_hud_computer_screenshot():
    comp = HudComputerTool()
    blocks = await comp(action="screenshot")
    # Screenshot might return ImageContent or TextContent (if error)
    assert blocks is not None
    assert len(blocks) > 0
    assert all(isinstance(b, (ImageContent | TextContent)) for b in blocks)


@pytest.mark.asyncio
async def test_hud_computer_click_simulation():
    comp = HudComputerTool()
    blocks = await comp(action="click", x=10, y=10)
    # Should return text confirming execution or screenshot block
    assert blocks
    assert len(blocks) > 0


@pytest.mark.asyncio
async def test_openai_computer_screenshot():
    comp = OpenAIComputerTool()
    blocks = await comp(type="screenshot")
    assert blocks is not None
    assert len(blocks) > 0
    assert all(isinstance(b, (ImageContent | TextContent)) for b in blocks)


@pytest.mark.asyncio
async def test_anthropic_computer_screenshot():
    comp = AnthropicComputerTool()
    blocks = await comp(action="screenshot")
    assert blocks is not None
    assert len(blocks) > 0
    assert all(isinstance(b, (ImageContent | TextContent)) for b in blocks)


@pytest.mark.asyncio
async def test_openai_computer_click():
    comp = OpenAIComputerTool()
    blocks = await comp(type="click", x=5, y=5)
    assert blocks


class TestHudComputerToolExtended:
    """Extended tests for HudComputerTool covering edge cases and platform logic."""

    @pytest.fixture
    def base_executor(self):
        """Create a BaseExecutor instance for testing."""
        return BaseExecutor()

    @pytest.mark.asyncio
    async def test_explicit_base_executor(self, base_executor):
        """Test explicitly using BaseExecutor."""
        tool = HudComputerTool(executor=base_executor)
        assert tool.executor is base_executor

        # Test that actions work with base executor
        result = await tool(action="click", x=100, y=200)
        assert result
        assert any(
            "[SIMULATED]" in content.text for content in result if isinstance(content, TextContent)
        )

    @pytest.mark.asyncio
    async def test_platform_auto_selection_linux(self):
        """Test platform auto-selection on Linux."""
        with (
            patch("platform.system", return_value="Linux"),
            patch("hud.tools.executors.xdo.XDOExecutor.is_available", return_value=False),
            patch(
                "hud.tools.executors.pyautogui.PyAutoGUIExecutor.is_available",
                return_value=False,
            ),
        ):
            tool = HudComputerTool()
            assert isinstance(tool.executor, BaseExecutor)

    @pytest.mark.asyncio
    async def test_platform_auto_selection_windows(self):
        """Test platform auto-selection on Windows."""
        with (
            patch("platform.system", return_value="Windows"),
            patch(
                "hud.tools.executors.pyautogui.PyAutoGUIExecutor.is_available", return_value=False
            ),
        ):
            tool = HudComputerTool()
            assert isinstance(tool.executor, BaseExecutor)

    @pytest.mark.asyncio
    async def test_platform_xdo_fallback(self):
        """Test XDO platform fallback to BaseExecutor."""
        with patch("hud.tools.executors.xdo.XDOExecutor.is_available", return_value=False):
            tool = HudComputerTool(platform_type="xdo")
            assert isinstance(tool.executor, BaseExecutor)

    @pytest.mark.asyncio
    async def test_platform_pyautogui_fallback(self):
        """Test PyAutoGUI platform fallback to BaseExecutor."""
        with patch(
            "hud.tools.executors.pyautogui.PyAutoGUIExecutor.is_available", return_value=False
        ):
            tool = HudComputerTool(platform_type="pyautogui")
            assert isinstance(tool.executor, BaseExecutor)

    @pytest.mark.asyncio
    async def test_invalid_platform_type(self):
        """Test invalid platform type raises ValueError."""
        with pytest.raises(ValueError, match="Invalid platform_type"):
            HudComputerTool(platform_type="invalid_platform")  # type: ignore[arg-type]

    @pytest.mark.asyncio
    async def test_coordinate_scaling(self, base_executor):
        """Test coordinate scaling with different screen sizes."""
        # Test with custom dimensions that require scaling
        tool = HudComputerTool(executor=base_executor, width=800, height=600)

        # Test click with scaling
        result = await tool(action="click", x=400, y=300)
        assert result

        # Test that coordinates are scaled properly
        assert tool.scale_x == 800 / 1920  # Default environment width is 1920
        assert tool.scale_y == 600 / 1080  # Default environment height is 1080
        assert tool.needs_scaling is True

    @pytest.mark.asyncio
    async def test_no_scaling_needed(self, base_executor):
        """Test when no scaling is needed."""
        tool = HudComputerTool(executor=base_executor, width=1920, height=1080)
        assert tool.needs_scaling is False
        assert tool.scale_x == 1.0
        assert tool.scale_y == 1.0

    @pytest.mark.asyncio
    async def test_type_action(self, base_executor):
        """Test type action with BaseExecutor."""
        tool = HudComputerTool(executor=base_executor)
        result = await tool(action="write", text="Hello World", enter_after=True)
        assert result
        assert any(
            "[SIMULATED] Type" in content.text
            for content in result
            if isinstance(content, TextContent)
        )

    @pytest.mark.asyncio
    async def test_press_action(self, base_executor):
        """Test press action with BaseExecutor."""
        tool = HudComputerTool(executor=base_executor)
        result = await tool(action="press", keys=["ctrl", "c"])
        assert result
        assert any(
            "[SIMULATED] Press" in content.text
            for content in result
            if isinstance(content, TextContent)
        )

    @pytest.mark.asyncio
    async def test_scroll_action(self, base_executor):
        """Test scroll action with BaseExecutor."""
        tool = HudComputerTool(executor=base_executor)
        result = await tool(action="scroll", x=500, y=500, scroll_x=0, scroll_y=5)
        assert result
        assert any(
            "Scroll" in content.text for content in result if isinstance(content, TextContent)
        )

    @pytest.mark.asyncio
    async def test_move_action(self, base_executor):
        """Test move action with BaseExecutor."""
        tool = HudComputerTool(executor=base_executor)
        result = await tool(action="move", x=100, y=100)
        assert result
        assert any("Move" in content.text for content in result if isinstance(content, TextContent))

    @pytest.mark.asyncio
    async def test_drag_action(self, base_executor):
        """Test drag action with BaseExecutor."""
        tool = HudComputerTool(executor=base_executor)
        result = await tool(action="drag", path=[(100, 100), (200, 200)])
        assert result
        assert any("Drag" in content.text for content in result if isinstance(content, TextContent))

    @pytest.mark.asyncio
    async def test_wait_action(self, base_executor):
        """Test wait action with BaseExecutor."""
        tool = HudComputerTool(executor=base_executor)
        result = await tool(action="wait", time=100)  # 100ms for quick test
        assert result
        assert any("Wait" in content.text for content in result if isinstance(content, TextContent))

    @pytest.mark.asyncio
    async def test_keydown_keyup_actions(self, base_executor):
        """Test keydown and keyup actions with BaseExecutor."""
        tool = HudComputerTool(executor=base_executor)

        # Test keydown
        result = await tool(action="keydown", keys=["shift"])
        assert result

        # Test keyup
        result = await tool(action="keyup", keys=["shift"])
        assert result

    @pytest.mark.asyncio
    async def test_hold_key_action(self, base_executor):
        """Test hold_key action with BaseExecutor."""
        tool = HudComputerTool(executor=base_executor)
        result = await tool(action="hold_key", text="a", duration=0.1)
        assert result

    @pytest.mark.asyncio
    async def test_mouse_down_up_actions(self, base_executor):
        """Test mouse_down and mouse_up actions with BaseExecutor."""
        tool = HudComputerTool(executor=base_executor)

        # Test mouse_down
        result = await tool(action="mouse_down", button="left")
        assert result

        # Test mouse_up
        result = await tool(action="mouse_up", button="left")
        assert result

    @pytest.mark.asyncio
    async def test_position_action(self, base_executor):
        """Test position action with BaseExecutor."""
        tool = HudComputerTool(executor=base_executor)
        result = await tool(action="position")
        assert result

    @pytest.mark.asyncio
    async def test_response_action(self, base_executor):
        """Test response action."""
        tool = HudComputerTool(executor=base_executor)
        result = await tool(action="response", text="Test response")
        assert result
        assert len(result) == 1
        assert isinstance(result[0], TextContent)
        assert result[0].text == "Test response"

    @pytest.mark.asyncio
    async def test_click_with_different_buttons(self, base_executor):
        """Test click with different mouse buttons."""
        tool = HudComputerTool(executor=base_executor)

        # Right click
        result = await tool(action="click", x=100, y=100, button="right")
        assert result

        # Middle click
        result = await tool(action="click", x=100, y=100, button="middle")
        assert result

        # Double click (using pattern)
        result = await tool(action="click", x=100, y=100, pattern=[100])
        assert result

    @pytest.mark.asyncio
    async def test_screenshot_action(self, base_executor):
        """Test screenshot action."""
        tool = HudComputerTool(executor=base_executor)

        # Mock the screenshot method
        base_executor.screenshot = AsyncMock(return_value="fake_base64_data")

        result = await tool(action="screenshot")
        assert result
        assert any(isinstance(content, ImageContent) for content in result)

    @pytest.mark.asyncio
    async def test_screenshot_rescaling(self, base_executor):
        """Test screenshot rescaling functionality."""
        tool = HudComputerTool(executor=base_executor, width=800, height=600, rescale_images=True)

        # Mock the screenshot method
        base_executor.screenshot = AsyncMock(return_value="fake_base64_data")

        # Mock the rescale method
        tool._rescale_screenshot = AsyncMock(return_value="rescaled_base64_data")

        result = await tool(action="screenshot")
        assert result
        # The rescale method is called twice - once for the screenshot action,
        # and once when processing the result
        assert tool._rescale_screenshot.call_count == 2
        tool._rescale_screenshot.assert_any_call("fake_base64_data")

    @pytest.mark.asyncio
    async def test_executor_initialization_with_display_num(self):
        """Test executor initialization with display number."""
        with patch(
            "hud.tools.executors.pyautogui.PyAutoGUIExecutor.is_available", return_value=False
        ):
            tool = HudComputerTool(display_num=1)
            assert tool.display_num == 1

    @pytest.mark.asyncio
    async def test_coordinate_none_values(self, base_executor):
        """Test actions with None coordinate values."""
        tool = HudComputerTool(executor=base_executor)

        # Test press without coordinates (keyboard shortcut)
        result = await tool(action="press", keys=["ctrl", "a"])
        assert result

        # Test type without coordinates
        result = await tool(action="write", text="test")
        assert result

    @pytest.mark.asyncio
    async def test_tool_metadata(self, base_executor):
        """Test tool metadata is set correctly."""
        tool = HudComputerTool(
            executor=base_executor,
            name="custom_computer",
            title="Custom Computer Tool",
            description="Custom description",
        )
        assert tool.name == "custom_computer"
        assert tool.title == "Custom Computer Tool"
        assert tool.description == "Custom description"

        # Test defaults
        default_tool = HudComputerTool(executor=base_executor)
        assert default_tool.name == "computer"
        assert default_tool.title == "Computer Control"
        assert default_tool.description == "Control computer with mouse, keyboard, and screenshots"

    @pytest.mark.asyncio
    async def test_missing_required_parameters(self, base_executor):
        """Test actions that are missing required parameters."""
        tool = HudComputerTool(executor=base_executor)

        # Test type without text
        from hud.tools.types import ToolError

        with pytest.raises(ToolError, match="text parameter is required"):
            await tool(action="write", text=None)

        # Test press without keys
        with pytest.raises(ToolError, match="keys parameter is required"):
            await tool(action="press", keys=None)

        # Test wait without time
        with pytest.raises(ToolError, match="time parameter is required"):
            await tool(action="wait", time=None)

        # Test drag without path
        with pytest.raises(ToolError, match="path parameter is required"):
            await tool(action="drag", path=None)

    @pytest.mark.asyncio
    async def test_relative_move(self, base_executor):
        """Test relative move with offsets."""
        tool = HudComputerTool(executor=base_executor)
        result = await tool(action="move", offset_x=50, offset_y=50)
        assert result

    @pytest.mark.asyncio
    async def test_screenshot_failure(self, base_executor):
        """Test screenshot failure handling."""
        tool = HudComputerTool(executor=base_executor)

        # Mock screenshot to return None (failure)
        base_executor.screenshot = AsyncMock(return_value=None)

        result = await tool(action="screenshot")
        assert result
        # Should contain error message
        assert any(
            "Failed" in content.text for content in result if isinstance(content, TextContent)
        )

    @pytest.mark.asyncio
    async def test_platform_selection_with_available_executors(self):
        """Test platform selection when executors are available."""
        # Test Linux with XDO available
        mock_xdo_instance = MagicMock()
        with (
            patch("platform.system", return_value="Linux"),
            patch("hud.tools.executors.xdo.XDOExecutor.is_available", return_value=True),
            patch("hud.tools.computer.hud.XDOExecutor", return_value=mock_xdo_instance) as mock_xdo,
        ):
            tool = HudComputerTool(platform_type="auto")
            mock_xdo.assert_called_once()
            assert tool.executor is mock_xdo_instance

        # Test with PyAutoGUI available
        mock_pyautogui_instance = MagicMock()
        with (
            patch(
                "hud.tools.executors.pyautogui.PyAutoGUIExecutor.is_available", return_value=True
            ),
            patch(
                "hud.tools.computer.hud.PyAutoGUIExecutor", return_value=mock_pyautogui_instance
            ) as mock_pyautogui,
        ):
            tool = HudComputerTool(platform_type="pyautogui")
            mock_pyautogui.assert_called_once()
            assert tool.executor is mock_pyautogui_instance
