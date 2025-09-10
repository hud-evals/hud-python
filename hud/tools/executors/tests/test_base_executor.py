"""Tests for BaseExecutor."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from hud.tools.executors.base import BaseExecutor
from hud.tools.types import ContentResult


class TestBaseExecutor:
    """Tests for BaseExecutor simulated actions."""

    def test_init(self):
        """Test BaseExecutor initialization."""
        # Without display num - defaults to computer_settings.DISPLAY_NUM
        executor = BaseExecutor()
        assert executor.display_num == 0  # Default from computer_settings
        assert executor._screenshot_delay == 0.5

        # With display num
        executor = BaseExecutor(display_num=1)
        assert executor.display_num == 1

    @pytest.mark.asyncio
    async def test_click_basic(self):
        """Test basic click action."""
        executor = BaseExecutor()
        result = await executor.click(x=100, y=200, button="left", take_screenshot=False)

        assert isinstance(result, ContentResult)
        assert result.output == "[SIMULATED] Click at (100, 200) with left button"
        assert result.base64_image is None  # No screenshot requested

    @pytest.mark.asyncio
    async def test_click_with_screenshot(self):
        """Test click with screenshot."""
        executor = BaseExecutor()
        result = await executor.click(x=100, y=200, take_screenshot=True)

        assert isinstance(result, ContentResult)
        assert result.output == "[SIMULATED] Click at (100, 200) with left button"
        assert result.base64_image is not None  # Screenshot included

    @pytest.mark.asyncio
    async def test_click_with_pattern(self):
        """Test click with multi-click pattern."""
        executor = BaseExecutor()
        result = await executor.click(x=100, y=200, pattern=[100, 50], take_screenshot=False)

        assert isinstance(result, ContentResult)
        assert result.output is not None
        assert (
            "[SIMULATED] Click at (100, 200) with left button (multi-click pattern: [100, 50])"
            in result.output
        )

    @pytest.mark.asyncio
    async def test_click_with_hold_keys(self):
        """Test click while holding keys."""
        executor = BaseExecutor()
        result = await executor.click(
            x=100, y=200, hold_keys=["ctrl", "shift"], take_screenshot=False
        )

        assert isinstance(result, ContentResult)
        assert result.output is not None
        assert "while holding ['ctrl', 'shift']" in result.output

    @pytest.mark.asyncio
    async def test_type_basic(self):
        """Test basic typing."""
        executor = BaseExecutor()
        result = await executor.write("Hello World", take_screenshot=False)

        assert isinstance(result, ContentResult)
        assert result.output == "[SIMULATED] Type 'Hello World'"

    @pytest.mark.asyncio
    async def test_type_with_enter(self):
        """Test typing with enter."""
        executor = BaseExecutor()
        result = await executor.write("Hello", enter_after=True, take_screenshot=False)

        assert isinstance(result, ContentResult)
        assert result.output == "[SIMULATED] Type 'Hello' followed by Enter"

    @pytest.mark.asyncio
    async def test_press_keys(self):
        """Test pressing key combination."""
        executor = BaseExecutor()
        result = await executor.press(["ctrl", "c"], take_screenshot=False)

        assert isinstance(result, ContentResult)
        assert result.output == "[SIMULATED] Press key combination: ctrl+c"

    @pytest.mark.asyncio
    async def test_key_single(self):
        """Test pressing single key."""
        executor = BaseExecutor()
        result = await executor.key("Return", take_screenshot=False)

        assert isinstance(result, ContentResult)
        assert result.output == "[SIMULATED] Press key: Return"

    @pytest.mark.asyncio
    async def test_keydown(self):
        """Test key down action."""
        executor = BaseExecutor()
        result = await executor.keydown(["shift", "ctrl"], take_screenshot=False)

        assert isinstance(result, ContentResult)
        assert result.output == "[SIMULATED] Key down: shift, ctrl"

    @pytest.mark.asyncio
    async def test_keyup(self):
        """Test key up action."""
        executor = BaseExecutor()
        result = await executor.keyup(["shift", "ctrl"], take_screenshot=False)

        assert isinstance(result, ContentResult)
        assert result.output == "[SIMULATED] Key up: shift, ctrl"

    @pytest.mark.asyncio
    async def test_scroll_basic(self):
        """Test basic scroll."""
        executor = BaseExecutor()
        result = await executor.scroll(x=100, y=200, scroll_y=5, take_screenshot=False)

        assert isinstance(result, ContentResult)
        assert result.output is not None
        assert "[SIMULATED] Scroll at (100, 200)" in result.output
        assert "vertically by 5" in result.output

    @pytest.mark.asyncio
    async def test_scroll_horizontal(self):
        """Test horizontal scroll."""
        executor = BaseExecutor()
        result = await executor.scroll(scroll_x=10, take_screenshot=False)

        assert isinstance(result, ContentResult)
        assert result.output is not None
        assert "[SIMULATED] Scroll" in result.output
        assert "horizontally by 10" in result.output

    @pytest.mark.asyncio
    async def test_scroll_with_hold_keys(self):
        """Test scroll with held keys."""
        executor = BaseExecutor()
        result = await executor.scroll(
            x=100, y=200, scroll_y=5, hold_keys=["shift"], take_screenshot=False
        )

        assert isinstance(result, ContentResult)
        assert result.output is not None
        assert "while holding ['shift']" in result.output

    @pytest.mark.asyncio
    async def test_move_absolute(self):
        """Test absolute mouse movement."""
        executor = BaseExecutor()
        result = await executor.move(x=300, y=400, take_screenshot=False)

        assert isinstance(result, ContentResult)
        assert result.output == "[SIMULATED] Move mouse to (300, 400)"

    @pytest.mark.asyncio
    async def test_move_relative(self):
        """Test relative mouse movement."""
        executor = BaseExecutor()
        result = await executor.move(offset_x=50, offset_y=-30, take_screenshot=False)

        assert isinstance(result, ContentResult)
        assert result.output == "[SIMULATED] Move mouse by offset (50, -30)"

    @pytest.mark.asyncio
    async def test_move_no_coords(self):
        """Test move with no coordinates."""
        executor = BaseExecutor()
        result = await executor.move(take_screenshot=False)

        assert isinstance(result, ContentResult)
        assert result.output == "[SIMULATED] Move mouse (no coordinates specified)"

    @pytest.mark.asyncio
    async def test_drag_basic(self):
        """Test basic drag operation."""
        executor = BaseExecutor()
        path = [(100, 100), (200, 200)]
        result = await executor.drag(path, take_screenshot=False)

        assert isinstance(result, ContentResult)
        assert result.output == "[SIMULATED] Drag from (100, 100) to (200, 200)"

    @pytest.mark.asyncio
    async def test_drag_with_intermediate_points(self):
        """Test drag with intermediate points."""
        executor = BaseExecutor()
        path = [(100, 100), (150, 150), (200, 200)]
        result = await executor.drag(path, take_screenshot=False)

        assert isinstance(result, ContentResult)
        assert result.output is not None
        assert (
            "[SIMULATED] Drag from (100, 100) to (200, 200) via 1 intermediate points"
            in result.output
        )

    @pytest.mark.asyncio
    async def test_drag_invalid_path(self):
        """Test drag with invalid path."""
        executor = BaseExecutor()
        result = await executor.drag([(100, 100)], take_screenshot=False)  # Only one point

        assert isinstance(result, ContentResult)
        assert result.error == "Drag path must have at least 2 points"
        assert result.output is None

    @pytest.mark.asyncio
    async def test_drag_with_hold_keys(self):
        """Test drag with held keys."""
        executor = BaseExecutor()
        path = [(100, 100), (200, 200)]
        result = await executor.drag(path, hold_keys=["alt"], take_screenshot=False)

        assert isinstance(result, ContentResult)
        assert result.output is not None
        assert "while holding ['alt']" in result.output

    @pytest.mark.asyncio
    async def test_mouse_down(self):
        """Test mouse down action."""
        executor = BaseExecutor()
        result = await executor.mouse_down(button="right", take_screenshot=False)

        assert isinstance(result, ContentResult)
        assert result.output == "[SIMULATED] Mouse down: right button"

    @pytest.mark.asyncio
    async def test_mouse_up(self):
        """Test mouse up action."""
        executor = BaseExecutor()
        result = await executor.mouse_up(button="middle", take_screenshot=False)

        assert isinstance(result, ContentResult)
        assert result.output == "[SIMULATED] Mouse up: middle button"

    @pytest.mark.asyncio
    async def test_hold_key(self):
        """Test holding a key for duration."""
        executor = BaseExecutor()

        # Mock sleep to avoid actual wait
        with patch("asyncio.sleep") as mock_sleep:
            result = await executor.hold_key("shift", 0.5, take_screenshot=False)

            assert isinstance(result, ContentResult)
            assert result.output == "[SIMULATED] Hold key 'shift' for 0.5 seconds"
            mock_sleep.assert_called_once_with(0.5)

    @pytest.mark.asyncio
    async def test_wait(self):
        """Test wait action."""
        executor = BaseExecutor()

        # Mock sleep to avoid actual wait
        with patch("asyncio.sleep") as mock_sleep:
            result = await executor.wait(1000)  # 1000ms

            assert isinstance(result, ContentResult)
            assert result.output == "Waited 1000ms"
            mock_sleep.assert_called_once_with(1.0)

    @pytest.mark.asyncio
    async def test_screenshot(self):
        """Test screenshot action."""
        executor = BaseExecutor()
        result = await executor.screenshot()

        assert isinstance(result, str)
        # Check it's a valid base64 string (starts with PNG header)
        assert result.startswith("iVBORw0KGgo")

    @pytest.mark.asyncio
    async def test_position(self):
        """Test getting cursor position."""
        executor = BaseExecutor()
        result = await executor.position()

        assert isinstance(result, ContentResult)
        assert result.output == "[SIMULATED] Mouse position: (0, 0)"

    @pytest.mark.asyncio
    async def test_execute(self):
        """Test execute command."""
        executor = BaseExecutor()
        result = await executor.execute("custom command", take_screenshot=False)

        assert isinstance(result, ContentResult)
        assert result.output == "[SIMULATED] Execute: custom command"

    @pytest.mark.asyncio
    async def test_type_text_alias(self):
        """Test type_text alias method."""
        executor = BaseExecutor()
        result = await executor.write("test", delay=20, take_screenshot=False)

        assert isinstance(result, ContentResult)
        assert result.output == "[SIMULATED] Type 'test'"

    @pytest.mark.asyncio
    async def test_mouse_move_alias(self):
        """Test mouse_move alias method."""
        executor = BaseExecutor()
        result = await executor.mouse_move(100, 200, take_screenshot=False)

        assert isinstance(result, ContentResult)
        assert result.output == "[SIMULATED] Move mouse to (100, 200)"

    @pytest.mark.asyncio
    async def test_multiple_actions_with_screenshots(self):
        """Test multiple actions with screenshots to ensure consistency."""
        executor = BaseExecutor()

        # Test that screenshots are consistent
        screenshot1 = await executor.screenshot()
        screenshot2 = await executor.screenshot()

        assert screenshot1 == screenshot2  # Simulated screenshots should be identical

        # Test actions with screenshots
        result1 = await executor.click(10, 20, take_screenshot=True)
        result2 = await executor.write("test", take_screenshot=True)

        assert result1.base64_image == screenshot1
        assert result2.base64_image == screenshot1


class TestLazyImports:
    """Tests for lazy import functionality in executors module."""

    def test_lazy_import_pyautogui_executor(self):
        """Test lazy import of PyAutoGUIExecutor."""
        # This should trigger the __getattr__ function and import PyAutoGUIExecutor
        from hud.tools.executors import PyAutoGUIExecutor

        # Verify it's imported correctly
        assert PyAutoGUIExecutor.__name__ == "PyAutoGUIExecutor"

    def test_lazy_import_xdo_executor(self):
        """Test lazy import of XDOExecutor."""
        # This should trigger the __getattr__ function and import XDOExecutor
        from hud.tools.executors import XDOExecutor

        # Verify it's imported correctly
        assert XDOExecutor.__name__ == "XDOExecutor"

    def test_lazy_import_invalid_attribute(self):
        """Test lazy import with invalid attribute name."""
        import hud.tools.executors as executors_module

        with pytest.raises(AttributeError, match=r"module '.*' has no attribute 'InvalidExecutor'"):
            _ = executors_module.InvalidExecutor
