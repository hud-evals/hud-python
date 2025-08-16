"""Tests for PyAutoGUI executor."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from hud.tools.executors.pyautogui import PyAutoGUIExecutor
from hud.tools.types import ContentResult

# Check if pyautogui is available for test skipping
PYAUTOGUI_AVAILABLE = PyAutoGUIExecutor.is_available()


class TestPyAutoGUIExecutor:
    """Tests for PyAutoGUIExecutor."""

    def test_is_available(self):
        """Test is_available method."""
        # The availability is determined by the module-level PYAUTOGUI_AVAILABLE
        assert PyAutoGUIExecutor.is_available() == PYAUTOGUI_AVAILABLE

    @pytest.mark.skipif(not PYAUTOGUI_AVAILABLE, reason="pyautogui not available")
    @pytest.mark.asyncio
    async def test_screenshot_with_pyautogui(self):
        """Test screenshot when pyautogui is available."""
        executor = PyAutoGUIExecutor()

        # Mock pyautogui screenshot
        with patch("pyautogui.screenshot") as mock_screenshot:
            mock_img = MagicMock()
            mock_img.save = MagicMock()
            mock_screenshot.return_value = mock_img

            result = await executor.screenshot()

            # screenshot() returns a base64 string, not a ContentResult
            assert isinstance(result, str)
            mock_screenshot.assert_called_once()

    @pytest.mark.skipif(not PYAUTOGUI_AVAILABLE, reason="pyautogui not available")
    @pytest.mark.asyncio
    async def test_click_with_pyautogui(self):
        """Test click when pyautogui is available."""
        executor = PyAutoGUIExecutor()

        with patch("pyautogui.click") as mock_click:
            result = await executor.click(100, 200, "left")

            assert isinstance(result, ContentResult)
            assert result.output and "Clicked" in result.output
            mock_click.assert_called_once_with(x=100, y=200, button="left")

    @pytest.mark.skipif(not PYAUTOGUI_AVAILABLE, reason="pyautogui not available")
    @pytest.mark.asyncio
    async def test_type_text_with_pyautogui(self):
        """Test type when pyautogui is available."""
        executor = PyAutoGUIExecutor()

        with patch("pyautogui.typewrite") as mock_type:
            result = await executor.write("Hello world")

            assert isinstance(result, ContentResult)
            assert result.output and "Typed" in result.output
            # The implementation adds interval=0.012 (12ms converted to seconds)
            mock_type.assert_called_once_with("Hello world", interval=0.012)

    @pytest.mark.skipif(not PYAUTOGUI_AVAILABLE, reason="pyautogui not available")
    @pytest.mark.asyncio
    async def test_press_keys_with_pyautogui(self):
        """Test press when pyautogui is available."""
        executor = PyAutoGUIExecutor()

        # For key combinations, the implementation uses hotkey
        with patch("pyautogui.hotkey") as mock_hotkey:
            result = await executor.press(["ctrl", "a"])

            assert isinstance(result, ContentResult)
            assert result.output and "Pressed" in result.output
            mock_hotkey.assert_called_once_with("ctrl", "a")

    @pytest.mark.skipif(not PYAUTOGUI_AVAILABLE, reason="pyautogui not available")
    @pytest.mark.asyncio
    async def test_scroll_with_pyautogui(self):
        """Test scroll when pyautogui is available."""
        executor = PyAutoGUIExecutor()

        with patch("pyautogui.moveTo") as mock_move, patch("pyautogui.scroll") as mock_scroll:
            result = await executor.scroll(100, 200, scroll_y=5)

            assert isinstance(result, ContentResult)
            assert result.output and "Scrolled" in result.output
            # First moves to position
            mock_move.assert_called_once_with(100, 200)
            # Then scrolls (note: implementation negates scroll_y)
            mock_scroll.assert_called_once_with(-5)

    @pytest.mark.skipif(not PYAUTOGUI_AVAILABLE, reason="pyautogui not available")
    @pytest.mark.asyncio
    async def test_move_with_pyautogui(self):
        """Test move when pyautogui is available."""
        executor = PyAutoGUIExecutor()

        with patch("pyautogui.moveTo") as mock_move:
            result = await executor.move(300, 400)

            assert isinstance(result, ContentResult)
            assert result.output and "Moved" in result.output
            # The implementation adds duration=0.1
            mock_move.assert_called_once_with(300, 400, duration=0.1)

    @pytest.mark.skipif(not PYAUTOGUI_AVAILABLE, reason="pyautogui not available")
    @pytest.mark.asyncio
    async def test_drag_with_pyautogui(self):
        """Test drag when pyautogui is available."""
        executor = PyAutoGUIExecutor()

        with patch("pyautogui.dragTo") as mock_drag:
            # drag expects a path (list of coordinate tuples)
            path = [(100, 100), (300, 400)]
            result = await executor.drag(path)

            assert isinstance(result, ContentResult)
            assert result.output and "Dragged" in result.output
            # Implementation uses dragTo to move to each point
            mock_drag.assert_called()

    @pytest.mark.asyncio
    async def test_wait(self):
        """Test wait method."""
        executor = PyAutoGUIExecutor()

        # Mock asyncio.sleep
        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            # wait expects time in milliseconds
            result = await executor.wait(2500)  # 2500ms = 2.5s

            assert isinstance(result, ContentResult)
            assert result.output and "Waited" in result.output
            # Implementation converts to seconds
            mock_sleep.assert_called_once_with(2.5)

    @pytest.mark.skipif(not PYAUTOGUI_AVAILABLE, reason="pyautogui not available")
    @pytest.mark.asyncio
    async def test_position_with_pyautogui(self):
        """Test position when pyautogui is available."""
        executor = PyAutoGUIExecutor()

        with patch("pyautogui.position") as mock_position:
            mock_position.return_value = (123, 456)
            result = await executor.position()

            assert isinstance(result, ContentResult)
            assert result.output is not None
            assert "Mouse position" in result.output
            assert "123" in result.output
            assert "456" in result.output
            mock_position.assert_called_once()

    def test_init_with_display_num(self):
        """Test initialization with display number."""
        # Should not raise
        executor = PyAutoGUIExecutor(display_num=0)
        assert executor.display_num == 0
