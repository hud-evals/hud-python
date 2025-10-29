from __future__ import annotations

import platform
from typing import Any

import pytest

from hud.tools.computer.gemini import GeminiComputerTool
from hud.tools.types import ContentResult


class DummyExecutor:
    """Minimal async executor used for tooling tests."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def click(self, *, x: int | None, y: int | None, **kwargs: Any) -> ContentResult:
        self.calls.append(("click", {"x": x, "y": y, **kwargs}))
        return ContentResult(url="about:blank")

    async def move(self, *, x: int | None, y: int | None, **kwargs: Any) -> ContentResult:
        self.calls.append(("move", {"x": x, "y": y, **kwargs}))
        return ContentResult(url="about:blank")

    async def press(self, *, keys: list[str], **kwargs: Any) -> ContentResult:
        self.calls.append(("press", {"keys": keys, **kwargs}))
        return ContentResult(url="about:blank")

    async def write(self, *, text: str, enter_after: bool, **kwargs: Any) -> ContentResult:
        self.calls.append(("write", {"text": text, "enter_after": enter_after, **kwargs}))
        return ContentResult(url="about:blank")

    async def scroll(
        self,
        *,
        x: int | None = None,
        y: int | None = None,
        scroll_x: int | None = None,
        scroll_y: int | None = None,
        **kwargs: Any,
    ) -> ContentResult:
        self.calls.append(
            (
                "scroll",
                {"x": x, "y": y, "scroll_x": scroll_x, "scroll_y": scroll_y, **kwargs},
            )
        )
        return ContentResult(url="about:blank")

    async def drag(self, *, path: list[tuple[int, int]], **kwargs: Any) -> ContentResult:
        self.calls.append(("drag", {"path": path, **kwargs}))
        return ContentResult(url="about:blank")

    async def screenshot(self) -> str | None:  # pragma: no cover - not used in tests
        return None

    async def wait(self, **kwargs: Any) -> ContentResult:  # pragma: no cover - unused
        return ContentResult(url="about:blank")


@pytest.mark.asyncio
async def test_coordinates_denormalize_to_screen_space():
    executor = DummyExecutor()
    tool = GeminiComputerTool(
        executor=executor,
        width=1920,
        height=1080,
        rescale_images=False,
    )

    await tool(action="click_at", x=500, y=250)

    assert executor.calls[0][0] == "click"
    # 500/1000 * 1920 = 960, 250/1000 * 1080 = 270
    assert executor.calls[0][1]["x"] == 960
    assert executor.calls[0][1]["y"] == 270


@pytest.mark.asyncio
async def test_type_text_clears_before_typing(monkeypatch: pytest.MonkeyPatch):
    executor = DummyExecutor()
    tool = GeminiComputerTool(
        executor=executor,
        width=1920,
        height=1080,
        rescale_images=False,
    )

    monkeypatch.setattr(platform, "system", lambda: "Linux")

    await tool(action="type_text_at", x=100, y=100, text="hello", press_enter=True)

    # Expect move + click for focus
    assert executor.calls[0][0] == "move"
    assert executor.calls[1][0] == "click"

    # Clearing sequence should press ctrl+a then delete
    press_calls = [call for call in executor.calls if call[0] == "press"]
    assert press_calls[0][1]["keys"] == ["ctrl", "a"]
    assert press_calls[1][1]["keys"] == ["delete"]

    # Final write call should include the text and Enter flag
    write_call = next(call for call in executor.calls if call[0] == "write")
    assert write_call[1]["text"] == "hello"
    assert write_call[1]["enter_after"] is True


@pytest.mark.asyncio
async def test_drag_and_drop_denormalizes_path():
    executor = DummyExecutor()
    tool = GeminiComputerTool(
        executor=executor,
        width=1920,
        height=1080,
        rescale_images=False,
    )

    await tool(action="drag_and_drop", x=100, y=200, destination_x=900, destination_y=800)

    drag_call = next(call for call in executor.calls if call[0] == "drag")
    path = drag_call[1]["path"]
    # Ensure both start and end points are denormalized from 0-1000 range
    assert path[0] == (192, 216)  # 100/1000 of width/height
    assert path[1] == (1728, 864)  # 900/1000 of width/height
