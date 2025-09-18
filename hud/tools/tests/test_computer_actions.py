from __future__ import annotations

import pytest
from mcp.types import ImageContent, TextContent

from hud.tools.computer.hud import HudComputerTool

# (action, kwargs)
CASES = [
    ("screenshot", {}),
    ("click", {"x": 1, "y": 1, "button": None, "pattern": None, "hold_keys": None}),
    ("press", {"keys": ["ctrl", "c"]}),
    ("keydown", {"keys": ["shift"]}),
    ("keyup", {"keys": ["shift"]}),
    ("type", {"text": "hello", "enter_after": None}),
    ("scroll", {"x": 10, "y": 10, "scroll_x": None, "scroll_y": 20, "hold_keys": None}),
    ("wait", {"time": 5}),
    ("drag", {"path": [(0, 0), (10, 10)], "pattern": None, "hold_keys": None}),
    ("mouse_down", {"button": None}),
    ("mouse_up", {"button": None}),
    ("hold_key", {"text": "a", "duration": 0.1}),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("action, params", CASES)
async def test_hud_computer_actions(action: str, params: dict):
    comp = HudComputerTool()
    blocks = await comp(action=action, **params)
    # Ensure at least one content block is returned
    assert blocks
    assert all(isinstance(b, ImageContent | TextContent) for b in blocks)
