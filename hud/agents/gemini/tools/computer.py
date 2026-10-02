"""Gemini Computer Use tool — backed by RFBClient."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Self

from google.genai import types as genai_types

from hud.agents.tools import RFBTool
from hud.agents.tools.base import tool_err
from hud.agents.tools.rfb import keysym
from hud.agents.types import GeminiConfig

from .base import GeminiToolSpec

if TYPE_CHECKING:
    from collections.abc import Sequence

    from hud.agents.tools.base import AgentToolSpec
    from hud.agents.types import ToolAgentConfig
    from hud.capabilities import RFBClient
    from hud.capabilities.rfb import ScreenshotEncoding
    from hud.types import ToolResult

logger = logging.getLogger(__name__)

GEMINI_DRAG_INSET = 25
#: Gemini places every coordinate on a 1000x1000 grid regardless of the screen size.
GEMINI_COORDINATE_SCALE = 1000

#: The desktop actions Gemini 3 models call; the browser and mobile ones need a page or app.
PREDEFINED_COMPUTER_USE_FUNCTIONS = (
    "click",
    "double_click",
    "triple_click",
    "middle_click",
    "right_click",
    "mouse_down",
    "mouse_up",
    "move",
    "type",
    "drag_and_drop",
    "scroll",
    "wait",
    "press_key",
    "key_down",
    "key_up",
    "hotkey",
    "take_screenshot",
)

_MAC_COMMAND = frozenset({"meta", "cmd", "command"})

GEMINI_COMPUTER_SPEC = GeminiToolSpec(
    api_type="computer_use",
    api_name="gemini_computer",
)


class GeminiComputerTool(RFBTool):
    """Translate Gemini's desktop computer actions into RFBTool primitives.

    The RFB capability is a bare display, so the tool declares the desktop
    environment: no page to navigate.
    """

    name = "computer_use"

    def __init__(
        self,
        *,
        spec: AgentToolSpec,
        client: RFBClient,
        screenshot_encoding: ScreenshotEncoding,
        excluded_predefined_functions: Sequence[str] = (),
    ) -> None:
        super().__init__(spec=spec, client=client, screenshot_encoding=screenshot_encoding)
        self.excluded_predefined_functions = list(excluded_predefined_functions)
        #: The calls this tool answers. An excluded action is left to an environment
        #: function of the same name.
        self.actions = frozenset(PREDEFINED_COMPUTER_USE_FUNCTIONS) - frozenset(
            excluded_predefined_functions
        )

    @classmethod
    def from_config(
        cls, *, spec: AgentToolSpec, client: RFBClient, config: ToolAgentConfig
    ) -> Self:
        assert isinstance(config, GeminiConfig)
        return cls(
            spec=spec,
            client=client,
            screenshot_encoding=config.screenshot_encoding,
            excluded_predefined_functions=config.excluded_predefined_functions,
        )

    @classmethod
    def default_spec(cls, model: str) -> GeminiToolSpec:
        del model
        return GEMINI_COMPUTER_SPEC

    def to_params(self) -> genai_types.Tool:
        return genai_types.Tool(
            computer_use=genai_types.ComputerUse(
                environment=genai_types.Environment.ENVIRONMENT_DESKTOP,
                excluded_predefined_functions=self.excluded_predefined_functions,
            ),
        )

    async def execute(self, arguments: dict[str, Any]) -> ToolResult:
        action = arguments.get("action")
        if not isinstance(action, str):
            return tool_err("action is required")
        try:
            return await self._dispatch(action, arguments)
        except Exception as exc:
            logger.exception("GeminiComputerTool action %s failed", action)
            return tool_err(f"computer action {action!r} failed: {exc}")

    async def _dispatch(self, action: str, args: dict[str, Any]) -> ToolResult:
        match action:
            case "take_screenshot":
                pass
            case "click":
                await self.click(*self._point(args))
            case "double_click":
                await self.click(*self._point(args), count=2, interval_ms=100)
            case "triple_click":
                await self.click(*self._point(args), count=3, interval_ms=100)
            case "middle_click":
                await self.click(*self._point(args), button="middle")
            case "right_click":
                await self.click(*self._point(args), button="right")
            case "move":
                await self.move(*self._point(args))
            case "mouse_down":
                await self.move(*self._point(args))
                await self.mouse_down()
            case "mouse_up":
                await self.move(*self._point(args))
                await self.mouse_up()
            case "type":
                if "x" in args and "y" in args:
                    await self.click(*self._point(args))
                await self.type_text(str(args["text"]))
                if args.get("press_enter"):
                    await self.press_keys(["Return"])
            case "drag_and_drop":
                width, height = self.display_width, self.display_height
                path = [
                    (_inset(args["start_x"], width), _inset(args["start_y"], height)),
                    (_inset(args["end_x"], width), _inset(args["end_y"], height)),
                ]
                await self.drag(path)
            case "scroll":
                # One wheel click per 100 of magnitude_in_pixels (0-999), at least one.
                clicks = max(1, int(args.get("magnitude_in_pixels", 300)) // 100)
                sx, sy = {
                    "down": (0, clicks),
                    "up": (0, -clicks),
                    "right": (clicks, 0),
                    "left": (-clicks, 0),
                }[args["direction"]]
                await self.scroll(*self._point(args), scroll_x=sx, scroll_y=sy)
            case "wait":
                await self.wait(int(float(args.get("seconds", 1)) * 1000))
            case "press_key":
                await self.press_keys(_keys(str(args["key"]).split("+")))
            case "key_down":
                await self.key_down(*_keys([str(args["key"])]))
            case "key_up":
                await self.key_up(*_keys([str(args["key"])]))
            case "hotkey":
                await self.press_keys(_keys([str(key) for key in args["keys"]]))
            case _:
                return tool_err(f"Unknown Gemini computer action: {action}")
        return await self.screenshot()

    def _point(self, args: dict[str, Any]) -> tuple[int, int]:
        return _pixel(args["x"], self.display_width), _pixel(args["y"], self.display_height)


def _pixel(value: float, size: int) -> int:
    """Map a coordinate on Gemini's normalized 0-999 grid onto a ``size``-pixel axis."""
    return int(value / GEMINI_COORDINATE_SCALE * size)


def _inset(value: float, size: int) -> int:
    """A drag point kept off the screen's edge, where a press can miss the window."""
    return min(max(_pixel(value, size), GEMINI_DRAG_INSET), size - GEMINI_DRAG_INSET)


def _keys(names: list[str]) -> list[str]:
    keys = [name.strip() for name in names if name.strip()]
    # Gemini names keys as a browser on macOS does, where Meta (Command) chords are
    # Control chords on Linux.
    return [keysym("ctrl" if key.lower() in _MAC_COMMAND else key) for key in keys]


__all__ = ["GEMINI_COMPUTER_SPEC", "PREDEFINED_COMPUTER_USE_FUNCTIONS", "GeminiComputerTool"]
