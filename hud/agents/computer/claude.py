"""``ClaudeComputer``: Claude's computer-use actions, performed on a ``Screen``."""

from __future__ import annotations

import asyncio
import logging
import re
from typing import TYPE_CHECKING, Annotated, Any, Literal, TypeAlias, get_args

from pydantic import BaseModel, BeforeValidator, Field, TypeAdapter, ValidationError

from hud.agents.computer.screen import Point, keysym
from hud.types import TextContent, ToolResult

if TYPE_CHECKING:
    from hud.agents.computer.screen import Button, Screen

logger = logging.getLogger(__name__)


def _chords(text: str) -> list[list[str]]:
    """``text`` in xdotool key syntax: space-separated presses, ``+`` binding a chord."""
    presses = re.sub(r"\s*\+\s*", "+", text).split()
    return [
        chord for press in presses if (chord := [keysym(key) for key in press.split("+") if key])
    ]


Chords = Annotated[list[list[str]], BeforeValidator(_chords)]


class _HoldsKeys(BaseModel):
    held: Chords = Field(default_factory=list, validation_alias="text")

    @property
    def held_keys(self) -> list[str]:
        return [key for chord in self.held for key in chord]


class Screenshot(BaseModel):
    action: Literal["screenshot"]


class Zoom(BaseModel):
    action: Literal["zoom"]
    region: tuple[int, int, int, int]


class CursorPosition(BaseModel):
    action: Literal["cursor_position"]


class Click(_HoldsKeys):
    action: Literal[
        "left_click", "click", "right_click", "middle_click", "double_click", "triple_click"
    ]
    coordinate: Point | None = None


class Move(BaseModel):
    action: Literal["mouse_move", "move"]
    coordinate: Point


class MouseButton(BaseModel):
    action: Literal["left_mouse_down", "left_mouse_up"]


class TypeText(BaseModel):
    action: Literal["type"]
    text: str = Field(min_length=1)


class Key(BaseModel):
    action: Literal["key"]
    chords: Chords = Field(validation_alias="text", min_length=1)
    repeat: int = Field(default=1, ge=1, le=100)


class HoldKey(BaseModel):
    action: Literal["hold_key"]
    chords: Chords = Field(validation_alias="text", min_length=1)
    duration: float = Field(default=0, ge=0)


class Scroll(_HoldsKeys):
    action: Literal["scroll"]
    coordinate: Point | None = None
    scroll_direction: Literal["up", "down", "left", "right"]
    scroll_amount: int = Field(ge=0)


class Drag(_HoldsKeys):
    action: Literal["left_click_drag", "drag"]
    start_coordinate: Point
    coordinate: Point


class Wait(BaseModel):
    action: Literal["wait"]
    duration: float = Field(default=0, ge=0)


ComputerCall: TypeAlias = (
    Screenshot
    | Zoom
    | CursorPosition
    | Click
    | Move
    | MouseButton
    | TypeText
    | Key
    | HoldKey
    | Scroll
    | Drag
    | Wait
)

_CALL: TypeAdapter[ComputerCall] = TypeAdapter(
    Annotated[ComputerCall, Field(discriminator="action")]
)

ACTIONS: tuple[str, ...] = tuple(
    action
    for call in get_args(ComputerCall)
    for action in get_args(call.model_fields["action"].annotation)
)

_CLICKS: dict[str, tuple[Button, int]] = {
    "left_click": ("left", 1),
    "click": ("left", 1),
    "right_click": ("right", 1),
    "middle_click": ("middle", 1),
    "double_click": ("left", 2),
    "triple_click": ("left", 3),
}

_SCROLLS: dict[str, tuple[int, int]] = {
    "down": (0, 1),
    "up": (0, -1),
    "right": (1, 0),
    "left": (-1, 0),
}


class ClaudeComputer:
    """Claude's computer actions (``left_click``, ``key``, ``zoom``, ...) on ``screen``."""

    def __init__(self, screen: Screen) -> None:
        self.screen = screen

    async def run(self, arguments: dict[str, Any]) -> ToolResult:
        """Perform one call; an invalid or failed call is an error result, not a raise."""
        try:
            return await self._perform(_CALL.validate_python(arguments))
        except ValidationError as exc:
            message = "invalid computer call: " + "; ".join(
                f"{'.'.join(map(str, error['loc'])) or 'call'}: {error['msg']}"
                for error in exc.errors()
            )
        except Exception as exc:
            logger.exception("computer action %r failed", arguments.get("action"))
            message = f"computer action {arguments.get('action')!r} failed: {exc}"
        return ToolResult(content=[TextContent(type="text", text=message)], isError=True)

    async def _perform(self, call: ComputerCall) -> ToolResult:
        screen = self.screen
        match call:
            case Screenshot():
                return await screen.screenshot()
            case Zoom(region=region):
                return await screen.screenshot(region=region)
            case CursorPosition():
                x, y = screen.client.cursor
                return ToolResult(content=[TextContent(type="text", text=f"({x}, {y})")])
            case Click(action=action, coordinate=coordinate):
                button, count = _CLICKS[action]
                await screen.click(
                    coordinate, button=button, count=count, interval=0.1, hold_keys=call.held_keys
                )
            case Move(coordinate=(x, y)):
                await screen.move(x, y)
            case MouseButton(action="left_mouse_down"):
                await screen.mouse_down("left")
            case MouseButton():
                await screen.mouse_up("left")
            case TypeText(text=text):
                await screen.type_text(text)
            case Key(chords=chords, repeat=repeat):
                for _ in range(repeat):
                    for chord in chords:
                        await screen.press_keys(chord)
            case HoldKey(chords=chords, duration=duration):
                await screen.hold_key(chords[0][0], duration=duration)
            case Scroll(coordinate=coordinate, scroll_direction=direction, scroll_amount=amount):
                step_x, step_y = _SCROLLS[direction]
                await screen.scroll(
                    coordinate,
                    scroll_x=step_x * amount,
                    scroll_y=step_y * amount,
                    hold_keys=call.held_keys,
                )
            case Drag(start_coordinate=start, coordinate=end):
                await screen.drag([start, end], hold_keys=call.held_keys)
            case Wait(duration=duration):
                await asyncio.sleep(duration)
        return await screen.screenshot()
