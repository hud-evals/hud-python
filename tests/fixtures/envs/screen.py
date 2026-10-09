"""A screen with a target: a reward of 1 proves the agent's click landed on it."""

from __future__ import annotations

from contextlib import AsyncExitStack

from hud import Environment
from hud.capabilities import Capability
from tests.harness import fake_screen

WIDTH, HEIGHT = 200, 100
TARGET = (100, 50)

env = Environment("screen")
stack = AsyncExitStack()
screens = []


@env.initialize
async def show_screen() -> None:
    screen = await stack.enter_async_context(fake_screen(width=WIDTH, height=HEIGHT))
    screens.append(screen)
    env.add_capability(Capability.rfb(name="screen", url=screen.url))


@env.shutdown
async def hide_screen() -> None:
    await stack.aclose()
    screens.clear()


@env.template()
async def click_target():
    (screen,) = screens
    already = len(screen.pointer_events())
    yield f"Left-click the pixel at {TARGET[0]},{TARGET[1]} on the {WIDTH}x{HEIGHT} screen."
    pressed = [
        (event.x, event.y) for event in screen.pointer_events()[already:] if event.buttons & 1
    ]
    yield 1.0 if pressed and set(pressed) == {TARGET} else 0.0
