"""A screen with a target: a reward of 1 proves the agent's click landed on it.

A click within ``SLACK`` pixels of the target counts, so a real model that rounds
its coordinates still lands.
"""

from __future__ import annotations

from contextlib import AsyncExitStack

from hud import Environment
from hud.capabilities import Capability
from tests.support import fake_screen

WIDTH, HEIGHT = 200, 100
TARGET = (100, 50)
SLACK = 2

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
    yield (
        f"Use the computer to left-click the pixel at {TARGET[0]},{TARGET[1]} on the "
        f"{WIDTH}x{HEIGHT} screen. Take the action; do not describe it."
    )
    pressed = [
        (event.x, event.y) for event in screen.pointer_events()[already:] if event.buttons & 1
    ]
    landed = all(abs(x - TARGET[0]) <= SLACK and abs(y - TARGET[1]) <= SLACK for x, y in pressed)
    yield 1.0 if pressed and landed else 0.0
