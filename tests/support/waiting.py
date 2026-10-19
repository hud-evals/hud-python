"""Waiting for what the SDK does in the background."""

from __future__ import annotations

import asyncio
import inspect
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable


async def eventually(
    condition: Callable[[], bool | Awaitable[bool]], *, within: float = 10.0
) -> None:
    """Poll until ``condition()`` holds; raise ``TimeoutError`` after ``within`` seconds.

    ``condition`` may be a coroutine function, for a check that has to await.
    """
    async with asyncio.timeout(within):
        while True:
            held = condition()
            if await held if inspect.isawaitable(held) else held:
                return
            await asyncio.sleep(0.01)
