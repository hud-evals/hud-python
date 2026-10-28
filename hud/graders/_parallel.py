"""Concurrent grade resolution with a shared cleanup boundary."""

from __future__ import annotations

import asyncio
import contextlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Awaitable

    from .results import SubScore


async def resolve_grades(*items: Awaitable[SubScore]) -> list[SubScore]:
    """Resolve grades in order, cancelling and draining siblings on failure."""
    pending = {item: asyncio.ensure_future(item) for item in dict.fromkeys(items)}
    batch = asyncio.gather(*(pending[item] for item in items))
    try:
        return list(await asyncio.shield(batch))
    except BaseException:
        for future in pending.values():
            future.cancel()
        cleanup = asyncio.gather(*pending.values(), batch, return_exceptions=True)
        # Repeated cancellation must not interrupt a grader's cleanup.
        while not cleanup.done():
            with contextlib.suppress(asyncio.CancelledError):
                await asyncio.shield(cleanup)
        raise
