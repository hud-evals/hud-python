"""Concurrent grade resolution with a shared cleanup boundary."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Awaitable

    from .results import SubScore


async def resolve_grades(*items: Awaitable[SubScore]) -> list[SubScore]:
    """Resolve grades in order, cancelling and draining siblings on failure."""
    pending = [asyncio.ensure_future(item) for item in items]
    try:
        return list(await asyncio.gather(*pending))
    except BaseException:
        for future in pending:
            future.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        raise
