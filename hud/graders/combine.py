"""``combine`` — the native subscore combiner."""

from __future__ import annotations

import asyncio
import warnings
from collections import Counter
from collections.abc import Awaitable
from typing import TYPE_CHECKING

# The module, not the name: hud.types imports hud.graders, so Grade is read at call time.
import hud.types

from .base import SubScore

if TYPE_CHECKING:
    from hud.types import Grade


async def combine(*items: SubScore | Awaitable[SubScore]) -> Grade:
    """Combine subscores into a ``Grade``, awaiting any grader coroutines among them concurrently.

    Positive weights are normalized to sum to ``1.0``; negative weights are penalties.
    Duplicate names get numeric suffixes.
    """
    for item in items:
        if not isinstance(item, SubScore | Awaitable):
            raise TypeError(f"Expected SubScore or Awaitable[SubScore], got {type(item).__name__}")
    # gather awaits a coroutine passed more than once a single time and reuses its result.
    subscores = await asyncio.gather(
        *(_ready(item) if isinstance(item, SubScore) else item for item in items)
    )
    if not subscores:
        raise ValueError("subscores must not be empty")

    positive_weight_sum = sum(item.weight for item in subscores if item.weight > 0)
    if positive_weight_sum <= 0:
        raise ValueError("subscores must include at least one positive weight")
    # Normalizing silently would hide a likely authoring mistake (0.5/0.3/0.3 meant
    # as 0.5/0.3/0.2), so warn when the declared weights don't already sum to 1.
    if abs(positive_weight_sum - 1.0) > 0.01:
        warnings.warn(
            f"grader weights sum to {positive_weight_sum:.4f}, not 1.0; "
            f"normalizing, but the effective weights differ from what you set. "
            f"Make the positive weights sum to 1.0 to silence this.",
            stacklevel=2,
        )

    normalized = [
        item.model_copy(
            update={
                "name": unique_name,
                "weight": item.weight / positive_weight_sum if item.weight > 0 else item.weight,
            }
        )
        for item, unique_name in zip(
            subscores, _unique_names([s.name for s in subscores]), strict=True
        )
    ]
    return hud.types.Grade(
        reward=float(sum(item.value * item.weight for item in normalized)),
        done=True,
        subscores=normalized,
    )


async def _ready(subscore: SubScore) -> SubScore:
    return subscore


def _unique_names(names: list[str]) -> list[str]:
    """``names`` with every repeated name suffixed ``-1``, ``-2``, ..., skipping names in use."""
    counts = Counter(names)
    taken = set(names)
    last_suffix: dict[str, int] = {}
    unique: list[str] = []
    for name in names:
        if counts[name] == 1:
            unique.append(name)
            continue
        suffix = last_suffix.get(name, 0) + 1
        while f"{name}-{suffix}" in taken:
            suffix += 1
        last_suffix[name] = suffix
        taken.add(f"{name}-{suffix}")
        unique.append(f"{name}-{suffix}")
    return unique
