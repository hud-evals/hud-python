"""``combine`` — the native subscore combiner."""

from __future__ import annotations

import asyncio
from collections import Counter
from collections.abc import Awaitable

from .base import Grade, SubScore


async def combine(*items: SubScore | Awaitable[SubScore]) -> Grade:
    """Combine subscores into a ``Grade``, awaiting any grader coroutines among them concurrently.

    Positive weights must sum to ``1.0`` (within 0.01) and are normalized to it exactly;
    negative weights are penalties. Duplicate names get numeric suffixes.
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
    if abs(positive_weight_sum - 1.0) > 0.01:
        raise ValueError(
            f"positive subscore weights must sum to 1.0, got {positive_weight_sum:.4f}"
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
    return Grade(
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
