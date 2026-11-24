"""``combine`` — the native subscore combiner."""

from __future__ import annotations

import asyncio
from collections import Counter
from collections.abc import Awaitable, Callable

from hud.utils.deprecation import warn_deprecated

from .base import Grade, SubScore


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
        warn_deprecated(
            f"`combine` with positive weights summing to {positive_weight_sum:.4f} (normalized "
            "here, so the effective weights differ from the ones set)",
            "make the positive weights sum to 1.0",
            change="will raise",
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


def combine_any(weight: float, subscores: list[SubScore], *, name: str = "any") -> SubScore:
    """A subscore that passes if any input passes (max), with the inputs as children."""
    warn_deprecated(
        "`combine_any`",
        "build the subscore directly: `SubScore(name=..., value=max(s.value for s in "
        "subscores), weight=..., children=subscores)`",
    )
    return _boolean_subscore(name, weight, subscores, max)


def combine_all(weight: float, subscores: list[SubScore], *, name: str = "all") -> SubScore:
    """A subscore that passes only if every input passes (min), with the inputs as children."""
    warn_deprecated(
        "`combine_all`",
        "build the subscore directly: `SubScore(name=..., value=min(s.value for s in "
        "subscores), weight=..., children=subscores)`",
    )
    return _boolean_subscore(name, weight, subscores, min)


def _boolean_subscore(
    name: str, weight: float, subscores: list[SubScore], pick: Callable[[list[float]], float]
) -> SubScore:
    if not subscores:
        raise ValueError("subscores must not be empty")
    children = [
        subscore.model_copy(update={"name": unique})
        for subscore, unique in zip(
            subscores, _unique_names([s.name for s in subscores]), strict=True
        )
    ]
    return SubScore(
        name=name, value=pick([s.value for s in subscores]), weight=weight, children=children
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
