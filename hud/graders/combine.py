"""``combine`` — the native subscore combiner, plus boolean combiners."""

from __future__ import annotations

import asyncio
import warnings
from typing import TYPE_CHECKING

from .results import EvaluationResult, SubScore

if TYPE_CHECKING:
    from collections.abc import Awaitable


def _dedupe_subscore_names(subscores: list[SubScore]) -> list[str]:
    """Return stable, unique names for a sequence of subscores."""
    name_counts: dict[str, int] = {}
    for item in subscores:
        name_counts[item.name] = name_counts.get(item.name, 0) + 1

    reserved_names = {item.name for item in subscores}
    name_usage: dict[str, int] = {}
    used_names: set[str] = set()
    final_names: list[str] = []

    for item in subscores:
        if name_counts[item.name] == 1 and item.name not in used_names:
            final_name = item.name
        else:
            suffix = name_usage.get(item.name, 0)
            while True:
                suffix += 1
                candidate = f"{item.name}-{suffix}"
                if candidate in used_names:
                    continue
                if candidate in reserved_names:
                    continue
                name_usage[item.name] = suffix
                final_name = candidate
                break
        used_names.add(final_name)
        final_names.append(final_name)

    return final_names


def _combine_subscores(subscores: list[SubScore]) -> EvaluationResult:
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
            stacklevel=3,
        )

    normalized_subscores: list[SubScore] = []

    for item, final_name in zip(subscores, _dedupe_subscore_names(subscores), strict=True):
        normalized_weight = item.weight / positive_weight_sum if item.weight > 0 else item.weight
        normalized_subscores.append(
            item.model_copy(update={"name": final_name, "weight": normalized_weight})
        )

    reward = float(sum(item.value * item.weight for item in normalized_subscores))

    return EvaluationResult(
        reward=reward,
        done=True,
        subscores=normalized_subscores,
    )


async def combine(*items: SubScore | Awaitable[SubScore]) -> EvaluationResult:
    """Combine subscores into a result, awaiting any grader coroutines among them concurrently.

    Positive weights are normalized to sum to ``1.0``; negative weights are penalties.
    Duplicate names get numeric suffixes.
    """
    from collections.abc import Awaitable as _Awaitable

    resolved: list[SubScore] = []
    pending: list[tuple[int, _Awaitable[SubScore]]] = []

    for item in items:
        if isinstance(item, SubScore):
            resolved.append(item)
        elif isinstance(item, _Awaitable):
            pending.append((len(resolved), item))
            resolved.append(SubScore(name="__placeholder__", value=0.0, weight=0.0))
        else:
            raise TypeError(f"Expected SubScore or Awaitable[SubScore], got {type(item).__name__}")

    if pending:
        results = await asyncio.gather(*(aw for _, aw in pending))
        for (slot, _), result in zip(pending, results, strict=True):
            resolved[slot] = result

    return _combine_subscores(resolved)


def _boolean_subscore(
    name: str,
    weight: float,
    subscores: list[SubScore],
    value: float,
) -> SubScore:
    children = [
        subscore.model_copy(update={"name": unique_name})
        for unique_name, subscore in zip(_dedupe_subscore_names(subscores), subscores, strict=True)
    ]
    return SubScore(
        name=name,
        value=value,
        weight=weight,
        children=children,
    )


def combine_any(weight: float, subscores: list[SubScore], *, name: str = "any") -> SubScore:
    """Subscore that passes if any input passes (max); inputs kept as children."""
    if not subscores:
        raise ValueError("subscores must not be empty")
    return _boolean_subscore(name, weight, subscores, max(s.value for s in subscores))


def combine_all(weight: float, subscores: list[SubScore], *, name: str = "all") -> SubScore:
    """Subscore that passes only if all inputs pass (min); inputs kept as children."""
    if not subscores:
        raise ValueError("subscores must not be empty")
    return _boolean_subscore(name, weight, subscores, min(s.value for s in subscores))
