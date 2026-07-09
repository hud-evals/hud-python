"""``combine`` — the native subscore combiner, plus boolean combiners."""

from __future__ import annotations

import asyncio
import warnings
from typing import TYPE_CHECKING, Any

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
    """Combine already-resolved subscores into a weighted result.

    Positive weights are normalized to sum to ``1.0``.
    Negative weights are preserved as penalties.
    """
    if not subscores:
        raise ValueError("subscores must not be empty")

    positive_weight_sum = sum(item.weight for item in subscores if item.weight > 0)
    if positive_weight_sum <= 0:
        raise ValueError("subscores must include at least one positive weight")

    # Surface a likely authoring mistake instead of silently reweighting: if the
    # declared positive weights don't already sum to ~1.0, the effective weights
    # after normalization differ from what was written (e.g. 0.5/0.3/0.3 was
    # meant to be 0.5/0.3/0.2). We still normalize (the result stays in [0,1]),
    # but the author should see it.
    if abs(positive_weight_sum - 1.0) > 0.01:
        warnings.warn(
            f"grader weights sum to {positive_weight_sum:.4f}, not 1.0; "
            f"normalizing, but the effective weights differ from what you set. "
            f"Make the positive weights sum to 1.0 to silence this.",
            stacklevel=3,
        )

    normalized_subscores: list[SubScore] = []
    metadata: dict[str, Any] = {}

    for item, final_name in zip(subscores, _dedupe_subscore_names(subscores), strict=True):
        normalized_weight = item.weight / positive_weight_sum if item.weight > 0 else item.weight
        normalized_subscores.append(
            SubScore(
                name=final_name,
                weight=normalized_weight,
                value=item.value,
                criteria=item.criteria,
                metadata=item.metadata,
            )
        )
        if item.metadata is not None:
            metadata[final_name] = item.metadata

    reward = float(sum(item.value * item.weight for item in normalized_subscores))

    return EvaluationResult(
        reward=reward,
        done=True,
        subscores=normalized_subscores,
        info=metadata,
    )


async def combine(*items: SubScore | Awaitable[SubScore]) -> EvaluationResult:
    """Resolve subscores and grader coroutines in parallel, then combine.

    Accepts a mix of:
    - ``SubScore`` objects (used immediately)
    - Awaitables returning ``SubScore`` (e.g. ``Grader.grade()``)

    All awaitables run concurrently via ``asyncio.gather``. Positive weights
    are normalized to sum to ``1.0``; negative weights are penalties.

    Example::

        yield await combine(
            BashGrader.grade(weight=0.3, command="pytest -q"),
            LLMJudgeGrader.grade(weight=0.4, answer=answer, criteria=[...]),
            SubScore(name="answer", value=exact_match(answer, "42"), weight=0.3),
        )
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
    name: str, weight: float, subscores: list[SubScore], value: float
) -> SubScore:
    unique_names = _dedupe_subscore_names(subscores)
    criteria = [
        c if c.source is not None else c.model_copy(update={"source": unique_name})
        for unique_name, subscore in zip(unique_names, subscores, strict=True)
        for c in subscore.criteria or []
    ]
    return SubScore(
        name=name,
        value=value,
        weight=weight,
        criteria=criteria or None,
        metadata={
            "subscores": unique_names,
            "subscore_metadata": {
                unique_name: subscore.metadata
                for unique_name, subscore in zip(unique_names, subscores, strict=True)
                if subscore.metadata is not None
            },
        },
    )


def combine_any(weight: float, subscores: list[SubScore], *, name: str = "any") -> SubScore:
    """Subscore that passes if any input passes (max)."""
    if not subscores:
        raise ValueError("subscores must not be empty")
    return _boolean_subscore(name, weight, subscores, max(s.value for s in subscores))


def combine_all(weight: float, subscores: list[SubScore], *, name: str = "all") -> SubScore:
    """Subscore that passes only if all inputs pass (min)."""
    if not subscores:
        raise ValueError("subscores must not be empty")
    return _boolean_subscore(name, weight, subscores, min(s.value for s in subscores))


__all__ = ["_combine_subscores", "combine", "combine_all", "combine_any"]
