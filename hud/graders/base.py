"""``Grader`` — the async base class for reusable graders."""

from __future__ import annotations

from typing import Any

from hud.utils.serialization import json_safe_dict

from .results import SubScore


class Grader:
    """Async base class for reusable graders.

    Subclasses implement ``compute_score`` (async) and return a ``SubScore``.
    The ``grade`` classmethod applies the caller's ``name`` override and
    ``weight``, and records parameters in metadata for reproducibility.
    """

    name: str = "BaseGrader"

    @classmethod
    async def grade(cls, weight: float, name: str | None = None, **kwargs: Any) -> SubScore:
        """Run the grader and package the result as a ``SubScore``."""
        result = await cls.compute_score(**kwargs)
        if isinstance(result, tuple):
            score, metadata = result
            result = SubScore(name=cls.name, value=float(score), metadata=metadata)
        elif not isinstance(result, SubScore):
            result = SubScore(name=cls.name, value=float(result))

        return result.model_copy(
            update={
                "name": name or result.name,
                "weight": weight,
                "metadata": {**(result.metadata or {}), "_parameters": json_safe_dict(kwargs)},
            }
        )

    @classmethod
    async def compute_score(cls, **kwargs: Any) -> float | tuple[float, dict[str, Any]] | SubScore:
        """Compute a score between ``0.0`` and ``1.0``.

        Return a ``SubScore``. ``grade`` also coerces legacy float and
        ``(float, metadata)`` results for backwards compatibility.
        """
        raise NotImplementedError("Subclasses must implement compute_score")


__all__ = ["Grader"]
