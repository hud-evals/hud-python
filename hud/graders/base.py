"""``Grader`` — the async base class for reusable graders."""

from __future__ import annotations

from typing import Any

from hud.utils.serialization import json_safe_dict

from .results import SubScore


class Grader:
    """Async base class for reusable graders.

    Subclasses implement ``compute_score`` (async). The ``grade`` classmethod
    calls it, wraps the result as a ``SubScore``, and records parameters
    in metadata for reproducibility.
    """

    name: str = "BaseGrader"

    @classmethod
    async def grade(cls, weight: float, name: str | None = None, **kwargs: Any) -> SubScore:
        """Run the grader and package the result as a ``SubScore``."""
        result = await cls.compute_score(**kwargs)

        if isinstance(result, tuple):
            score, metadata = result
        else:
            score = result
            metadata = {}

        return SubScore(
            name=name or cls.name,
            weight=weight,
            value=float(score),
            metadata={**metadata, "_parameters": json_safe_dict(kwargs)},
        )

    @classmethod
    async def compute_score(cls, **kwargs: Any) -> float | tuple[float, dict[str, Any]]:
        """Compute a score between ``0.0`` and ``1.0``.

        Return a float, or ``(float, metadata_dict)`` to attach extra info.
        """
        raise NotImplementedError("Subclasses must implement compute_score")


__all__ = ["Grader"]
