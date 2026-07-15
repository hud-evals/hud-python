"""``Grader`` — the async base class for reusable graders."""

from __future__ import annotations

from typing import Any

from hud.utils.serialization import json_safe_dict

from .results import SubScore


class Grader:
    """Async base class for reusable graders.

    Subclasses implement ``compute_score`` (async). The ``grade`` classmethod
    calls it, stamps the caller's ``name`` and ``weight`` onto the resulting
    ``SubScore``, and records parameters in metadata for reproducibility.
    """

    name: str = "BaseGrader"

    @classmethod
    async def grade(cls, weight: float, name: str | None = None, **kwargs: Any) -> SubScore:
        """Run the grader and package the result as a ``SubScore``."""
        result = await cls.compute_score(**kwargs)
        if not isinstance(result, SubScore):
            result = SubScore(name=cls.name, value=float(result))

        return result.model_copy(
            update={
                "name": name or cls.name,
                "weight": weight,
                "metadata": {**(result.metadata or {}), "_parameters": json_safe_dict(kwargs)},
            }
        )

    @classmethod
    async def compute_score(cls, **kwargs: Any) -> float | SubScore:
        """Compute a score between ``0.0`` and ``1.0``.

        Return a float, or a ``SubScore`` to attach children, a reason, or
        metadata (its ``name`` and ``weight`` are overwritten by ``grade``).
        """
        raise NotImplementedError("Subclasses must implement compute_score")


__all__ = ["Grader"]
