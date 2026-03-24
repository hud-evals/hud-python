"""Generic graders for native HUD evaluation."""

from __future__ import annotations

import json
import logging
import subprocess
from typing import Any

from hud.tools.types import EvaluationResult, SubScore

logger = logging.getLogger(__name__)

__all__ = ["BashGrader", "Grade", "Grader"]


def _safe_params(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Return a JSON-safe copy of grader parameters for metadata storage."""
    result: dict[str, Any] = {}
    for key, value in kwargs.items():
        try:
            json.dumps(value)
            result[key] = value
        except (TypeError, ValueError):
            result[key] = f"<{type(value).__name__}: not serializable>"
    return result


class Grade:
    """Factory for building ``EvaluationResult`` objects from ``SubScore`` items."""

    @staticmethod
    def from_subscores(subscores: list[SubScore]) -> EvaluationResult:
        """Combine subscores into a clipped reward and ready-to-yield result.

        Positive weights are normalized to sum to ``1.0`` so the returned
        ``EvaluationResult`` lines up with the SDK's subscore semantics.
        Negative weights are preserved as penalties.
        """

        if not subscores:
            raise ValueError("subscores must not be empty")

        positive_weight_sum = sum(item.weight for item in subscores if item.weight > 0)
        if positive_weight_sum <= 0:
            raise ValueError("subscores must include at least one positive weight")

        name_counts: dict[str, int] = {}
        for item in subscores:
            name_counts[item.name] = name_counts.get(item.name, 0) + 1

        name_usage: dict[str, int] = {}
        normalized_subscores: list[SubScore] = []
        metadata: dict[str, Any] = {}

        for item in subscores:
            if name_counts[item.name] == 1:
                final_name = item.name
            else:
                name_usage[item.name] = name_usage.get(item.name, 0) + 1
                final_name = f"{item.name}-{name_usage[item.name]}"

            normalized_weight = (
                item.weight / positive_weight_sum if item.weight > 0 else item.weight
            )
            normalized_subscores.append(
                SubScore(
                    name=final_name,
                    weight=normalized_weight,
                    value=item.value,
                    metadata=item.metadata,
                )
            )
            if item.metadata is not None:
                metadata[final_name] = item.metadata

        reward = float(
            min(
                max(sum(item.value * item.weight for item in normalized_subscores), 0.0),
                1.0,
            )
        )

        return EvaluationResult(
            reward=reward,
            done=True,
            subscores=normalized_subscores,
            info=metadata,
        )


class Grader:
    """Base class for reusable graders that emit ``SubScore`` objects."""

    name: str = "BaseGrader"

    @classmethod
    def grade(cls, weight: float, name: str | None = None, **kwargs: Any) -> SubScore:
        """Run the grader and package the result as a ``SubScore``."""
        result = cls.compute_score(**kwargs)

        if isinstance(result, tuple):
            score, metadata = result
        else:
            score = result
            metadata = {}

        return SubScore(
            name=name or cls.name,
            weight=weight,
            value=float(score),
            metadata={**metadata, "_parameters": _safe_params(kwargs)},
        )

    @classmethod
    def compute_score(cls, *args: Any, **kwargs: Any) -> float | tuple[float, dict[str, Any]]:
        """Compute a score between ``0.0`` and ``1.0``."""
        raise NotImplementedError("Subclasses must implement compute_score")

    @classmethod
    def any(cls, weight: float, subscores: list[SubScore]) -> SubScore:
        """Return a subscore that passes if any input subscore passes."""
        if not subscores:
            raise ValueError("subscores must not be empty")

        return SubScore(
            name=f"{cls.name}_any",
            value=max(subscore.value for subscore in subscores),
            weight=weight,
            metadata={
                "subscores": [subscore.name for subscore in subscores],
                "subscore_metadata": {
                    subscore.name: subscore.metadata
                    for subscore in subscores
                    if subscore.metadata is not None
                },
            },
        )

    @classmethod
    def all(cls, weight: float, subscores: list[SubScore]) -> SubScore:
        """Return a subscore that passes only if all input subscores pass."""
        if not subscores:
            raise ValueError("subscores must not be empty")

        return SubScore(
            name=f"{cls.name}_all",
            value=min(subscore.value for subscore in subscores),
            weight=weight,
            metadata={
                "subscores": [subscore.name for subscore in subscores],
                "subscore_metadata": {
                    subscore.name: subscore.metadata
                    for subscore in subscores
                    if subscore.metadata is not None
                },
            },
        )


class BashGrader(Grader):
    """Run a shell command and score it by exit code."""

    name = "BashGrader"

    @classmethod
    def compute_score(
        cls,
        command: str,
        cwd: str | None = None,
        timeout: int = 60,
        **kwargs: Any,
    ) -> tuple[float, dict[str, Any]]:
        """Run ``command`` via ``bash -lc`` and return score plus execution metadata."""
        del kwargs
        logger.info("Running grader command: %s (cwd=%s, timeout=%ss)", command, cwd, timeout)
        try:
            result = subprocess.run(
                ["/bin/bash", "-lc", command],
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as exc:
            stdout = (
                (exc.stdout or b"").decode(errors="replace")
                if isinstance(exc.stdout, bytes)
                else (exc.stdout or "")
            )
            stderr = (
                (exc.stderr or b"").decode(errors="replace")
                if isinstance(exc.stderr, bytes)
                else (exc.stderr or "")
            )
            return (
                0.0,
                {
                    "exit_code": None,
                    "stdout": stdout,
                    "stderr": stderr,
                    "timed_out": True,
                    "timeout": timeout,
                },
            )

        score = 1.0 if result.returncode == 0 else 0.0
        return (
            score,
            {
                "exit_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
            },
        )
