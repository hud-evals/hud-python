"""Grading result shapes: ``SubScore`` and ``EvaluationResult``."""

from __future__ import annotations

import warnings
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class SubScore(BaseModel):
    """Individual subscore for debugging and transparency.

    SubScores allow breaking down the final reward into component parts,
    making it easier to understand what contributed to the evaluation.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., description="Name of this subscore component")
    weight: float = Field(
        default=1.0,
        description="Weight of this subscore (for weighted average). "
        "Negative weights represent penalties.",
    )
    value: float = Field(..., ge=0.0, le=1.0, description="Value of this subscore, 0.0 to 1.0")
    metadata: dict[str, Any] | None = Field(default=None, exclude=True)

    @property
    def score(self) -> float:
        """Alias for value. Deprecated — use .value instead."""
        return self.value


class EvaluationResult(BaseModel):
    """Result of a task's evaluate phase.

    In eval mode, populate reward and subscores for scoring.
    In production, use content and info for diagnostics and stats.
    """

    reward: float = Field(default=0.0, description="Final score, usually 0.0 to 1.0")
    done: bool = Field(default=True, description="Whether the task/episode is complete")
    content: str | None = Field(default=None, description="Human-readable explanation")
    info: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    isError: bool = Field(default=False, description="Whether the evaluation itself failed")
    subscores: list[SubScore] | None = Field(
        default=None,
        description="Optional breakdown of score components for debugging",
    )

    model_config = ConfigDict(extra="allow")

    @model_validator(mode="after")
    def _check_subscores(self) -> EvaluationResult:
        if not self.subscores:
            return self
        names = [s.name for s in self.subscores]
        dupes = [n for n in names if names.count(n) > 1]
        if dupes:
            warnings.warn(f"Duplicate subscore names: {set(dupes)}", stacklevel=2)
        pos_weight_sum = sum(s.weight for s in self.subscores if s.weight > 0)
        if abs(pos_weight_sum - 1.0) > 0.01:
            warnings.warn(
                f"Positive subscore weights should sum to ~1.0 (got {pos_weight_sum:.4f}). "
                f"Weights represent proportional contributions to the reward.",
                stacklevel=2,
            )
        weighted_sum = sum(s.value * s.weight for s in self.subscores)
        if abs(weighted_sum - self.reward) > 0.01:
            warnings.warn(
                f"Subscores don't match reward: "
                f"sum(value*weight)={weighted_sum:.4f} but reward={self.reward:.4f}",
                stacklevel=2,
            )
        return self

    @classmethod
    def from_float(cls, value: float) -> EvaluationResult:
        """Create an EvaluationResult from a simple float reward."""
        return cls(reward=value, done=True)


__all__ = ["EvaluationResult", "SubScore"]
