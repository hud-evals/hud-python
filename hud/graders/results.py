"""Grading result shapes: ``SubScore`` and ``EvaluationResult``."""

from __future__ import annotations

import warnings
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class SubScore(BaseModel):
    """One node in the grade breakdown tree.

    A leaf subscore is a single measurement: a grader run, or one rubric
    criterion (value ``0``/``1``, with ``reason`` justifying the verdict).
    A node with ``aggregation`` derived its value from ``children``, which are
    kept verbatim so the full grading history stays inspectable.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., description="Name of this subscore component")
    weight: float = Field(
        default=1.0,
        description="Weight of this subscore (for weighted average). "
        "Negative weights represent penalties.",
    )
    value: float = Field(..., ge=0.0, le=1.0, description="Value of this subscore, 0.0 to 1.0")
    reason: str | None = Field(
        default=None, description="The grader's justification for this value"
    )
    aggregation: Literal["any", "all", "rubric"] | None = Field(
        default=None,
        description="How value was derived from children: 'any' (max), 'all' (min), "
        "or 'rubric' (weighted pass-fraction). None for leaf subscores.",
    )
    children: list[SubScore] | None = Field(
        default=None,
        description="Input subscores this node was combined from",
    )
    metadata: dict[str, Any] | None = Field(
        default=None,
        description="Grader-specific details (parameters, stdout, judge model, ...)",
    )

    @property
    def score(self) -> float:
        """Alias for value. Deprecated — use .value instead."""
        return self.value

    @model_validator(mode="after")
    def _check_aggregation(self) -> SubScore:
        if self.aggregation is None:
            return self
        if not self.children:
            raise ValueError(f"aggregation={self.aggregation!r} requires children")
        if self.aggregation == "any":
            expected = max(c.value for c in self.children)
        elif self.aggregation == "all":
            expected = min(c.value for c in self.children)
        else:
            expected = _rubric_value(self.children)
        if abs(expected - self.value) > 0.01:
            warnings.warn(
                f"SubScore {self.name!r}: value={self.value:.4f} disagrees with "
                f"{self.aggregation}(children)={expected:.4f}",
                stacklevel=2,
            )
        return self


def _rubric_value(subscores: list[SubScore]) -> float:
    """Weighted pass-fraction: sum(value*weight) over positive weight, clamped to 0-1.

    Negative weights are penalties: on those children, value ``1.0`` means the
    error is present and subtracts. An all-negative rubric starts at ``1.0``.
    """
    total_positive = sum(max(0.0, s.weight) for s in subscores)
    total_negative = sum(abs(s.weight) for s in subscores if s.weight < 0)
    weighted_sum = sum(s.value * s.weight for s in subscores)
    if total_positive > 0:
        return max(0.0, min(1.0, weighted_sum / total_positive))
    if total_negative > 0:
        return max(0.0, min(1.0, 1.0 + weighted_sum / total_negative))
    return 0.0


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
