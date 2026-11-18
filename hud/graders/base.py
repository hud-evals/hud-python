"""The grading contract: ``SubScore``, ``Grade``, ``EvaluationResult`` and ``Grader``."""

from __future__ import annotations

import warnings
from typing import Any, cast

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    SerializationInfo,
    SerializerFunctionWrapHandler,
    ValidationError,
    model_serializer,
    model_validator,
)

from hud.utils.exceptions import HudDeprecationWarning
from hud.utils.serialization import json_safe_dict

_SUMMARY_SERIALIZATION_CONTEXT = "hud_summary"


class SubScore(BaseModel):
    """One node in the grade breakdown tree.

    A leaf subscore is a single measurement. A parent preserves the subscores
    used to derive its value in ``children``.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., description="Name of this subscore component")
    weight: float = Field(
        default=1.0,
        description="Weight of this subscore (for weighted average). "
        "Negative weights represent penalties.",
    )
    value: float = Field(..., ge=0.0, le=1.0, description="Value of this subscore, 0.0 to 1.0")
    children: list[SubScore] | None = Field(
        default=None,
        description="Input subscores this node was combined from",
    )
    # Environments built before ``info`` was named send it as ``metadata``.
    info: dict[str, Any] | None = Field(
        default=None,
        validation_alias=AliasChoices("info", "metadata"),
        description="Grader-specific details recorded on the evaluation span",
    )

    @model_serializer(mode="wrap")
    def _serialize(
        self,
        handler: SerializerFunctionWrapHandler,
        serialization_info: SerializationInfo,
    ) -> dict[str, Any]:
        data = cast("dict[str, Any]", handler(self))
        if (serialization_info.context or {}).get(_SUMMARY_SERIALIZATION_CONTEXT):
            data.pop("info", None)
        return data

    def to_summary(self) -> dict[str, Any]:
        """Serialize the score breakdown without span-only info."""
        return self.model_dump(
            mode="json",
            context={_SUMMARY_SERIALIZATION_CONTEXT: True},
        )


class Grade(BaseModel):
    """A task's verdict; it serializes as the ``tasks.grade`` frame, extra keys included."""

    model_config = ConfigDict(
        extra="allow",
        validate_by_name=True,
        validate_by_alias=True,
        serialize_by_alias=True,
    )

    reward: float = Field(default=0.0, alias="score", strict=True, allow_inf_nan=False)
    done: bool = True
    content: str | None = None
    info: dict[str, Any] = Field(default_factory=dict)
    is_error: bool = Field(default=False, alias="isError")
    subscores: list[SubScore] | None = None

    @property
    def evaluation(self) -> dict[str, Any]:
        """The frame the platform records, without ``info`` at any level."""
        frame = self.model_dump(mode="json", exclude={"info"})
        if self.subscores is not None:
            frame["subscores"] = [subscore.to_summary() for subscore in self.subscores]
        return frame

    @classmethod
    def ungraded(cls) -> Grade:
        """The grade of a trace that has not been graded."""
        return cls(is_error=True, content="not graded")


def grade_problems(error: ValidationError) -> str:
    """One line per invalid ``Grade`` field: its frame key, the problem and the input."""
    return "; ".join(
        f"{'.'.join(map(str, item['loc']))!r}: {item['msg']}, got {item['input']!r}"
        for item in error.errors()
    )


class EvaluationResult(BaseModel):
    """Deprecated: a task's grade, now ``hud.Grade``. Construction warns until hud v0.7.0."""

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

    def __init__(self, **data: Any) -> None:
        warnings.warn(
            "`EvaluationResult` is deprecated and will be removed in hud v0.7.0; "
            "yield a `hud.Grade` (`is_error=` in place of `isError=`).",
            HudDeprecationWarning,
            stacklevel=2,
        )
        super().__init__(**data)

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
        return cls(reward=value, done=True)


class Grader:
    """Async base class for reusable graders.

    Subclasses implement ``compute_score`` (async). ``grade`` runs it, applies the caller's
    ``weight`` and optional ``name``, and records the call's parameters in ``info``.
    """

    name: str = "BaseGrader"

    @classmethod
    async def grade(cls, weight: float, name: str | None = None, **kwargs: Any) -> SubScore:
        """Run the grader and package the result as a ``SubScore``."""
        score = await cls.compute_score(**kwargs)
        if not isinstance(score, SubScore):
            value, info = score if isinstance(score, tuple) else (score, None)
            score = SubScore(name=cls.name, value=float(value), info=info)
        return score.model_copy(
            update={
                "name": name or cls.name,
                "weight": weight,
                "info": {**(score.info or {}), "_parameters": json_safe_dict(kwargs)},
            }
        )

    @classmethod
    async def compute_score(cls, **kwargs: Any) -> SubScore | float | tuple[float, dict[str, Any]]:
        """Score between ``0.0`` and ``1.0``; a bare float or ``(float, info)`` is also accepted."""
        raise NotImplementedError("Subclasses must implement compute_score")
