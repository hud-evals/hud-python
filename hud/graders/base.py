"""The grading contract: ``SubScore``, ``Grade`` and ``Grader``."""

from __future__ import annotations

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

    @model_validator(mode="after")
    def _subscores_explain_reward(self) -> Grade:
        if not self.subscores:
            return self
        weighted = sum(subscore.value * subscore.weight for subscore in self.subscores)
        if abs(weighted - self.reward) > 0.01:
            raise ValueError(
                f"subscores don't explain the reward: value x weight sums to {weighted:.4f}, "
                f"the reward is {self.reward:.4f}; set the reward to the weighted sum, or the "
                "weights to what the reward used"
            )
        return self

    def to_summary(self) -> dict[str, Any]:
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
