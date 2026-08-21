from __future__ import annotations

from typing import Annotated, TypeAlias, TypeVar

from pydantic import BaseModel, ConfigDict, Field

T = TypeVar("T")


class DataFileRef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    file_id: str = Field(description="HUD data-file id")
    path: str | None = Field(default=None, description="Environment-owned destination path")


PromptArg: TypeAlias = Annotated[
    str,
    Field(json_schema_extra={"x-hud-hint": "prompt"}),
]

DataFileArg: TypeAlias = Annotated[
    T,
    Field(json_schema_extra={"x-hud-hint": "data-file"}),
]

DataFilesArg: TypeAlias = Annotated[
    list[T],
    Field(json_schema_extra={"x-hud-hint": "data-files"}),
]

GradingArg: TypeAlias = Annotated[
    list[T],
    Field(json_schema_extra={"x-hud-hint": "grading"}),
]


__all__ = ["DataFileArg", "DataFileRef", "DataFilesArg", "GradingArg", "PromptArg"]
