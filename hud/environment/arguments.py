"""Retired argument-hint helpers, served with a warning until hud v0.7.0.

A template argument gets its console editor from an ``x-hud-hint`` in its JSON
Schema; annotate the argument with ``Field(json_schema_extra={"x-hud-hint": ...})``.
"""

from __future__ import annotations

from typing import Annotated, Any, TypeAlias, TypeVar

from pydantic import ConfigDict, Field, create_model

from hud.utils.deprecation import Retired, resolve_retired

_T = TypeVar("_T")

_DataFileRef = create_model(
    "DataFileRef",
    __config__=ConfigDict(extra="forbid"),
    __module__=__name__,
    file_id=(str, Field(description="HUD data-file id")),
    path=(str | None, Field(default=None, description="Environment-owned destination path")),
)


_PromptArg: TypeAlias = Annotated[str, Field(json_schema_extra={"x-hud-hint": "prompt"})]
_DataFileArg: TypeAlias = Annotated[_T, Field(json_schema_extra={"x-hud-hint": "data-file"})]
_DataFilesArg: TypeAlias = Annotated[
    list[_T], Field(json_schema_extra={"x-hud-hint": "data-files"})
]
_GradingArg: TypeAlias = Annotated[list[_T], Field(json_schema_extra={"x-hud-hint": "grading"})]


def _hint(hint: str) -> str:
    return f'annotate the argument with `Field(json_schema_extra={{"x-hud-hint": "{hint}"}})`'


RETIRED = {
    "DataFileRef": Retired(
        lambda: _DataFileRef, "define a model with `file_id` and an optional `path`"
    ),
    "PromptArg": Retired(lambda: _PromptArg, _hint("prompt")),
    "DataFileArg": Retired(lambda: _DataFileArg, _hint("data-file")),
    "DataFilesArg": Retired(lambda: _DataFilesArg, _hint("data-files")),
    "GradingArg": Retired(lambda: _GradingArg, _hint("grading")),
}


def __getattr__(name: str) -> Any:
    return resolve_retired(__name__, name, RETIRED)
