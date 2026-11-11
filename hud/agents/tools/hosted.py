"""Shared hosted-tool machinery configured by agent harnesses."""

from __future__ import annotations

import fnmatch
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Generic, TypeVar, cast

HostedToolParamT_co = TypeVar("HostedToolParamT_co", covariant=True)
ToolParamT = TypeVar("ToolParamT", bound=Mapping[str, Any])


@dataclass(frozen=True, kw_only=True)
class HostedTool(ABC, Generic[HostedToolParamT_co]):
    """Provider-side tool activated only through explicit agent config."""

    supported_models: tuple[str, ...] | None = None

    def supports_model(self, model: str | None) -> bool:
        if not self.supported_models:
            return True
        if not model or model == "unknown":
            return False
        model_lower = model.lower()
        return any(
            fnmatch.fnmatch(model_lower, pattern.lower()) for pattern in self.supported_models
        )

    @abstractmethod
    def to_params(self) -> HostedToolParamT_co:
        raise NotImplementedError


def defer_past_threshold(
    tools: list[ToolParamT], threshold: int, deferrable: Callable[[ToolParamT], bool]
) -> list[ToolParamT]:
    """``tools`` with the deferrable ones marked ``defer_loading`` once more than
    ``threshold`` of them are offered, so the model finds them through tool search."""
    if sum(1 for tool in tools if deferrable(tool)) <= threshold:
        return tools
    return [
        cast("ToolParamT", {**tool, "defer_loading": True}) if deferrable(tool) else tool
        for tool in tools
    ]
