"""Shared hosted-tool machinery configured by agent harnesses."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, TypeVar

from hud.agents.tools.base import model_matches

HostedToolParamT_co = TypeVar("HostedToolParamT_co", covariant=True)


@dataclass(frozen=True, kw_only=True)
class HostedTool(ABC, Generic[HostedToolParamT_co]):
    """Provider-side tool activated only through explicit agent config."""

    supported_models: tuple[str, ...] | None = None

    def supports_model(self, model: str | None) -> bool:
        return model_matches(model, self.supported_models)

    @abstractmethod
    def to_params(self) -> HostedToolParamT_co:
        raise NotImplementedError
