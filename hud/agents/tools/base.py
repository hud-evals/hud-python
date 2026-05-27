"""AgentTool + AgentToolSpec.

``AgentTool`` is the provider-facing tool, generic in its ``CapabilityClient``
type. Capability bases (``SSHTool``, ``MCPTool``, ``RFBTool``) bind the
generic and add per-protocol helpers. Provider subclasses declare
``default_spec(model)`` and implement ``to_params`` + ``execute``.

Result formatting (turning a ``MCPToolResult`` into a provider message) lives
on the agent, not on the tool — the agent owns that wire shape.
"""

from __future__ import annotations

import fnmatch
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Generic, TypeVar

from hud.capabilities import CapabilityClient

if TYPE_CHECKING:
    from hud.types import MCPToolResult

ClientT = TypeVar("ClientT", bound=CapabilityClient)


@dataclass(frozen=True)
class AgentToolSpec:
    """Provider tool spec — api id + optional model-version gating."""

    api_type: str
    api_name: str
    supported_models: tuple[str, ...] | None = None

    def supports_model(self, model: str | None) -> bool:
        if not self.supported_models:
            return True
        if not model or model == "unknown":
            return False
        m = model.lower()
        return any(fnmatch.fnmatch(m, p.lower()) for p in self.supported_models)


class AgentTool(ABC, Generic[ClientT]):
    """Provider-facing tool bound to one ``CapabilityClient`` instance.

    Tools only execute — result formatting belongs to the agent.
    """

    name: ClassVar[str]
    #: Runtime dispatch key — set by each capability base.
    client_type: ClassVar[type[CapabilityClient]]

    def __init__(self, *, spec: AgentToolSpec, client: ClientT) -> None:
        self.spec = spec
        self.client: ClientT = client

    @property
    def provider_name(self) -> str:
        """Name advertised to the LLM. Overridden by ``MCPTool``."""
        return self.name

    @classmethod
    def default_spec(cls, model: str) -> AgentToolSpec | None:
        """Return the spec for this model, or ``None`` to skip registration."""
        del model
        return None

    @abstractmethod
    async def execute(self, arguments: dict[str, Any]) -> MCPToolResult: ...

    @abstractmethod
    def to_params(self) -> Any: ...


__all__ = ["AgentTool", "AgentToolSpec", "ClientT"]
