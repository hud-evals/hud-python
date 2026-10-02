"""AgentTool + AgentToolSpec.

``AgentTool`` is the provider-facing tool, generic in its ``CapabilityClient``
type. Capability bases (``SSHTool``, ``MCPTool``, ``RFBTool``) bind the
generic and add per-protocol helpers. Provider subclasses declare
``default_spec(model)`` and implement ``to_params`` + ``execute``.

Result formatting (turning a ``ToolResult`` into a provider message) lives
on the agent, not on the tool — the agent owns that wire shape.
"""

from __future__ import annotations

import fnmatch
import hashlib
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Generic, Self, TypeVar

from hud.capabilities import CapabilityClient
from hud.types import TextContent, ToolResult

if TYPE_CHECKING:
    from hud.agents.types import ToolAgentConfig

ClientT = TypeVar("ClientT", bound=CapabilityClient)
_PROVIDER_TOOL_NAME_PATTERN = re.compile(r"[^A-Za-z0-9_-]+")


def provider_tool_name(name: str) -> str:
    sanitized = _PROVIDER_TOOL_NAME_PATTERN.sub("_", name).strip("_") or "tool"
    if sanitized == name and len(sanitized) <= 64:
        return sanitized
    digest = hashlib.sha256(name.encode()).hexdigest()[:8]
    prefix = sanitized[: 64 - len(digest) - 1].rstrip("_") or "tool"
    return f"{prefix}_{digest}"


def tool_ok(text: str) -> ToolResult:
    """Build a success ToolResult with one text block."""
    return ToolResult(content=[TextContent(type="text", text=text)])


def tool_err(text: str) -> ToolResult:
    """Build an error ToolResult with one text block."""
    return ToolResult(content=[TextContent(type="text", text=text)], isError=True)


def result_text(result: ToolResult) -> str:
    """Extract concatenated text from a ToolResult's TextContent blocks."""
    return "".join(block.text for block in result.content if isinstance(block, TextContent))


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

    @classmethod
    def from_config(cls, *, spec: AgentToolSpec, client: ClientT, config: ToolAgentConfig) -> Self:
        """This tool as an agent with *config* runs it; a tool reads what it needs from it."""
        del config
        return cls(spec=spec, client=client)

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
    async def execute(self, arguments: dict[str, Any]) -> ToolResult: ...

    @abstractmethod
    def to_params(self) -> Any: ...
