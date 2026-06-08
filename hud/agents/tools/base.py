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
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar, Generic, TypeVar, cast

import mcp.types as mcp_types

from hud.capabilities import CapabilityClient
from hud.types import MCPToolResult

logger = logging.getLogger(__name__)

ClientT = TypeVar("ClientT", bound=CapabilityClient)


def tool_ok(text: str) -> MCPToolResult:
    """Build a success MCPToolResult with one text block."""
    return MCPToolResult(content=[mcp_types.TextContent(type="text", text=text)])


def tool_err(text: str) -> MCPToolResult:
    """Build an error MCPToolResult with one text block."""
    return MCPToolResult(content=[mcp_types.TextContent(type="text", text=text)], isError=True)


def result_text(result: MCPToolResult) -> str:
    """Extract concatenated text from a MCPToolResult's TextContent blocks."""
    return "".join(
        block.text for block in result.content if isinstance(block, mcp_types.TextContent)
    )


def last_image_data(result: MCPToolResult) -> str | None:
    """Return the base64 data of the last image block in a result, or ``None``."""
    for block in reversed(result.content):
        if isinstance(block, mcp_types.ImageContent):
            return block.data
    return None


def parse_tool_arguments(raw: str | None, name: str | None = None) -> dict[str, Any]:
    """Parse model-emitted tool-call argument JSON, tolerating malformed output.

    Providers occasionally emit truncated or invalid JSON. Falling back to empty
    arguments lets the tool surface a normal validation error the model can retry,
    instead of raising ``JSONDecodeError`` and collapsing the whole rollout.
    """
    if not raw:
        return {}
    try:
        parsed: Any = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("invalid tool-call JSON for %r; using empty arguments", name)
        return {}
    return cast("dict[str, Any]", parsed) if isinstance(parsed, dict) else {}


def model_matches(model: str | None, patterns: tuple[str, ...] | None) -> bool:
    """fnmatch ``model`` against provider version ``patterns``.

    ``None``/empty patterns match any model; an unknown/missing model matches
    nothing when patterns are present. Shared by ``AgentToolSpec`` and
    ``HostedTool`` so the gating rule lives in one place.
    """
    if not patterns:
        return True
    if not model or model == "unknown":
        return False
    m = model.lower()
    return any(fnmatch.fnmatch(m, p.lower()) for p in patterns)


@dataclass(frozen=True)
class AgentToolSpec:
    """Provider tool spec — api id + optional model-version gating."""

    api_type: str
    api_name: str
    supported_models: tuple[str, ...] | None = None

    def supports_model(self, model: str | None) -> bool:
        return model_matches(model, self.supported_models)


class AgentTool(ABC, Generic[ClientT]):
    """Provider-facing tool bound to one ``CapabilityClient`` instance.

    Tools only execute — result formatting belongs to the agent.
    """

    name: ClassVar[str]
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

    @classmethod
    async def bind(
        cls,
        *,
        model: str,
        connections: dict[str, CapabilityClient],
    ) -> tuple[dict[str, AgentTool[Any]], list[Any]]:
        """Bind this provider tool class to its capability client for one run."""
        client = connections.get(cls.client_type.protocol)
        if client is None:
            return {}, []
        spec = cls.default_spec(model)
        if spec is None:
            return {}, []
        tool = cls(spec=spec, client=cast("ClientT", client))
        return {tool.provider_name: cast("AgentTool[Any]", tool)}, [tool.to_params()]

    @abstractmethod
    async def execute(self, arguments: dict[str, Any]) -> MCPToolResult: ...

    @abstractmethod
    def to_params(self) -> Any: ...


__all__ = [
    "AgentTool",
    "AgentToolSpec",
    "ClientT",
    "last_image_data",
    "model_matches",
    "parse_tool_arguments",
    "result_text",
    "tool_err",
    "tool_ok",
]
