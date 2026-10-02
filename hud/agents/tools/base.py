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
import hashlib
import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar, Generic, TypeVar

import mcp.types as mcp_types

from hud.capabilities import CapabilityClient
from hud.types import MCPToolResult

ClientT = TypeVar("ClientT", bound=CapabilityClient)
_PROVIDER_TOOL_NAME_PATTERN = re.compile(r"[^A-Za-z0-9_-]+")


def provider_tool_name(name: str) -> str:
    sanitized = _PROVIDER_TOOL_NAME_PATTERN.sub("_", name).strip("_") or "tool"
    if sanitized == name and len(sanitized) <= 64:
        return sanitized
    digest = hashlib.sha256(name.encode()).hexdigest()[:8]
    prefix = sanitized[: 64 - len(digest) - 1].rstrip("_") or "tool"
    return f"{prefix}_{digest}"


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


def truncation_notice(total: int, limit: int) -> str:
    """The marker left where an oversized tool result was cut."""
    return (
        f"\n\n[... output truncated: this tool result had {total:,} characters, over the "
        f"{limit:,}-character limit. Read large files in smaller line ranges, or narrow "
        f"command output with grep, head, tail, or sed -n ...]\n\n"
    )


def _block_text(block: mcp_types.ContentBlock) -> str | None:
    match block:
        case mcp_types.TextContent():
            return block.text
        case mcp_types.EmbeddedResource(resource=mcp_types.TextResourceContents() as resource):
            return resource.text
        case _:
            return None


def _with_text(block: mcp_types.ContentBlock, text: str) -> mcp_types.ContentBlock:
    match block:
        case mcp_types.TextContent():
            return block.model_copy(update={"text": text})
        case mcp_types.EmbeddedResource(resource=mcp_types.TextResourceContents() as resource):
            return block.model_copy(update={"resource": resource.model_copy(update={"text": text})})
        case _:
            raise TypeError(f"{type(block).__name__} carries no text")


def bound_tool_result(result: MCPToolResult, limit: int) -> MCPToolResult:
    """The tool result as the model receives it, within ``limit`` characters.

    The model sees a result's ``content`` (text blocks and embedded text
    resources count), or its ``structuredContent`` as JSON when ``content`` is
    empty; ``structuredContent`` beside content is dropped. An oversized result
    keeps the head and tail of its text around a :func:`truncation_notice`; a
    structured-only result becomes one bounded text block. Images and binary
    resources pass through.
    """
    if not result.content and result.structuredContent is not None:
        structured = json.dumps(result.structuredContent, default=str)
        if len(structured) <= limit:
            return result
        text = mcp_types.TextContent(type="text", text=structured)
        return bound_tool_result(
            result.model_copy(update={"content": [text], "structuredContent": None}), limit
        )
    if result.structuredContent is not None:
        result = result.model_copy(update={"structuredContent": None})

    texts = [_block_text(block) for block in result.content]
    total = sum(len(text) for text in texts if text is not None)
    if total <= limit:
        return result

    notice = truncation_notice(total, limit)
    keep = limit - len(notice)
    head_left, tail_left = (keep + 1) // 2, keep // 2
    heads: list[int] = []
    for text in texts:
        taken = min(len(text or ""), head_left)
        heads.append(taken)
        head_left -= taken
    tails = [0] * len(texts)
    for index in reversed(range(len(texts))):
        taken = min(len(texts[index] or "") - heads[index], tail_left)
        tails[index] = taken
        tail_left -= taken
    cut = next(
        index for index, text in enumerate(texts) if text is not None and heads[index] < len(text)
    )

    bounded: list[mcp_types.ContentBlock] = []
    for index, (block, text) in enumerate(zip(result.content, texts, strict=True)):
        if text is None:
            bounded.append(block)
            continue
        kept = (
            text[: heads[index]]
            + (notice if index == cut else "")
            + text[len(text) - tails[index] :]
        )
        if kept:
            bounded.append(_with_text(block, kept))
    return result.model_copy(update={"content": bounded})


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

    def bound_result(self, result: MCPToolResult, limit: int) -> MCPToolResult:
        """Fit one of this tool's results within ``limit`` model-visible characters.

        Tools whose provider sends a structured payload instead of ``content``
        override this to truncate inside that payload.
        """
        return bound_tool_result(result, limit)

    @abstractmethod
    def to_params(self) -> Any: ...


__all__ = [
    "AgentTool",
    "AgentToolSpec",
    "ClientT",
    "bound_tool_result",
    "provider_tool_name",
    "result_text",
    "tool_err",
    "tool_ok",
    "truncation_notice",
]
