"""Common agent-side Claude tool support."""

from __future__ import annotations

from dataclasses import dataclass
from inspect import cleandoc
from typing import TYPE_CHECKING, Any, Literal, cast

import mcp.types as types
from anthropic.types.beta import (
    BetaBase64ImageSourceParam,
    BetaBase64PDFSourceParam,
    BetaImageBlockParam,
    BetaMessageParam,
    BetaPlainTextSourceParam,
    BetaRequestDocumentBlockParam,
    BetaTextBlockParam,
    BetaToolParam,
    BetaToolResultBlockParam,
)

from hud.agents.tools import AgentTool, AgentToolSpec

if TYPE_CHECKING:
    from anthropic.types.beta import BetaToolUnionParam

    from hud.types import MCPToolCall, MCPToolResult
else:
    BetaToolUnionParam = Any

ClaudeImageMediaType = Literal["image/jpeg", "image/png", "image/gif", "image/webp"]
ClaudeToolResultContent = BetaTextBlockParam | BetaImageBlockParam | BetaRequestDocumentBlockParam


@dataclass(frozen=True)
class ClaudeToolSpec(AgentToolSpec):
    """Claude provider tool definition."""

    beta: str | None = None


class ClaudeTool(AgentTool["BetaToolUnionParam"]):
    """Agent-side Claude provider tool backed by an environment tool."""

    def __init__(self, *, env_tool_name: str, spec: ClaudeToolSpec) -> None:
        super().__init__(env_tool_name=env_tool_name, spec=spec)
        self.spec: ClaudeToolSpec = spec

    @property
    def required_beta(self) -> str | None:
        return self.spec.beta

    def format_result(self, call: MCPToolCall, result: MCPToolResult) -> BetaMessageParam | None:
        tool_use_id = call.id
        if not tool_use_id:
            return None

        result_content = result.content
        if result.isError:
            error_msg = next(
                (
                    content.text
                    for content in result.content
                    if isinstance(content, types.TextContent)
                ),
                "Tool execution failed",
            )
            result_content = [types.TextContent(type="text", text=f"Error: {error_msg}")]

        claude_blocks: list[ClaudeToolResultContent] = []
        sibling_docs: list[BetaRequestDocumentBlockParam] = []
        enable_citations = bool(getattr(call.meta, "enable_citations", False))
        for content in result_content:
            citation_doc = None
            match content:
                case types.TextContent():
                    block = BetaTextBlockParam(type="text", text=content.text)
                    if enable_citations and not result.isError:
                        citation_doc = BetaRequestDocumentBlockParam(
                            type="document",
                            source=BetaPlainTextSourceParam(
                                type="text",
                                media_type="text/plain",
                                data=content.text,
                            ),
                            title=call.name,
                            citations={"enabled": True},
                        )
                case types.ImageContent():
                    block = BetaImageBlockParam(
                        type="image",
                        source=BetaBase64ImageSourceParam(
                            type="base64",
                            media_type=cast("ClaudeImageMediaType", content.mimeType),
                            data=content.data,
                        ),
                    )
                case types.EmbeddedResource(
                    resource=types.BlobResourceContents(mimeType="application/pdf") as resource
                ):
                    block = BetaRequestDocumentBlockParam(
                        type="document",
                        source=BetaBase64PDFSourceParam(
                            type="base64",
                            media_type="application/pdf",
                            data=resource.blob,
                        ),
                    )
                    if enable_citations and not result.isError:
                        citation_doc = BetaRequestDocumentBlockParam(
                            type="document",
                            source=block["source"],
                            citations={"enabled": True},
                        )
                case _:
                    raise ValueError(f"Unknown content block type: {type(content)}")
            claude_blocks.append(block)
            if citation_doc is not None:
                sibling_docs.append(citation_doc)

        return BetaMessageParam(
            role="user",
            content=[
                BetaToolResultBlockParam(
                    type="tool_result",
                    tool_use_id=tool_use_id,
                    content=claude_blocks,
                ),
                *sibling_docs,
            ],
        )


class ClaudeFunctionTool(ClaudeTool):
    """Regular environment tool exposed as a Claude function tool."""

    name = "function"
    capability = "function"

    def __init__(
        self,
        *,
        env_tool_name: str,
        description: str,
        input_schema: dict[str, Any],
    ) -> None:
        super().__init__(
            env_tool_name=env_tool_name,
            spec=ClaudeToolSpec(api_type="function", api_name=env_tool_name),
        )
        self.description = description
        self.input_schema = input_schema

    @classmethod
    def from_tool(cls, tool: types.Tool) -> ClaudeFunctionTool:
        if tool.description is None:
            raise ValueError(
                cleandoc(f"""MCP tool {tool.name} requires both a description and inputSchema.
                Add these by:
                1. Adding a docstring to your @mcp.tool decorated function for the description
                2. Using pydantic Field() annotations on function parameters for the schema
                """)
            )
        return cls(
            env_tool_name=tool.name,
            description=tool.description,
            input_schema=tool.inputSchema,
        )

    @property
    def provider_name(self) -> str:
        return self.env_tool_name

    def to_params(self) -> BetaToolUnionParam:
        return BetaToolParam(
            name=self.provider_name,
            description=self.description,
            input_schema=self.input_schema,
            eager_input_streaming=True,
        )
