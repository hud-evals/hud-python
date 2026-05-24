"""Common agent-side OpenAI tool support."""

from __future__ import annotations

import copy
import json
import logging
from abc import ABC
from inspect import cleandoc
from typing import TYPE_CHECKING, Any, cast

from mcp import types
from openai.types.responses import (
    FunctionToolParam,
    ResponseFunctionCallOutputItemListParam,
    ResponseInputFileContentParam,
    ResponseInputImageContentParam,
    ResponseInputTextContentParam,
    ResponseInputTextParam,
    ToolParam,
)
from openai.types.responses.response_input_param import FunctionCallOutput

from hud.agents.tools import AgentTool, AgentToolSpec
from hud.utils.strict_schema import ensure_strict_json_schema

if TYPE_CHECKING:
    from openai.types.responses import ResponseInputItemParam

    from hud.types import MCPToolCall, MCPToolResult

logger = logging.getLogger(__name__)

OpenAIToolSpec = AgentToolSpec


class OpenAITool(AgentTool[ToolParam], ABC):
    """Agent-side OpenAI provider tool backed by an environment tool."""

    def format_result(
        self, call: MCPToolCall, result: MCPToolResult
    ) -> ResponseInputItemParam | None:
        """Format a generic provider tool result for the OpenAI Responses API."""
        if not call.id:
            logger.warning("Tool '%s' missing call_id; skipping output.", call.name)
            return None

        output_items: ResponseFunctionCallOutputItemListParam = []
        if result.isError:
            output_items.append(
                ResponseInputTextContentParam(type="input_text", text="[tool_error] true")
            )

        if result.structuredContent is not None:
            output_items.append(
                ResponseInputTextContentParam(
                    type="input_text",
                    text=json.dumps(result.structuredContent, default=str),
                )
            )

        for block in result.content:
            match block:
                case types.TextContent():
                    output_items.append(
                        ResponseInputTextContentParam(type="input_text", text=block.text)
                    )
                case types.ImageContent():
                    mime_type = getattr(block, "mimeType", "image/png")
                    output_items.append(
                        ResponseInputImageContentParam(
                            type="input_image",
                            image_url=f"data:{mime_type};base64,{block.data}",
                        )
                    )
                case types.ResourceLink():
                    output_items.append(
                        ResponseInputFileContentParam(type="input_file", file_url=str(block.uri))
                    )
                case types.EmbeddedResource(resource=types.TextResourceContents() as resource):
                    output_items.append(
                        ResponseInputTextContentParam(type="input_text", text=resource.text)
                    )
                case types.EmbeddedResource(resource=types.BlobResourceContents() as resource):
                    output_items.append(
                        ResponseInputFileContentParam(type="input_file", file_data=resource.blob)
                    )
                case types.EmbeddedResource():
                    logger.warning("Unknown resource type: %s", type(block.resource))
                case _:
                    logger.warning("Unknown content block type: %s", type(block))

        if not output_items:
            output_items.append(ResponseInputTextParam(type="input_text", text=""))

        return FunctionCallOutput(type="function_call_output", call_id=call.id, output=output_items)


class OpenAIFunctionTool(OpenAITool):
    """Generic OpenAI function tool backed by an MCP tool."""

    name = "function"
    capability = "function"

    def __init__(
        self,
        *,
        env_tool_name: str,
        description: str,
        parameters: dict[str, Any],
    ) -> None:
        super().__init__(
            env_tool_name=env_tool_name,
            spec=OpenAIToolSpec(api_type="function", api_name=env_tool_name),
        )
        self.description = description
        self.parameters = parameters

    @classmethod
    def from_tool(cls, tool: types.Tool) -> OpenAIFunctionTool | None:
        if tool.description is None:
            raise ValueError(
                cleandoc(f"""MCP tool {tool.name} requires both a description and inputSchema.
                Add these by:
                1. Adding a docstring to your @mcp.tool decorated function for the description
                2. Using pydantic Field() annotations on function parameters for the schema
                """)
            )

        try:
            parameters = ensure_strict_json_schema(copy.deepcopy(tool.inputSchema))
        except Exception as e:
            logger.warning("Failed to convert tool '%s' schema to strict: %s", tool.name, e)
            return None

        return cls(
            env_tool_name=tool.name,
            description=tool.description,
            parameters=parameters,
        )

    @property
    def provider_name(self) -> str:
        return self.env_tool_name

    def to_params(self) -> ToolParam:
        return cast(
            "ToolParam",
            FunctionToolParam(
                type="function",
                name=self.provider_name,
                description=self.description,
                parameters=self.parameters,
                strict=True,
            ),
        )
