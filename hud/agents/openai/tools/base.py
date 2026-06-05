"""OpenAI tool spec + result formatting."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, cast

import mcp.types as types
from openai.types.responses import (
    ResponseFunctionCallOutputItemListParam,
    ResponseInputFileContentParam,
    ResponseInputImageContentParam,
    ResponseInputTextContentParam,
    ResponseInputTextParam,
)
from openai.types.responses.response_input_param import FunctionCallOutput, ResponseInputItemParam

from hud.agents.tools.base import AgentToolSpec

if TYPE_CHECKING:
    from hud.types import MCPToolCall, MCPToolResult

logger = logging.getLogger(__name__)

OpenAIToolSpec = AgentToolSpec


def format_openai_result(call: MCPToolCall, result: MCPToolResult) -> ResponseInputItemParam | None:
    """Format a generic tool result for the OpenAI Responses API."""
    if not call.id:
        logger.warning("Tool '%s' missing call_id; skipping output.", call.name)
        return None

    output_items: ResponseFunctionCallOutputItemListParam = []
    if result.isError:
        output_items.append(
            ResponseInputTextContentParam(type="input_text", text="[tool_error] true"),
        )

    if result.structuredContent is not None:
        output_items.append(
            ResponseInputTextContentParam(
                type="input_text",
                text=json.dumps(result.structuredContent, default=str),
            ),
        )

    for block in result.content:
        match block:
            case types.TextContent():
                output_items.append(
                    ResponseInputTextContentParam(type="input_text", text=block.text),
                )
            case types.ImageContent():
                mime_type = getattr(block, "mimeType", "image/png")
                output_items.append(
                    ResponseInputImageContentParam(
                        type="input_image",
                        image_url=f"data:{mime_type};base64,{block.data}",
                    ),
                )
            case types.ResourceLink():
                output_items.append(
                    ResponseInputFileContentParam(type="input_file", file_url=str(block.uri)),
                )
            case types.EmbeddedResource(resource=types.TextResourceContents() as resource):
                output_items.append(
                    ResponseInputTextContentParam(type="input_text", text=resource.text),
                )
            case types.EmbeddedResource(resource=types.BlobResourceContents() as resource):
                output_items.append(
                    ResponseInputFileContentParam(type="input_file", file_data=resource.blob),
                )
            case _:
                logger.warning("Unknown content block type: %s", type(block))

    if not output_items:
        output_items.append(ResponseInputTextParam(type="input_text", text=""))

    return cast(
        "ResponseInputItemParam",
        FunctionCallOutput(type="function_call_output", call_id=call.id, output=output_items),
    )


__all__ = ["OpenAIToolSpec", "format_openai_result"]
