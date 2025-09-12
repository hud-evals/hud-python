"""Generic OpenAI chat-completions agent.

This class provides the minimal glue required to connect any endpoint that
implements the OpenAI compatible *chat.completions* API with MCP tool calling
through the existing :class:`hud.agent.MCPAgent` scaffolding.

Key points:
- Stateless, no special server-side conversation state is assumed.
- Accepts an :class:`openai.AsyncOpenAI` client, caller can supply their own
  base_url / api_key (e.g. llama.cpp, together.ai, …)
- All HUD features (step_count, OTel spans, tool filtering, screenshots, …)
  come from the ``MCPAgent`` base class, we only implement the three abstract
  methods
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any, ClassVar, cast

import mcp.types as types

from hud import instrument
from hud.types import AgentResponse, MCPToolCall, MCPToolResult

from .base import MCPAgent

if TYPE_CHECKING:
    from openai import AsyncOpenAI
    from openai.types.chat import ChatCompletionToolParam

logger = logging.getLogger(__name__)


class GenericOpenAIChatAgent(MCPAgent):
    """MCP-enabled agent that speaks the OpenAI *chat.completions* protocol."""

    metadata: ClassVar[dict[str, Any]] = {}

    def __init__(
        self,
        *,
        openai_client: AsyncOpenAI,
        model_name: str = "gpt-4o-mini",
        parallel_tool_calls: bool = False,
        completion_kwargs: dict[str, Any] | None = None,
        **agent_kwargs: Any,
    ) -> None:
        # Accept base-agent settings via **agent_kwargs (e.g., mcp_client, system_prompt, etc.)
        super().__init__(**agent_kwargs)
        self.oai = openai_client
        self.model_name = model_name
        self.parallel_tool_calls = parallel_tool_calls
        self.completion_kwargs: dict[str, Any] = completion_kwargs or {}
        self.conversation_history = []

    @staticmethod
    def _oai_to_mcp(tool_call: Any) -> MCPToolCall:  # type: ignore[valid-type]
        """Convert an OpenAI ``tool_call`` to :class:`MCPToolCall`."""
        return MCPToolCall(
            id=tool_call.id,
            name=tool_call.function.name,
            arguments=json.loads(tool_call.function.arguments or "{}"),
        )

    async def get_system_messages(self) -> list[Any]:
        """Get system messages for OpenAI."""
        return [{"role": "system", "content": self.system_prompt}]

    async def format_blocks(self, blocks: list[types.ContentBlock]) -> list[Any]:
        """Format blocks for OpenAI."""
        content = []
        for block in blocks:
            if isinstance(block, types.TextContent):
                content.append({"type": "text", "text": block.text})
            elif isinstance(block, types.ImageContent):
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{block.mimeType};base64,{block.data}"},
                    }
                )

        return [{"role": "user", "content": content}]

    def _sanitize_schema_for_openai(self, schema: dict) -> dict:
        """Convert MCP JSON Schema to OpenAI-compatible format.

        Handles unsupported features like anyOf and prefixItems.
        """
        if not isinstance(schema, dict):
            return schema

        sanitized = {}

        for key, value in schema.items():
            if key == "anyOf" and isinstance(value, list):
                # Handle anyOf patterns (usually for nullable fields)
                non_null_types = [
                    v for v in value if not (isinstance(v, dict) and v.get("type") == "null")
                ]
                if non_null_types:
                    # Use the first non-null type
                    sanitized.update(self._sanitize_schema_for_openai(non_null_types[0]))
                else:
                    sanitized["type"] = "string"  # Fallback

            elif key == "prefixItems":
                # Convert prefixItems to simple items
                sanitized["type"] = "array"
                if isinstance(value, list) and value:
                    # Use the type from the first item as the items schema
                    first_item = value[0]
                    if isinstance(first_item, dict):
                        sanitized["items"] = {"type": first_item.get("type", "string")}
                    else:
                        sanitized["items"] = {"type": "string"}

            elif key == "properties" and isinstance(value, dict):
                # Recursively sanitize property schemas
                sanitized[key] = {
                    prop_name: self._sanitize_schema_for_openai(prop_schema)
                    for prop_name, prop_schema in value.items()
                }

            elif key == "items" and isinstance(value, dict):
                # Recursively sanitize items schema
                sanitized[key] = self._sanitize_schema_for_openai(value)

            elif key in (
                "type",
                "description",
                "enum",
                "required",
                "default",
                "minimum",
                "maximum",
                "minItems",
                "maxItems",
            ):
                # These are supported by OpenAI
                sanitized[key] = value

        return sanitized or {"type": "object"}

    def get_tool_schemas(self) -> list[dict]:
        tool_schemas = super().get_tool_schemas()
        openai_tools = []
        for schema in tool_schemas:
            parameters = schema.get("parameters", {})

            if parameters:
                sanitized_params = self._sanitize_schema_for_openai(parameters)
            else:
                sanitized_params = {"type": "object", "properties": {}}

            openai_tool = {
                "type": "function",
                "function": {
                    "name": schema["name"],
                    "description": schema.get("description", ""),
                    "parameters": sanitized_params,
                },
            }
            openai_tools.append(openai_tool)
        return openai_tools

    @instrument(
        span_type="agent",
        record_args=False,
        record_result=True,
    )
    async def get_response(self, messages: list[Any]) -> AgentResponse:
        """Send chat request to OpenAI and convert the response."""

        # Convert MCP tool schemas to OpenAI format
        mcp_schemas = self.get_tool_schemas()

        protected_keys = {"model", "messages", "tools", "parallel_tool_calls"}
        extra = {k: v for k, v in (self.completion_kwargs or {}).items() if k not in protected_keys}

        response = await self.oai.chat.completions.create(
            model=self.model_name,
            messages=messages,
            tools=cast("list[ChatCompletionToolParam]", mcp_schemas),
            parallel_tool_calls=self.parallel_tool_calls,
            **extra,
        )

        choice = response.choices[0]
        msg = choice.message

        assistant_msg: dict[str, Any] = {"role": "assistant"}

        if msg.content:
            assistant_msg["content"] = msg.content

        if msg.tool_calls:
            assistant_msg["tool_calls"] = msg.tool_calls

        messages.append(assistant_msg)

        # Store the complete conversation history
        self.conversation_history = messages.copy()

        tool_calls = []
        if msg.tool_calls:
            for tc in msg.tool_calls:
                if tc.function.name is not None:  # type: ignore
                    tool_calls.append(self._oai_to_mcp(tc))
                    if not self.parallel_tool_calls:
                        break

        return AgentResponse(
            content=msg.content or "",
            tool_calls=tool_calls,
            done=choice.finish_reason in ("stop", "length"),
            raw=response,  # Include raw response for access to Choice objects
        )

    async def format_tool_results(
        self,
        tool_calls: list[MCPToolCall],
        tool_results: list[MCPToolResult],
    ) -> list[Any]:
        """Render MCP tool results as OpenAI messages.

        Note: OpenAI tool messages only support string content.
        When images are present, we return both a tool message and a user message.
        """
        rendered: list[dict[str, Any]] = []
        for call, res in zip(tool_calls, tool_results, strict=False):
            # Use structuredContent.result if available, otherwise use content
            items = res.content
            if res.structuredContent and isinstance(res.structuredContent, dict):
                items = res.structuredContent.get("result", res.content)

            # Separate text and image content
            text_parts = []
            image_parts = []

            for item in items:
                if isinstance(item, dict):
                    if item.get("type") == "text":
                        text_parts.append(item.get("text", ""))
                    elif item.get("type") == "image":
                        mime_type = item.get("mimeType", "image/png")
                        data = item.get("data", "")
                        image_parts.append(
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:{mime_type};base64,{data}"},
                            }
                        )
                elif isinstance(item, types.TextContent):
                    text_parts.append(item.text)
                elif isinstance(item, types.ImageContent):
                    image_parts.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{item.mimeType};base64,{item.data}"},
                        }
                    )

            text_content = "".join(text_parts) if text_parts else "Tool executed successfully"
            rendered.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": text_content,
                }
            )

            # If there are images, add them as a separate user message
            if image_parts:
                # Add a user message with the images
                content_with_images = [
                    {"type": "text", "text": "Tool returned the following:"},
                    *image_parts,
                ]
                rendered.append(
                    {
                        "role": "user",
                        "content": content_with_images,
                    }
                )

        return rendered
