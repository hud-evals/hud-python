"""OpenAI Chat Completions Agent.

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
from openai import AsyncOpenAI
from pydantic import ConfigDict, Field

from hud import instrument
from hud.types import AgentResponse, BaseAgentConfig, MCPToolCall, MCPToolResult
from hud.utils.hud_console import HUDConsole
from hud.utils.types import with_signature

from .base import BaseCreateParams, MCPAgent

if TYPE_CHECKING:
    from openai.types.chat import ChatCompletionToolParam


logger = logging.getLogger(__name__)


class OpenAIChatConfig(BaseAgentConfig):
    """Configuration for `OpenAIChatAgent`."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    model_name: str = "OpenAI"
    checkpoint_name: str = "gpt-5-mini"
    openai_client: AsyncOpenAI | None = None
    api_key: str | None = None
    base_url: str | None = None
    completion_kwargs: dict[str, Any] = Field(default_factory=dict)


class OpenAIChatCreateParams(BaseCreateParams, OpenAIChatConfig):
    pass


class OpenAIChatAgent(MCPAgent):
    """MCP-enabled agent that speaks the OpenAI *chat.completions* protocol."""

    metadata: ClassVar[dict[str, Any] | None] = None
    config_cls: ClassVar[type[BaseAgentConfig]] = OpenAIChatConfig

    @with_signature(OpenAIChatCreateParams)
    @classmethod
    def create(cls, **kwargs: Any) -> OpenAIChatAgent:  # pyright: ignore[reportIncompatibleMethodOverride]
        return MCPAgent.create.__func__(cls, **kwargs)  # type: ignore[return-value]

    def __init__(self, params: OpenAIChatCreateParams | None = None, **kwargs: Any) -> None:
        super().__init__(params, **kwargs)
        self.config: OpenAIChatConfig

        if self.config.openai_client is not None:
            self.oai = self.config.openai_client
        elif self.config.api_key is not None or self.config.base_url is not None:
            self.oai = AsyncOpenAI(api_key=self.config.api_key, base_url=self.config.base_url)
        else:
            raise ValueError(
                "Either openai_client or api_key must be provided. "
                "Set OPENAI_API_KEY environment variable or pass api_key explicitly."
            )

        self.completion_kwargs = dict(self.config.completion_kwargs)
        self.mcp_schemas: list[ChatCompletionToolParam] = []
        self.hud_console = HUDConsole(logger=logger)

    @staticmethod
    def _oai_to_mcp(tool_call: Any) -> MCPToolCall:  # type: ignore[valid-type]
        """Convert an OpenAI ``tool_call`` to :class:`MCPToolCall`."""
        args = json.loads(tool_call.function.arguments or "{}")
        if isinstance(args, list):
            args = args[0]
        if not isinstance(args, dict):
            args = {}
        return MCPToolCall(
            id=tool_call.id,
            name=tool_call.function.name,
            arguments=args,
        )

    async def get_system_messages(self) -> list[Any]:
        """Get system messages for OpenAI."""
        if self.system_prompt is not None:
            return [{"role": "system", "content": self.system_prompt}]
        else:
            return []

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

    async def _invoke_chat_completion(
        self,
        *,
        messages: list[Any],
        tools: list[dict] | None,
        extra: dict[str, Any],
    ) -> Any:
        if self.oai is None:
            raise ValueError("openai_client is required for OpenAIChatAgent")
        # default transport = OpenAI SDK
        return await self.oai.chat.completions.create(
            model=self.config.checkpoint_name,
            messages=messages,
            tools=tools,  # type: ignore ready ChatCompletionToolParam-shaped
            **extra,
        )  # type: ignore

    @instrument(
        span_type="agent",
        record_args=False,
        record_result=True,
    )
    async def get_response(self, messages: list[Any]) -> AgentResponse:
        """Send chat request to OpenAI and convert the response."""

        # Convert MCP tool schemas to OpenAI format
        tools = cast("list[ChatCompletionToolParam]", self.get_tool_schemas())

        protected_keys = {"model", "messages", "tools"}
        extra = {k: v for k, v in (self.completion_kwargs or {}).items() if k not in protected_keys}

        try:
            response = await self._invoke_chat_completion(
                messages=messages,
                tools=tools,  # type: ignore
                extra=extra,
            )
        except Exception as e:
            error_content = f"Error getting response {e}"
            if "Invalid JSON" in str(e):
                error_content = "Invalid JSON, response was truncated"
            self.hud_console.warning_log(error_content)

            return AgentResponse(
                content=error_content,
                tool_calls=[],
                done=True,
                isError=True,
                raw=None,
            )

        choice = response.choices[0]
        msg = choice.message
        assistant_msg: dict[str, Any] = {"role": "assistant"}

        if msg.content:
            assistant_msg["content"] = msg.content

        if msg.tool_calls:
            serialized_tool_calls = []
            for tc in msg.tool_calls:
                serialized_tc = {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                serialized_tool_calls.append(serialized_tc)
            assistant_msg["tool_calls"] = serialized_tool_calls

        messages.append(assistant_msg)

        tool_calls = []
        if msg.tool_calls:
            for tc in msg.tool_calls:
                if tc.function.name is not None:  # type: ignore
                    # _oai_to_mcp returns a single MCPToolCall; append it
                    tool_calls.append(self._oai_to_mcp(tc))  # noqa: PERF401

        # Only stop on length (token limit), never on "stop"
        done = choice.finish_reason == "length"
        if done:
            self.hud_console.info_log(f"Done decision: finish_reason={choice.finish_reason}")

        return AgentResponse(
            content=msg.content or "",
            tool_calls=tool_calls,
            done=done,
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

        # Separate text and image content
        image_parts = []
        for call, res in zip(tool_calls, tool_results, strict=False):
            # Use structuredContent.result if available, otherwise use content
            text_parts = []
            items = res.content
            if not res.content and res.structuredContent:
                items = [res.structuredContent.get("result", res.content)]

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
                image_parts[-1],
            ]
            rendered.append(
                {
                    "role": "user",
                    "content": content_with_images,
                }
            )

        return rendered
