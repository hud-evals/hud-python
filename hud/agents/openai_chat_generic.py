"""Generic OpenAI chat-completions agent.

This class provides the minimal glue required to connect any endpoint that
implements the OpenAI compatible ``chat.completions`` API with MCP tool
calling through the existing :class:`hud.agent.MCPAgent` scaffolding.

Key points:

* Stateless, no special server-side conversation state is assumed.
* Accepts an :class:`openai.AsyncOpenAI` client; caller can supply their
  own base_url / api_key (e.g. llama.cpp, together.ai, …)
* All HUD features (step_count, OTel spans, tool filtering, screenshots,
  …) come from the ``MCPAgent`` base class; we only implement the
  three abstract methods.

This version of the implementation has been adapted for robustness
across Python versions.  In particular, Python 3.10 and earlier do not
support the ``strict`` keyword argument for built-in ``zip``.  As
``strict`` isn't required for the intended semantics here (we simply
iterate over the shortest iterable), it has been removed to maintain
compatibility.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any, ClassVar, Dict, List, cast

import mcp.types as types

from hud import instrument
from hud.types import AgentResponse, MCPToolCall, MCPToolResult
from hud.utils.hud_console import HUDConsole

from .base import MCPAgent

if TYPE_CHECKING:
    from openai import AsyncOpenAI
    from openai.types.chat import ChatCompletionToolParam

logger = logging.getLogger(__name__)


class GenericOpenAIChatAgent(MCPAgent):
    """MCP-enabled agent that speaks the OpenAI *chat.completions* protocol."""

    metadata: ClassVar[Dict[str, Any]] = {}

    def __init__(
        self,
        *,
        openai_client: "AsyncOpenAI" | None,
        model_name: str = "gpt-4o-mini",
        completion_kwargs: Dict[str, Any] | None = None,
        **agent_kwargs: Any,
    ) -> None:
        # Accept base-agent settings via **agent_kwargs (e.g., mcp_client,
        # system_prompt, etc.)
        super().__init__(**agent_kwargs)
        self.oai = openai_client
        self.model_name = model_name
        self.completion_kwargs: Dict[str, Any] = completion_kwargs or {}
        self.mcp_schemas: List[Any] = []
        self.hud_console = HUDConsole(logger=logger)

    @staticmethod
    def _oai_to_mcp(tool_call: Any) -> MCPToolCall:  # type: ignore[valid-type]
        """Convert chat ``tool_call`` payloads into :class:`MCPToolCall`.

        LiteLLM providers sometimes return dict or list-of-dict arguments
        rather than a JSON string, so coerce everything to the object MCP
        expects.
        """
        function = getattr(tool_call, "function", None)
        raw_args = getattr(function, "arguments", None)
        parsed: Dict[str, Any] = {}
        if isinstance(raw_args, str):
            try:
                parsed = json.loads(raw_args) if raw_args.strip() else {}
            except Exception:  # pragma: no cover - fall back to empty args
                parsed = {}
        elif isinstance(raw_args, dict):
            parsed = raw_args
        elif isinstance(raw_args, list):
            parsed = next((item for item in raw_args if isinstance(item, dict)), {}) or {}
        name = getattr(function, "name", "") or ""
        return MCPToolCall(
            id=getattr(tool_call, "id", None),
            name=name,
            arguments=parsed,
        )

    async def get_system_messages(self) -> List[Any]:  # type: ignore[override]
        """Get system messages for OpenAI."""
        return [{"role": "system", "content": self.system_prompt}]

    async def format_blocks(self, blocks: List[types.ContentBlock]) -> List[Any]:  # type: ignore[override]
        """Format blocks for OpenAI."""
        content: List[Dict[str, Any]] = []
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

    def _sanitize_schema_for_openai(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Convert MCP JSON Schema to OpenAI-compatible format.

        Handles unsupported features like anyOf and prefixItems.
        """
        if not isinstance(schema, dict):
            return schema  # type: ignore[return-value]
        sanitized: Dict[str, Any] = {}
        for key, value in schema.items():
            if key == "anyOf" and isinstance(value, list):
                # Handle anyOf patterns (usually for nullable fields).  OpenAI
                # does not support anyOf, so pick a single variant.  Prefer
                # non-array types if available to avoid array schemas that
                # require additional processing.
                non_null_types = [
                    v for v in value if not (isinstance(v, dict) and v.get("type") == "null")
                ]
                if non_null_types:
                    # Choose the first variant that is not an array, falling
                    # back to the first non-null variant.
                    selected = None
                    for candidate in non_null_types:
                        if isinstance(candidate, dict) and candidate.get("type") != "array":
                            selected = candidate
                            break
                    if selected is None:
                        selected = non_null_types[0]
                    sanitized.update(self._sanitize_schema_for_openai(selected))
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
        # If the schema represents an array but does not specify
        # the item type, default to an array of strings.  OpenAI
        # requires ``items`` on array schemas.
        if sanitized.get("type") == "array" and "items" not in sanitized:
            sanitized["items"] = {"type": "string"}
        return sanitized or {"type": "object"}

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        tool_schemas = super().get_tool_schemas()
        openai_tools: List[Dict[str, Any]] = []
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
        messages: List[Any],
        tools: List[Dict[str, Any]] | None,
        extra: Dict[str, Any],
    ) -> Any:  # type: ignore[override]
        if self.oai is None:
            raise ValueError("openai_client is required for GenericOpenAIChatAgent")
        # default transport = OpenAI SDK
        return await self.oai.chat.completions.create(
            model=self.model_name,
            messages=messages,
            tools=tools,  # type: ignore[call-arg]
            **extra,
        )  # type: ignore[call-arg]

    @instrument(
        span_type="agent",
        record_args=False,
        record_result=True,
    )
    async def get_response(self, messages: List[Any]) -> AgentResponse:
        """Send chat request to OpenAI and convert the response."""
        # Convert MCP tool schemas to OpenAI format
        tools = cast("List[ChatCompletionToolParam]", self.get_tool_schemas())
        protected_keys = {"model", "messages", "tools"}
        extra = {k: v for k, v in (self.completion_kwargs or {}).items() if k not in protected_keys}
        try:
            response = await self._invoke_chat_completion(
                messages=messages,
                tools=tools,  # type: ignore[arg-type]
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
        assistant_msg: Dict[str, Any] = {"role": "assistant"}
        if msg.content:
            assistant_msg["content"] = msg.content
        if msg.tool_calls:
            serialized_tool_calls: List[Dict[str, Any]] = []
            for tc in msg.tool_calls:
                serialized_tc = {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                serialized_tool_calls.append(serialized_tc)
            assistant_msg["tool_calls"] = serialized_tool_calls
        messages.append(assistant_msg)
        tool_calls: List[MCPToolCall] = []
        if msg.tool_calls:
            for tc in msg.tool_calls:
                if tc.function.name is not None:
                    tool_calls.append(self._oai_to_mcp(tc))
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
        tool_calls: List[MCPToolCall],
        tool_results: List[MCPToolResult],
    ) -> List[Any]:  # type: ignore[override]
        """Render MCP tool results as OpenAI messages.

        Note: OpenAI tool messages only support string content.  When
        images are present, we return both a tool message and a user
        message.  Python 3.10 and earlier do not support passing
        ``strict=False`` to ``zip``, so we simply rely on the default
        behaviour of zipping until the shortest iterable is exhausted.
        """
        rendered: List[Dict[str, Any]] = []
        # Separate text and image content
        image_parts: List[Dict[str, Any]] = []
        for call, res in zip(tool_calls, tool_results):
            # Use structuredContent.result if available, otherwise use content
            text_parts: List[str] = []
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
            # Add a user message with the last image and a prefix
            content_with_images: List[Dict[str, Any]] = [
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