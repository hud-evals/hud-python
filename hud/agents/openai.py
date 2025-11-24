"""OpenAI MCP Agent implementation."""

from __future__ import annotations

import copy
import json
import logging
from typing import Any, Literal, cast

import mcp.types as types
from openai import AsyncOpenAI, OpenAI
from openai.types.responses import (
    ResponseFunctionToolCall,
    ResponseInputImageParam,
    ResponseInputMessageContentListParam,
    ResponseInputParam,
    ResponseInputTextParam,
    ResponseOutputMessage,
    ResponseOutputText,
    ResponseReasoningItem,
    ToolParam,
)
from openai.types.responses.response_input_param import (
    FunctionCallOutput,  # noqa: TC002
    Message,  # noqa: TC002
)

import hud
from hud.settings import settings
from hud.types import AgentResponse, MCPToolCall, MCPToolResult, Trace
from hud.utils.strict_schema import ensure_strict_json_schema

from .base import MCPAgent

logger = logging.getLogger(__name__)


class OpenAIAgent(MCPAgent):
    """Generic OpenAI agent that can execute MCP tools through the Responses API."""

    metadata: dict[str, Any] | None = None

    def __init__(
        self,
        model_client: AsyncOpenAI | None = None,
        model: str = "gpt-5.1",
        max_output_tokens: int | None = None,
        temperature: float | None = None,
        reasoning: dict[str, Any] | Literal["auto"] | None = None,
        tool_choice: dict[str, Any] | Literal["auto"] | None = None,
        parallel_tool_calls: bool | None = None,
        validate_api_key: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        if model_client is None:
            api_key = settings.openai_api_key
            if not api_key:
                raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY.")
            model_client = AsyncOpenAI(api_key=api_key)

        if validate_api_key:
            try:
                OpenAI(api_key=model_client.api_key).models.list()
            except Exception as exc:  # pragma: no cover - network validation
                raise ValueError(f"OpenAI API key is invalid: {exc}") from exc

        self.openai_client = model_client
        self.model = model
        self.max_output_tokens = max_output_tokens
        self.temperature = temperature
        self.reasoning = reasoning
        self.tool_choice = tool_choice
        self.parallel_tool_calls = parallel_tool_calls

        self._openai_tools: list[ToolParam] = []
        self._tool_name_map: dict[str, str] = {}

        self.last_response_id: str | None = None
        self.pending_call_id: str | None = None
        self.pending_safety_checks: list[Any] = []
        self._message_cursor = 0

        self.model_name = "OpenAI"
        self.checkpoint_name = self.model

    async def initialize(self, task: Any | None = None) -> None:
        """Initialize agent and build tool metadata."""
        await super().initialize(task)
        self._build_openai_tools()

    def _build_openai_tools(self) -> None:
        """Convert MCP tools into OpenAI Responses tool definitions."""
        self._openai_tools = []
        self._tool_name_map = {}

        for tool in self.get_available_tools():
            if tool.description is None or tool.inputSchema is None:
                self.console.warning_log(
                    f"Skipping tool '{tool.name}' - description and input schema "
                    "are required for OpenAI tools."
                )
                continue
            self._tool_name_map[tool.name] = tool.name
            schema_copy = copy.deepcopy(tool.inputSchema)
            strict_schema = schema_copy
            strict_enforced = True
            try:
                strict_schema = ensure_strict_json_schema(schema_copy)
            except Exception as exc:  # pragma: no cover - defensive
                strict_enforced = False
                self.console.warning_log(
                    f"Failed to convert schema for tool '{tool.name}' to strict mode: {exc}"
                )

            function_tool = cast(
                "ToolParam",
                {
                    "type": "function",
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": strict_schema,
                    "strict": strict_enforced,
                },
            )
            self._openai_tools.append(function_tool)

    async def _run_context(
        self, context: list[types.ContentBlock], *, max_steps: int = 10
    ) -> Trace:
        """Reset internal state before delegating to the base loop."""
        self._reset_response_state()
        return await super()._run_context(context, max_steps=max_steps)

    def _reset_response_state(self) -> None:
        self.last_response_id = None
        self.pending_call_id = None
        self.pending_safety_checks = []
        self._message_cursor = 0

    async def get_system_messages(self) -> list[types.ContentBlock]:
        """System messages are provided via the `instructions` field."""
        return []

    async def format_blocks(self, blocks: list[types.ContentBlock]) -> ResponseInputParam:
        """Convert MCP content blocks into OpenAI user messages."""
        content: ResponseInputMessageContentListParam = []
        for block in blocks:
            if isinstance(block, types.TextContent):
                content.append(
                    cast(
                        "ResponseInputTextParam",
                        {"type": "input_text", "text": block.text},
                    )
                )
            elif isinstance(block, types.ImageContent):
                mime_type = getattr(block, "mimeType", "image/png")
                content.append(
                    cast(
                        "ResponseInputImageParam",
                        {
                            "type": "input_image",
                            "image_url": f"data:{mime_type};base64,{block.data}",
                        },
                    )
                )
        if not content:
            content.append(cast("ResponseInputTextParam", {"type": "input_text", "text": ""}))
        return [cast("Message", {"role": "user", "content": content})]

    @hud.instrument(
        span_type="agent",
        record_args=False,
        record_result=True,
    )
    async def get_response(self, messages: ResponseInputParam) -> AgentResponse:
        """Send the latest input items to OpenAI's Responses API."""
        new_items = cast("ResponseInputParam", messages[self._message_cursor :])
        if not new_items:
            if self.last_response_id is None:
                new_items = cast(
                    "ResponseInputParam",
                    [
                        cast(
                            "Message",
                            {
                                "role": "user",
                                "content": [
                                    cast(
                                        "ResponseInputTextParam",
                                        {"type": "input_text", "text": ""},
                                    )
                                ],
                            },
                        )
                    ],
                )
            else:
                self.console.debug("No new messages to send to OpenAI.")
                return AgentResponse(content="", tool_calls=[], done=True)

        payload = self._build_request_payload(new_items)
        response = await self.openai_client.responses.create(**payload)

        self.last_response_id = response.id
        self._message_cursor = len(messages)
        self.pending_call_id = None

        agent_response = AgentResponse(content="", tool_calls=[], done=True)
        text_chunks: list[str] = []
        reasoning_chunks: list[str] = []

        for item in response.output:
            if isinstance(item, ResponseOutputMessage) and item.type == "message":
                text = "".join(
                    content.text
                    for content in item.content
                    if isinstance(content, ResponseOutputText)
                )
                if text:
                    text_chunks.append(text)
            elif isinstance(item, ResponseFunctionToolCall):
                tool_call = self._convert_function_tool_call(item)
                if tool_call:
                    agent_response.tool_calls.append(tool_call)
            elif isinstance(item, ResponseReasoningItem) and item.summary:
                reasoning_chunks.append(
                    "".join(f"Thinking: {summary.text}\n" for summary in item.summary)
                )

        if agent_response.tool_calls:
            agent_response.done = False

        agent_response.content = "".join(reasoning_chunks) + "".join(text_chunks)
        return agent_response

    def _build_request_payload(self, new_items: ResponseInputParam) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "input": new_items,
            "instructions": self.system_prompt,
            "max_output_tokens": self.max_output_tokens,
            "temperature": self.temperature,
            "tool_choice": self.tool_choice,
            "parallel_tool_calls": self.parallel_tool_calls,
        }
        if self.reasoning is not None:
            payload["reasoning"] = self.reasoning
        if self._openai_tools:
            payload["tools"] = self._openai_tools
        if self.last_response_id is not None:
            payload["previous_response_id"] = self.last_response_id
        return {k: v for k, v in payload.items() if v is not None}

    def _convert_function_tool_call(
        self, tool_call: ResponseFunctionToolCall
    ) -> MCPToolCall | None:
        target_name = self._tool_name_map.get(tool_call.name, tool_call.name)
        try:
            arguments = json.loads(tool_call.arguments) if tool_call.arguments else {}
        except json.JSONDecodeError:
            self.console.warning_log(
                f"Failed to parse arguments for tool '{tool_call.name}', passing raw string."
            )
            arguments = {"raw_arguments": tool_call.arguments}
        return MCPToolCall(name=target_name, arguments=arguments, id=tool_call.call_id)

    async def format_tool_results(
        self, tool_calls: list[MCPToolCall], tool_results: list[MCPToolResult]
    ) -> ResponseInputParam:
        """Convert MCP tool outputs into Responses input items."""
        formatted: ResponseInputParam = []
        for call, result in zip(tool_calls, tool_results, strict=False):
            if not call.id:
                self.console.warning_log(f"Tool '{call.name}' missing call_id; skipping output.")
                continue

            output_items: list[dict[str, Any]] = []
            if result.isError:
                output_items.append({"type": "input_text", "text": "[tool_error] true"})

            if result.structuredContent is not None:
                output_items.append(
                    {
                        "type": "input_text",
                        "text": json.dumps(result.structuredContent, default=str),
                    }
                )

            if result.content:
                for block in result.content:
                    if isinstance(block, types.TextContent):
                        output_items.append({"type": "input_text", "text": block.text})
                    elif isinstance(block, types.ImageContent):
                        mime_type = getattr(block, "mimeType", "image/png")
                        output_items.append(
                            {
                                "type": "input_image",
                                "image_url": f"data:{mime_type};base64,{block.data}",
                            }
                        )
                    else:
                        output_items.append(
                            {
                                "type": "input_text",
                                "text": getattr(block, "text", str(block)),
                            }
                        )

            if not output_items:
                output_items.append({"type": "input_text", "text": ""})

            formatted.append(
                cast(
                    "FunctionCallOutput",
                    {
                        "type": "function_call_output",
                        "call_id": call.id,
                        "output": output_items,
                    },
                )
            )
        return formatted
