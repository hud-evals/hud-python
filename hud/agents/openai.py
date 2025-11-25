"""OpenAI MCP Agent implementation."""

from __future__ import annotations

import copy
import json
import logging
from inspect import cleandoc
from typing import TYPE_CHECKING, Any, Literal

import mcp.types as types
from openai import AsyncOpenAI, Omit, OpenAI
from openai.types.responses import (
    ApplyPatchToolParam,
    FunctionShellToolParam,
    FunctionToolParam,
    ResponseFunctionCallOutputItemListParam,
    ResponseInputFileContentParam,
    ResponseInputImageContentParam,
    ResponseInputImageParam,
    ResponseInputMessageContentListParam,
    ResponseInputParam,
    ResponseInputTextContentParam,
    ResponseInputTextParam,
    ResponseOutputText,
    ToolParam,
)
from openai.types.responses.response_computer_tool_call import PendingSafetyCheck
from openai.types.responses.response_input_param import FunctionCallOutput, Message

import hud
from hud.settings import settings
from hud.types import AgentResponse, MCPToolCall, MCPToolResult, Trace
from hud.utils.strict_schema import ensure_strict_json_schema

from .base import MCPAgent

if TYPE_CHECKING:
    from openai.types.responses.response_create_params import ToolChoice
    from openai.types.shared_params.reasoning import Reasoning

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
        reasoning: Reasoning | None = None,
        tool_choice: ToolChoice | None = None,
        truncation: Literal["auto", "disabled"] | None = None,
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
        self.tool_choice: ToolChoice | None = tool_choice
        self.parallel_tool_calls = parallel_tool_calls
        self.truncation: Literal["auto", "disabled"] | None = truncation
        self._openai_tools: list[ToolParam] = []
        self._tool_name_map: dict[str, str] = {}

        self.last_response_id: str | None = None
        self.pending_call_id: str | None = None
        self.pending_safety_checks: list[PendingSafetyCheck] = []
        self._message_cursor = 0

        self.model_name = "OpenAI"
        self.checkpoint_name = self.model

    async def initialize(self, task: Any | None = None) -> None:
        """Initialize agent and build tool metadata."""
        await super().initialize(task)
        self._convert_tools_for_openai()

    def _convert_tools_for_openai(self) -> None:
        """Convert MCP tools into OpenAI Responses tool definitions."""
        available_tools = self.get_available_tools()

        def to_api_tool(
            tool: types.Tool,
        ) -> FunctionShellToolParam | ApplyPatchToolParam | FunctionToolParam | None:
            # Special case: shell tool -> OpenAI native shell
            if tool.name == "shell":
                return FunctionShellToolParam(type="shell")

            # Special case: apply_patch tool -> OpenAI native apply_patch
            if tool.name == "apply_patch":
                return ApplyPatchToolParam(type="apply_patch")

            # Regular function tool
            if tool.description is None or tool.inputSchema is None:
                raise ValueError(
                    cleandoc(f"""MCP tool {tool.name} requires both a description and inputSchema.
                    Add these by:
                    1. Adding a docstring to your @mcp.tool decorated function for the description
                    2. Using pydantic Field() annotations on function parameters for the schema
                    """)
                )

            # schema must be strict

            try:
                strict_schema = ensure_strict_json_schema(copy.deepcopy(tool.inputSchema))
            except Exception as e:
                self.console.warning_log(f"Failed to convert tool '{tool.name}' schema to strict: {e}")
                logger.error(json.dumps(tool.inputSchema, indent=2))
                raise e

            return FunctionToolParam(
                type="function",
                name=tool.name,
                description=tool.description,
                parameters=strict_schema,
                strict=True,
            )

        self._openai_tools = []
        self._tool_name_map = {}

        for tool in available_tools:
            openai_tool = to_api_tool(tool)
            if openai_tool is None:
                continue

            if "name" in openai_tool:
                self._tool_name_map[openai_tool["name"]] = tool.name
            self._openai_tools.append(openai_tool)

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
                content.append(ResponseInputTextParam(type="input_text", text=block.text))
            elif isinstance(block, types.ImageContent):
                mime_type = getattr(block, "mimeType", "image/png")
                content.append(
                    ResponseInputImageParam(
                        type="input_image",
                        image_url=f"data:{mime_type};base64,{block.data}",
                        detail="auto",
                    )
                )
        if not content:
            content.append(ResponseInputTextParam(type="input_text", text=""))
        return [Message(role="user", content=content)]

    @hud.instrument(
        span_type="agent",
        record_args=False,
        record_result=True,
    )
    async def get_response(self, messages: ResponseInputParam) -> AgentResponse:
        """Send the latest input items to OpenAI's Responses API."""
        new_items: ResponseInputParam = messages[self._message_cursor :]
        if not new_items:
            if self.last_response_id is None:
                new_items = [
                    Message(
                        role="user", content=[ResponseInputTextParam(type="input_text", text="")]
                    )
                ]
            else:
                self.console.debug("No new messages to send to OpenAI.")
                return AgentResponse(content="", tool_calls=[], done=True)

        response = await self.openai_client.responses.create(
            model=self.model,
            input=new_items,
            instructions=self.system_prompt,
            max_output_tokens=self.max_output_tokens,
            temperature=self.temperature,
            tool_choice=self.tool_choice if self.tool_choice is not None else Omit(),
            parallel_tool_calls=self.parallel_tool_calls,
            reasoning=self.reasoning,
            tools=self._openai_tools if self._openai_tools else Omit(),
            previous_response_id=(
                self.last_response_id if self.last_response_id is not None else Omit()
            ),
            truncation=self.truncation,
        )

        self.last_response_id = response.id
        self._message_cursor = len(messages)
        self.pending_call_id = None

        agent_response = AgentResponse(content="", tool_calls=[], done=True)
        text_chunks: list[str] = []
        reasoning_chunks: list[str] = []

        for item in response.output:
            if item.type == "message":
                text = "".join(
                    content.text
                    for content in item.content
                    if isinstance(content, ResponseOutputText)
                )
                if text:
                    text_chunks.append(text)
            elif item.type == "function_call":
                target_name = self._tool_name_map.get(item.name, item.name)
                try:
                    arguments = json.loads(item.arguments)
                except json.JSONDecodeError:
                    self.console.warning_log(
                        f"Failed to parse arguments for tool '{item.name}', passing raw string."
                    )
                    arguments = {"raw_arguments": item.arguments}
                agent_response.tool_calls.append(
                    MCPToolCall(name=target_name, arguments=arguments, id=item.call_id)
                )
            elif item.type == "shell_call":
                agent_response.tool_calls.append(
                    MCPToolCall(name="shell", arguments=item.action.to_dict(), id=item.call_id)
                )
            elif item.type == "apply_patch_call":
                agent_response.tool_calls.append(
                    MCPToolCall(
                        name="apply_patch", arguments=item.operation.to_dict(), id=item.call_id
                    )
                )
            elif item.type == "computer_call":
                self.pending_safety_checks = item.pending_safety_checks
                agent_response.tool_calls.append(
                    MCPToolCall(name="computer", arguments=item.action.to_dict(), id=item.call_id)
                )
            elif item.type == "reasoning":
                reasoning_chunks.append(
                    "".join(f"Thinking: {summary.text}\n" for summary in item.summary)
                )

        if agent_response.tool_calls:
            agent_response.done = False

        agent_response.content = "".join(reasoning_chunks) + "".join(text_chunks)
        return agent_response

    async def format_tool_results(
        self, tool_calls: list[MCPToolCall], tool_results: list[MCPToolResult]
    ) -> ResponseInputParam:
        """Convert MCP tool outputs into Responses input items."""
        formatted: ResponseInputParam = []
        for call, result in zip(tool_calls, tool_results, strict=False):
            if not call.id:
                self.console.warning_log(f"Tool '{call.name}' missing call_id; skipping output.")
                continue

            output_items: ResponseFunctionCallOutputItemListParam = []
            if result.isError:
                output_items.append(
                    ResponseInputTextParam(type="input_text", text="[tool_error] true")
                )

            if result.structuredContent is not None:
                output_items.append(
                    ResponseInputTextParam(
                        type="input_text", text=json.dumps(result.structuredContent, default=str)
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
                            ResponseInputFileContentParam(
                                type="input_file", file_url=str(block.uri)
                            )
                        )
                    case types.EmbeddedResource():
                        match block.resource:
                            case types.TextResourceContents():
                                output_items.append(
                                    ResponseInputTextContentParam(
                                        type="input_text", text=block.resource.text
                                    )
                                )
                            case types.BlobResourceContents():
                                output_items.append(
                                    ResponseInputFileContentParam(
                                        type="input_file", file_data=block.resource.blob
                                    )
                                )
                            case _:
                                self.console.warning_log(
                                    f"Unknown resource type: {type(block.resource)}"
                                )
                    case _:
                        self.console.warning_log(f"Unknown content block type: {type(block)}")

            if not output_items:
                output_items.append(ResponseInputTextParam(type="input_text", text=""))

            formatted.append(
                FunctionCallOutput(
                    type="function_call_output", call_id=call.id, output=output_items
                ),
            )
        return formatted
