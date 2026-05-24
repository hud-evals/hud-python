"""OpenAI MCP Agent implementation."""

from __future__ import annotations

import json
import logging
from functools import cached_property
from typing import Any, Literal, cast

import mcp.types as types
from openai import AsyncOpenAI, Omit, OpenAI
from openai.types.responses import (
    ResponseIncludable,
    ResponseInputImageParam,
    ResponseInputMessageContentListParam,
    ResponseInputParam,
    ResponseInputTextParam,
    ResponseOutputText,
    ToolParam,
)
from openai.types.responses.easy_input_message_param import EasyInputMessageParam
from openai.types.responses.response_create_params import ToolChoice  # noqa: TC002
from openai.types.responses.response_input_param import (
    Message,
    ResponseInputItemParam,
)
from openai.types.shared_params.reasoning import Reasoning  # noqa: TC002

from hud.agents import gateway
from hud.agents.base import MCPAgent
from hud.agents.types import OpenAIConfig
from hud.settings import settings
from hud.types import AgentResponse, MCPToolCall
from hud.utils.types import with_signature

from .tools import OpenAIAgentTools

logger = logging.getLogger(__name__)


class OpenAIAgent(MCPAgent[ResponseInputItemParam]):
    """Generic OpenAI agent that can execute MCP tools through the Responses API."""

    @with_signature(OpenAIConfig)
    @classmethod
    def create(cls, **kwargs: object) -> OpenAIAgent:  # pyright: ignore[reportIncompatibleMethodOverride]
        return cls(OpenAIConfig.model_validate(kwargs))

    def __init__(self, config: OpenAIConfig | None = None) -> None:
        config = config or OpenAIConfig()
        super().__init__(config)
        self.config: OpenAIConfig

        model_client = self.config.model_client
        if model_client is None:
            if settings.api_key:
                model_client = gateway.build_gateway_client("openai")
            elif settings.openai_api_key:
                model_client = AsyncOpenAI(api_key=settings.openai_api_key)
                if self.config.validate_api_key:
                    try:
                        OpenAI(api_key=settings.openai_api_key).models.list()
                    except Exception as exc:  # pragma: no cover - network validation
                        raise ValueError(f"OpenAI API key is invalid: {exc}") from exc
            else:
                raise ValueError(
                    "No API key found for OpenAI.\n"
                    "  • Set HUD_API_KEY to use HUD Gateway"
                    " (add your OpenAI key at"
                    " hud.ai/project/secrets for BYOK)\n"
                    "  • Or set OPENAI_API_KEY for direct"
                    " access"
                )

        self.openai_client: AsyncOpenAI = cast("AsyncOpenAI", model_client)
        self._model = self.config.model
        self.max_output_tokens = self.config.max_output_tokens
        self.temperature = self.config.temperature
        self.reasoning: Reasoning | None = self.config.reasoning
        self.tool_choice: ToolChoice | None = self.config.tool_choice
        self.parallel_tool_calls = self.config.parallel_tool_calls
        self.text = self.config.text
        self.truncation: Literal["auto", "disabled"] | None = self.config.truncation

        self.last_response_id: str | None = None
        self._message_cursor = 0

    @cached_property
    def tools(self) -> OpenAIAgentTools:
        return OpenAIAgentTools()

    async def format_messages(
        self, messages: list[types.PromptMessage]
    ) -> list[ResponseInputItemParam]:
        """Convert MCP prompt messages into OpenAI Responses input items."""
        formatted_messages: list[ResponseInputItemParam] = []
        for message in messages:
            match message.content:
                case types.TextContent() as block:
                    content: ResponseInputMessageContentListParam = [
                        ResponseInputTextParam(type="input_text", text=block.text)
                    ]
                case types.ImageContent() as block:
                    mime_type = getattr(block, "mimeType", "image/png")
                    content = [
                        ResponseInputImageParam(
                            type="input_image",
                            image_url=f"data:{mime_type};base64,{block.data}",
                            detail="auto",
                        )
                    ]
                case _:
                    content = [ResponseInputTextParam(type="input_text", text="")]

            formatted_messages.append(EasyInputMessageParam(role=message.role, content=content))
        return formatted_messages

    async def get_response(self, messages: list[ResponseInputItemParam]) -> AgentResponse:
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
                logger.debug("No new messages to send to OpenAI.")
                return AgentResponse(content="", tool_calls=[], done=True)

        include_param: list[ResponseIncludable] | Omit = Omit()
        if self.enable_citations:
            include_param = ["web_search_call.action.sources"]

        effective_tools: list[ToolParam] = list(self.tools.params)
        if self.tools.tool_search_threshold is not None:
            fn_count = sum(1 for t in effective_tools if t.get("type") == "function")
            if fn_count > self.tools.tool_search_threshold:
                logger.debug(
                    "tool_search: %d function tools > threshold %d, applying defer_loading",
                    fn_count,
                    self.tools.tool_search_threshold,
                )
                effective_tools = cast(
                    "list[ToolParam]",
                    [
                        {**t, "defer_loading": True} if t.get("type") == "function" else t
                        for t in effective_tools
                    ],
                )

        response = await self.openai_client.responses.create(
            model=self._model,
            input=new_items,
            instructions=self.system_prompt,
            max_output_tokens=self.max_output_tokens,
            temperature=self.temperature,
            text=self.text if self.text is not None else Omit(),
            tool_choice=self.tool_choice if self.tool_choice is not None else Omit(),
            parallel_tool_calls=self.parallel_tool_calls,
            reasoning=self.reasoning if self.reasoning is not None else Omit(),
            tools=effective_tools if effective_tools else Omit(),
            previous_response_id=(
                self.last_response_id if self.last_response_id is not None else Omit()
            ),
            truncation=self.truncation if self.truncation is not None else Omit(),
            include=include_param,
        )

        self.last_response_id = response.id
        self._message_cursor = len(messages)

        text_chunks: list[str] = []
        reasoning_chunks: list[str] = []
        citations: list[dict[str, object]] = []
        tool_calls: list[MCPToolCall] = []

        for item in response.output:
            match item.type:
                case "message":
                    for content_block in item.content:
                        if not isinstance(content_block, ResponseOutputText):
                            continue
                        if content_block.text:
                            text_chunks.append(content_block.text)
                        for ann in content_block.annotations or []:
                            match ann.type:
                                case "url_citation":
                                    citation = ann
                                    citations.append(
                                        {
                                            "type": "url_citation",
                                            "text": citation.title,
                                            "source": citation.url,
                                            "title": citation.title,
                                            "start_index": citation.start_index,
                                            "end_index": citation.end_index,
                                        }
                                    )
                                case "file_citation":
                                    citation = ann
                                    citations.append(
                                        {
                                            "type": "file_citation",
                                            "text": citation.filename,
                                            "source": citation.file_id,
                                            "title": citation.filename,
                                        }
                                    )
                                case _:
                                    continue
                case "reasoning":
                    reasoning_chunks.append("".join(summary.text for summary in item.summary))
                case "function_call":
                    tool_name = item.name or ""
                    tool_calls.append(
                        MCPToolCall(
                            name=self.tools.name_map.get(tool_name, tool_name),
                            arguments=json.loads(item.arguments),
                            id=item.call_id,
                        )
                    )
                case "computer_call":
                    if item.actions:
                        arguments = {"actions": [action.to_dict() for action in item.actions]}
                    elif item.action is not None:
                        arguments = item.action.to_dict()
                    else:
                        raise ValueError("OpenAI computer_call missing action")
                    call: dict[str, Any] = {
                        "name": self.tools.name_map.get("computer", "computer"),
                        "arguments": arguments,
                        "id": item.call_id,
                    }
                    if item.pending_safety_checks:
                        call["pending_safety_checks"] = [
                            check.model_dump() if hasattr(check, "model_dump") else check
                            for check in item.pending_safety_checks
                        ]
                    tool_calls.append(MCPToolCall.model_validate(call))
                case "shell_call":
                    tool_calls.append(
                        MCPToolCall(
                            name="shell",
                            arguments=item.action.to_dict(),
                            id=item.call_id,
                        )
                    )
                case _:
                    continue

        return AgentResponse(
            content="".join(text_chunks),
            reasoning="\n".join(reasoning_chunks) if reasoning_chunks else None,
            citations=citations,
            tool_calls=tool_calls,
            done=not tool_calls,
        )
