"""OpenAIAgent — ``ToolAgent`` over OpenAI's Responses API."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Literal, cast

from openai import AsyncOpenAI, Omit
from openai.types.responses import (
    ResponseIncludable,
    ResponseInputParam,
    ResponseInputTextParam,
    ResponseOutputText,
    ToolParam,
)
from openai.types.responses.easy_input_message_param import EasyInputMessageParam
from openai.types.responses.response_create_params import ToolChoice  # noqa: TC002
from openai.types.responses.response_input_param import (
    ComputerCallOutput,
    Message,
    ResponseInputItemParam,
)
from openai.types.shared_params.reasoning import Reasoning  # noqa: TC002

from hud.agents.tool_agent import RunState, ToolAgent
from hud.agents.types import OpenAIConfig
from hud.settings import settings
from hud.types import AgentResponse, MCPToolCall, MCPToolResult
from hud.utils import gateway

from .tools import OpenAIComputerTool, OpenAIMCPProxyTool, OpenAIShellTool
from .tools.base import format_openai_result
from .tools.coding import _shell_output
from .tools.computer import last_image_data

logger = logging.getLogger(__name__)


@dataclass
class OpenAIRunState(RunState[ResponseInputItemParam]):
    last_response_id: str | None = None
    message_cursor: int = 0


class OpenAIAgent(ToolAgent[ResponseInputItemParam, OpenAIConfig]):
    """OpenAI agent using the Responses API. Drives SSH, RFB, and MCP capabilities."""

    tool_catalog = (
        OpenAIShellTool,
        OpenAIComputerTool,
        OpenAIMCPProxyTool,
    )

    def __init__(self, config: OpenAIConfig | None = None) -> None:
        config = config or OpenAIConfig()
        self.config = config

        model_client = config.model_client
        if model_client is None:
            if settings.api_key:
                model_client = gateway.build_gateway_client("openai")
            elif settings.openai_api_key:
                model_client = AsyncOpenAI(api_key=settings.openai_api_key)
            else:
                raise ValueError(
                    "No API key for OpenAI. Set HUD_API_KEY or OPENAI_API_KEY.",
                )

        self.openai_client: AsyncOpenAI = cast("AsyncOpenAI", model_client)
        self._model = config.model
        self.max_output_tokens = config.max_output_tokens
        self.temperature = config.temperature
        self.reasoning: Reasoning | None = config.reasoning
        self.tool_choice: ToolChoice | None = config.tool_choice
        self.parallel_tool_calls = config.parallel_tool_calls
        self.text = config.text
        self.truncation: Literal["auto", "disabled"] | None = config.truncation

    # ─── ToolAgent hooks ──────────────────────────────────────────────

    async def _initialize_state(self, *, prompt: str | list[Any] | None) -> OpenAIRunState:
        return OpenAIRunState(messages=self._initial_messages(prompt))

    def _format_message(self, role: str, text: str) -> ResponseInputItemParam:
        return cast(
            "ResponseInputItemParam",
            EasyInputMessageParam(
                role="assistant" if role == "assistant" else "user",
                content=[ResponseInputTextParam(type="input_text", text=text)],
            ),
        )

    def _format_result(
        self,
        call: MCPToolCall,
        result: MCPToolResult,
        state: RunState[ResponseInputItemParam],
    ) -> ResponseInputItemParam | list[ResponseInputItemParam] | None:
        tool = state.tools.get(call.name)

        if isinstance(tool, OpenAIComputerTool):
            screenshot = last_image_data(result)
            if not screenshot:
                logger.warning("Computer tool result missing screenshot for call %s", call.name)
                return None
            output = ComputerCallOutput(
                type="computer_call_output",
                call_id=call.id,
                output=cast(
                    "Any",
                    {
                        "type": "computer_screenshot",
                        "image_url": f"data:image/png;base64,{screenshot}",
                        "detail": "original",
                    },
                ),
            )
            checks = (call.model_extra or {}).get("pending_safety_checks")
            if isinstance(checks, list):
                acknowledged = []
                for raw_check in cast("list[Any]", checks):
                    if hasattr(raw_check, "model_dump"):
                        acknowledged.append(raw_check.model_dump())
                    elif isinstance(raw_check, dict):
                        acknowledged.append(raw_check)
                if acknowledged:
                    output["acknowledged_safety_checks"] = acknowledged
            return cast("ResponseInputItemParam", output)

        if isinstance(tool, OpenAIShellTool):
            structured = (
                result.structuredContent if isinstance(result.structuredContent, dict) else {}
            )
            output_list = structured.get("output")
            if not isinstance(output_list, list):
                from hud.agents.tools.base import result_text

                text = result_text(result)
                output_list = [_shell_output("", text, 1 if result.isError else 0)]
            response: dict[str, Any] = {
                "type": "shell_call_output",
                "call_id": call.id,
                "status": "completed",
                "output": output_list,
            }
            max_output_length = structured.get("max_output_length")
            if isinstance(max_output_length, int):
                response["max_output_length"] = max_output_length
            return cast("ResponseInputItemParam", response)

        return format_openai_result(call, result)

    async def get_response(
        self,
        state: RunState[ResponseInputItemParam],
        *,
        system_prompt: str | None = None,
        citations_enabled: bool = False,
    ) -> AgentResponse:
        oai_state = cast("OpenAIRunState", state)
        messages = oai_state.messages
        new_items: ResponseInputParam = messages[oai_state.message_cursor :]
        if not new_items:
            if oai_state.last_response_id is None:
                new_items = [
                    Message(
                        role="user",
                        content=[ResponseInputTextParam(type="input_text", text="")],
                    ),
                ]
            else:
                return AgentResponse(content="", tool_calls=[], done=True)

        include_param: list[ResponseIncludable] | Omit = Omit()
        if citations_enabled:
            include_param = ["web_search_call.action.sources"]

        effective_tools: list[ToolParam] = list(state.params)

        # tool_search: if a ToolSearchTool is configured and function count exceeds
        # its threshold, apply defer_loading to function tools.
        from hud.agents.openai.tools.hosted import OpenAIToolSearchTool

        tool_search_threshold: int | None = None
        for hosted in self.config.hosted_tools:
            if isinstance(hosted, OpenAIToolSearchTool):
                tool_search_threshold = hosted.threshold
                break
        if tool_search_threshold is not None:
            fn_count = sum(1 for t in effective_tools if t.get("type") == "function")
            if fn_count > tool_search_threshold:
                logger.debug(
                    "tool_search: %d function tools > threshold %d, applying defer_loading",
                    fn_count,
                    tool_search_threshold,
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
            instructions=system_prompt,
            max_output_tokens=self.max_output_tokens,
            temperature=self.temperature,
            text=self.text if self.text is not None else Omit(),
            tool_choice=self.tool_choice if self.tool_choice is not None else Omit(),
            parallel_tool_calls=self.parallel_tool_calls,
            reasoning=self.reasoning if self.reasoning is not None else Omit(),
            tools=effective_tools if effective_tools else Omit(),
            previous_response_id=(
                oai_state.last_response_id if oai_state.last_response_id is not None else Omit()
            ),
            truncation=self.truncation if self.truncation is not None else Omit(),
            include=include_param,
        )

        oai_state.last_response_id = response.id
        oai_state.message_cursor = len(messages)

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
                                    citations.append(
                                        {
                                            "type": "url_citation",
                                            "text": ann.title,
                                            "source": ann.url,
                                            "title": ann.title,
                                            "start_index": ann.start_index,
                                            "end_index": ann.end_index,
                                        }
                                    )
                                case "file_citation":
                                    citations.append(
                                        {
                                            "type": "file_citation",
                                            "text": ann.filename,
                                            "source": ann.file_id,
                                            "title": ann.filename,
                                        }
                                    )
                                case _:
                                    continue
                case "reasoning":
                    reasoning_chunks.append(
                        "".join(summary.text for summary in item.summary),
                    )
                case "function_call":
                    tool_calls.append(
                        MCPToolCall(
                            name=item.name or "",
                            arguments=json.loads(item.arguments),
                            id=item.call_id,
                        )
                    )
                case "computer_call":
                    if item.actions:
                        arguments = {"actions": [a.to_dict() for a in item.actions]}
                    elif item.action is not None:
                        arguments = item.action.to_dict()
                    else:
                        raise ValueError("OpenAI computer_call missing action")
                    call_dict: dict[str, Any] = {
                        "name": "computer",
                        "arguments": arguments,
                        "id": item.call_id,
                    }
                    if item.pending_safety_checks:
                        call_dict["pending_safety_checks"] = [
                            check.model_dump() if hasattr(check, "model_dump") else check
                            for check in item.pending_safety_checks
                        ]
                    tool_calls.append(MCPToolCall.model_validate(call_dict))
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


__all__ = ["OpenAIAgent"]
