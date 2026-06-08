"""OpenAIAgent — ``ToolAgent`` over OpenAI's Responses API."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypeVar, cast

from openai import AsyncOpenAI, Omit
from openai.types.responses import (
    ResponseIncludable,
    ResponseInputParam,
    ResponseInputTextParam,
    ResponseOutputText,
    ToolParam,
)
from openai.types.responses.easy_input_message_param import EasyInputMessageParam
from openai.types.responses.response_input_param import (
    ComputerCallOutput,
    Message,
    ResponseInputItemParam,
)

from hud.agents import gateway
from hud.agents.tool_agent import RunState, ToolAgent
from hud.agents.tools.base import last_image_data, parse_tool_arguments, result_text
from hud.agents.types import OpenAIConfig
from hud.settings import settings
from hud.types import AgentResponse, MCPToolCall, MCPToolResult

from .tools import (
    OpenAIComputerTool,
    OpenAIMCPProxyTool,
    OpenAIShellTool,
)
from .tools.base import format_openai_result

if TYPE_CHECKING:
    from hud.agents.tools.base import AgentTool
    from hud.capabilities import CapabilityClient

logger = logging.getLogger(__name__)

_T = TypeVar("_T")


def _omit_none(value: _T | None) -> _T | Omit:
    """Pass a value through, or OpenAI's ``Omit`` sentinel when it is ``None``."""
    return value if value is not None else Omit()


def _shell_output(stdout: str, stderr: str, exit_code: int) -> dict[str, Any]:
    return {
        "stdout": stdout,
        "stderr": stderr,
        "outcome": {"type": "exit", "exit_code": exit_code},
    }


def _format_shell_result(call: MCPToolCall, result: MCPToolResult) -> ResponseInputItemParam:
    structured = result.structuredContent if isinstance(result.structuredContent, dict) else {}
    output = structured.get("output")
    if not isinstance(output, list):
        output = [_shell_output("", result_text(result), 1 if result.isError else 0)]
    response: dict[str, Any] = {
        "type": "shell_call_output",
        "call_id": call.id,
        "status": "completed",
        "output": output,
    }
    max_output_length = structured.get("max_output_length")
    if isinstance(max_output_length, int):
        response["max_output_length"] = max_output_length
    return cast("ResponseInputItemParam", response)


def _format_computer_result(
    call: MCPToolCall,
    result: MCPToolResult,
) -> ResponseInputItemParam | None:
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
    if checks:
        output["acknowledged_safety_checks"] = checks
    return cast("ResponseInputItemParam", output)


def _apply_tool_search_policy(
    params: list[ToolParam],
    threshold: int | None,
) -> list[ToolParam]:
    if threshold is None or not any(tool.get("type") == "tool_search" for tool in params):
        return params
    fn_count = sum(1 for tool in params if tool.get("type") == "function")
    if fn_count <= threshold:
        return params
    logger.debug(
        "tool_search: %d function tools > threshold %d, applying defer_loading",
        fn_count,
        threshold,
    )
    # `defer_loading` is accepted by Responses but not modeled by the installed SDK.
    return [
        cast("ToolParam", {**tool, "defer_loading": True})
        if tool.get("type") == "function"
        else tool
        for tool in params
    ]


@dataclass
class OpenAIRunState(RunState[ResponseInputItemParam, ToolParam]):
    last_response_id: str | None = None
    message_cursor: int = 0


class OpenAIAgent(ToolAgent[ResponseInputItemParam, ToolParam, OpenAIConfig]):
    """OpenAI agent using the Responses API. Drives SSH, RFB, and MCP capabilities."""

    tools = (OpenAIShellTool, OpenAIComputerTool, OpenAIMCPProxyTool)

    def __init__(self, config: OpenAIConfig | None = None) -> None:
        self.config = config or OpenAIConfig()
        self.openai_client: AsyncOpenAI = cast(
            "AsyncOpenAI",
            self.config.model_client
            or gateway.resolve_model_client(
                "openai",
                direct_key=settings.openai_api_key,
                build_direct=lambda: AsyncOpenAI(api_key=settings.openai_api_key),
                direct_key_name="OPENAI_API_KEY",
            ),
        )

    # ─── ToolAgent hooks ──────────────────────────────────────────────

    async def _build_tools(
        self,
        connections: dict[str, CapabilityClient],
    ) -> tuple[dict[str, AgentTool[Any]], list[ToolParam]]:
        tools, params = await super()._build_tools(connections)
        return tools, _apply_tool_search_policy(params, self.config.tool_search_threshold)

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
        state: RunState[ResponseInputItemParam, ToolParam],
    ) -> ResponseInputItemParam | list[ResponseInputItemParam] | None:
        tool = state.tools.get(call.name)
        api_type = tool.spec.api_type if tool else None
        if api_type == "computer":
            return _format_computer_result(call, result)
        if api_type == "shell":
            return _format_shell_result(call, result)
        return format_openai_result(call, result)

    async def get_response(
        self,
        state: RunState[ResponseInputItemParam, ToolParam],
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

        # Hosted-tool transforms such as defer_loading are applied at tool-build time.
        effective_tools = list(state.params)

        response = await self.openai_client.responses.create(
            model=self.model,
            input=new_items,
            instructions=system_prompt,
            max_output_tokens=self.config.max_output_tokens,
            temperature=self.config.temperature,
            text=_omit_none(self.config.text),
            tool_choice=_omit_none(self.config.tool_choice),
            parallel_tool_calls=self.config.parallel_tool_calls,
            reasoning=_omit_none(self.config.reasoning),
            tools=effective_tools if effective_tools else Omit(),
            previous_response_id=_omit_none(oai_state.last_response_id),
            truncation=_omit_none(self.config.truncation),
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
                            arguments=parse_tool_arguments(item.arguments, item.name),
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
                            check.model_dump() for check in item.pending_safety_checks
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
                    logger.debug("Unhandled OpenAI response item type: %s", item.type)
                    continue

        return AgentResponse(
            content="".join(text_chunks),
            reasoning="\n".join(reasoning_chunks) if reasoning_chunks else None,
            citations=citations,
            tool_calls=tool_calls,
            done=not tool_calls,
            finish_reason=response.status,
        )


__all__ = ["OpenAIAgent"]
