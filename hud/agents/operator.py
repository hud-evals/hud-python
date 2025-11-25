"""Operator agent built on top of OpenAIAgent."""

from __future__ import annotations

import json
from typing import Any, ClassVar, Literal, cast

import mcp.types as types
from openai import AsyncOpenAI, Omit
from openai.types.responses import (
    ResponseComputerToolCall,
    ResponseFunctionToolCall,
    ResponseInputParam,
    ResponseInputTextParam,
    ResponseOutputMessage,
    ResponseOutputText,
    ResponseReasoningItem,
    ToolParam,
)
from openai.types.responses.response_input_param import (
    ComputerCallOutput,  # noqa: TC002
    Message,  # noqa: TC002
)
from openai.types.shared_params.reasoning import Reasoning

import hud
from hud.tools.computer.settings import computer_settings
from hud.types import AgentResponse, MCPToolCall, MCPToolResult

from .openai import OpenAIAgent

OPERATOR_INSTRUCTIONS = """
You are an autonomous computer-using agent. Follow these guidelines:

1. NEVER ask for confirmation. Complete all tasks autonomously.
2. Do NOT send messages like "I need to confirm before..." or "Do you want me to
   continue?" - just proceed.
3. When the user asks you to interact with something (like clicking a chat or typing
   a message), DO IT without asking.
4. Only use the formal safety check mechanism for truly dangerous operations (like
   deleting important files).
5. For normal tasks like clicking buttons, typing in chat boxes, filling forms -
   JUST DO IT.
6. The user has already given you permission by running this agent. No further
   confirmation is needed.
7. Be decisive and action-oriented. Complete the requested task fully.

Remember: You are expected to complete tasks autonomously. The user trusts you to do
what they asked.
""".strip()


class OperatorAgent(OpenAIAgent):
    """
    Backwards-compatible Operator agent built on top of OpenAIAgent.
    """

    metadata: ClassVar[dict[str, Any] | None] = {
        "display_width": computer_settings.OPENAI_COMPUTER_WIDTH,
        "display_height": computer_settings.OPENAI_COMPUTER_HEIGHT,
    }
    required_tools: ClassVar[list[str]] = ["openai_computer"]

    def __init__(
        self,
        model_client: AsyncOpenAI | None = None,
        model: str = "computer-use-preview",
        environment: Literal["windows", "mac", "linux", "browser"] = "linux",
        validate_api_key: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model_client=model_client,
            model=model,
            validate_api_key=validate_api_key,
            **kwargs,
        )
        self._operator_computer_tool_name = "openai_computer"
        self._operator_display_width = computer_settings.OPENAI_COMPUTER_WIDTH
        self._operator_display_height = computer_settings.OPENAI_COMPUTER_HEIGHT
        self._operator_environment = environment
        self.model_name = "Operator"
        self.environment = environment

        # override reasoning to "summary": "auto"
        if self.reasoning is None:
            self.reasoning = Reasoning(summary="auto")
        else:
            self.reasoning["summary"] = "auto"

        if self.system_prompt:
            self.system_prompt = f"{self.system_prompt}\n\n{OPERATOR_INSTRUCTIONS}"
        else:
            self.system_prompt = OPERATOR_INSTRUCTIONS

    def _build_openai_tools(self) -> None:
        super()._convert_tools_for_openai()
        if not any(
            tool.name == self._operator_computer_tool_name for tool in self.get_available_tools()
        ):
            raise ValueError(
                f"MCP computer tool '{self._operator_computer_tool_name}' is required "
                "but not available."
            )
        self._openai_tools.append(
            cast(
                "ToolParam",
                {
                    "type": "computer_use_preview",
                    "display_width": self._operator_display_width,
                    "display_height": self._operator_display_height,
                    "environment": self._operator_environment,
                },
            )
        )

    @hud.instrument(
        span_type="agent",
        record_args=False,
        record_result=True,
    )
    async def get_response(self, messages: ResponseInputParam) -> AgentResponse:
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
            # truncation MUST be set to "auto" for the computer-use-preview model
            truncation="auto",
        )
        self.last_response_id = response.id
        self._message_cursor = len(messages)
        self.pending_call_id = None

        agent_response = AgentResponse(content="", tool_calls=[], done=True)
        text_chunks: list[str] = []
        reasoning_chunks: list[str] = []

        for item in response.output:
            if isinstance(item, ResponseComputerToolCall):
                tool_call = self._convert_computer_tool_call(item)
                if tool_call:
                    agent_response.tool_calls.append(tool_call)
            elif isinstance(item, ResponseFunctionToolCall):
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
            elif isinstance(item, ResponseOutputMessage) and item.type == "message":
                text = "".join(
                    content.text
                    for content in item.content
                    if isinstance(content, ResponseOutputText)
                )
                if text:
                    text_chunks.append(text)
            elif isinstance(item, ResponseReasoningItem) and item.summary:
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
        remaining_calls: list[MCPToolCall] = []
        remaining_results: list[MCPToolResult] = []
        computer_outputs: ResponseInputParam = []
        ordering: list[tuple[str, int]] = []

        for call, result in zip(tool_calls, tool_results, strict=False):
            if call.name == self._operator_computer_tool_name:
                screenshot = self._extract_latest_screenshot(result)
                if not screenshot:
                    self.console.warning_log(
                        "Computer tool result missing screenshot; skipping output."
                    )
                    continue
                call_id = call.id or self.pending_call_id
                if not call_id:
                    self.console.warning_log("Computer tool call missing ID; skipping output.")
                    continue
                acknowledged_checks = []
                for check in self.pending_safety_checks:
                    if hasattr(check, "model_dump"):
                        acknowledged_checks.append(check.model_dump())
                    elif isinstance(check, dict):
                        acknowledged_checks.append(check)
                output_payload: dict[str, Any] = {
                    "type": "computer_call_output",
                    "call_id": call_id,
                    "output": {
                        "type": "input_image",
                        "image_url": f"data:image/png;base64,{screenshot}",
                    },
                }
                if acknowledged_checks:
                    output_payload["acknowledged_safety_checks"] = acknowledged_checks
                computer_outputs.append(cast("ComputerCallOutput", output_payload))
                self.pending_call_id = None
                self.pending_safety_checks = []
                ordering.append(("computer", len(computer_outputs) - 1))
            else:
                remaining_calls.append(call)
                remaining_results.append(result)
                ordering.append(("function", len(remaining_calls) - 1))

        formatted: ResponseInputParam = []
        function_outputs: ResponseInputParam = []
        if remaining_calls:
            function_outputs = await super().format_tool_results(remaining_calls, remaining_results)

        for kind, idx in ordering:
            if kind == "computer":
                if idx < len(computer_outputs):
                    formatted.append(computer_outputs[idx])
            else:
                if idx < len(function_outputs):
                    formatted.append(function_outputs[idx])
        return formatted

    def _extract_latest_screenshot(self, result: MCPToolResult) -> str | None:
        if not result.content:
            return None
        for content in reversed(result.content):
            if isinstance(content, types.ImageContent):
                return content.data
            if isinstance(content, types.TextContent) and result.isError:
                self.console.error_log(f"Computer tool error: {content.text}")
        return None

    def _convert_computer_tool_call(
        self, tool_call: ResponseComputerToolCall
    ) -> MCPToolCall | None:
        self.pending_call_id = tool_call.call_id
        self.pending_safety_checks = tool_call.pending_safety_checks
        call = MCPToolCall(
            name=self._operator_computer_tool_name,
            arguments=tool_call.action.model_dump(),
            id=tool_call.call_id,
        )
        call.pending_safety_checks = tool_call.pending_safety_checks  # type: ignore[attr-defined]
        return call
