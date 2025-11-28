"""Operator agent built on top of OpenAIAgent."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, Literal

import mcp.types as types
from openai.types.responses import (
    ApplyPatchToolParam,
    ComputerToolParam,
    FunctionShellToolParam,
    FunctionToolParam,
    ResponseComputerToolCallOutputScreenshotParam,
    ResponseInputParam,
)
from openai.types.responses.response_input_param import (
    ComputerCallOutput,
)
from openai.types.shared_params.reasoning import Reasoning
from pydantic import ConfigDict

from hud.tools.computer.settings import computer_settings
from hud.types import BaseAgentConfig, MCPToolCall, MCPToolResult
from hud.utils.types import with_signature

from .base import BaseCreateParams, MCPAgent
from .openai import OpenAIAgent, OpenAIConfig

if TYPE_CHECKING:
    from openai.types.responses.response_computer_tool_call import PendingSafetyCheck

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


class OperatorConfig(OpenAIConfig):
    """Configuration model for `OperatorAgent`."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    model_name: str = "Operator"
    checkpoint_name: str = "computer-use-preview"
    environment: Literal["windows", "mac", "linux", "ubuntu", "browser"] = "linux"


class OperatorCreateParams(BaseCreateParams, OperatorConfig):
    pass


class OperatorAgent(OpenAIAgent):
    """
    Backwards-compatible Operator agent built on top of OpenAIAgent.
    """

    metadata: ClassVar[dict[str, Any] | None] = {
        "display_width": computer_settings.OPENAI_COMPUTER_WIDTH,
        "display_height": computer_settings.OPENAI_COMPUTER_HEIGHT,
    }
    # base class will ensure that the computer tool is available
    required_tools: ClassVar[list[str]] = ["openai_computer"]
    config_cls: ClassVar[type[BaseAgentConfig]] = OperatorConfig

    @with_signature(OperatorCreateParams)
    @classmethod
    def create(cls, **kwargs: Any) -> OperatorAgent:  # pyright: ignore[reportIncompatibleMethodOverride]
        return MCPAgent.create.__func__(cls, **kwargs)  # type: ignore[return-value]

    def __init__(self, params: OperatorCreateParams | None = None, **kwargs: Any) -> None:
        super().__init__(params, **kwargs)  # type: ignore[arg-type]
        self.config: OperatorConfig  # type: ignore[assignment]

        self._operator_computer_tool_name = "openai_computer"
        self._operator_display_width = computer_settings.OPENAI_COMPUTER_WIDTH
        self._operator_display_height = computer_settings.OPENAI_COMPUTER_HEIGHT
        self._operator_environment: Literal["windows", "mac", "linux", "ubuntu", "browser"] = (
            self.config.environment
        )
        self.environment = self.config.environment

        # add pending call id and safety checks to the agent
        self.pending_call_id: str | None = None
        self.pending_safety_checks: list[PendingSafetyCheck] = []

        # override reasoning to "summary": "auto"
        if self.reasoning is None:
            self.reasoning = Reasoning(summary="auto")
        else:
            self.reasoning["summary"] = "auto"

        # override truncation to "auto"
        self.truncation = "auto"

        if self.system_prompt:
            self.system_prompt = f"{self.system_prompt}\n\n{OPERATOR_INSTRUCTIONS}"
        else:
            self.system_prompt = OPERATOR_INSTRUCTIONS

    def _reset_response_state(self) -> None:
        super()._reset_response_state()
        self.pending_call_id = None
        self.pending_safety_checks = []

    def _to_openai_tool(
        self, tool: types.Tool
    ) -> (
        FunctionShellToolParam | ApplyPatchToolParam | FunctionToolParam | ComputerToolParam | None
    ):
        if tool.name == self._operator_computer_tool_name:
            return ComputerToolParam(
                type="computer_use_preview",
                display_width=self._operator_display_width,
                display_height=self._operator_display_height,
                environment=self._operator_environment,
            )
        return super()._to_openai_tool(tool)

    def _extract_tool_call(self, item: Any) -> MCPToolCall | None:
        """Route computer_call to the OpenAI-specific computer tool."""
        if item.type == "computer_call":
            self.pending_safety_checks = item.pending_safety_checks
            return MCPToolCall(
                name=self._operator_computer_tool_name,
                arguments=item.action.to_dict(),
                id=item.call_id,
            )
        return super()._extract_tool_call(item)

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
                output_payload = ComputerCallOutput(
                    type="computer_call_output",
                    call_id=call_id,
                    output=ResponseComputerToolCallOutputScreenshotParam(
                        type="computer_screenshot",
                        image_url=f"data:image/png;base64,{screenshot}",
                    ),
                    acknowledged_safety_checks=acknowledged_checks if acknowledged_checks else None,
                )
                computer_outputs.append(output_payload)
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
