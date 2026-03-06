"""Operator agent built on top of OpenAIAgent."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, ClassVar, Literal

from openai.types.responses import (
    ComputerUsePreviewToolParam,
    ToolParam,
)
from openai.types.shared_params.reasoning import Reasoning

from hud.tools.computer.settings import computer_settings
from hud.tools.native_types import NativeToolSpec
from hud.types import AgentType, BaseAgentConfig, MCPToolCall
from hud.utils.types import with_signature

from .base import MCPAgent
from .openai import OpenAIAgent
from .types import OperatorConfig, OperatorCreateParams

if TYPE_CHECKING:
    import mcp.types as types

logger = logging.getLogger(__name__)

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
    # base class will ensure that the computer tool is available
    required_tools: ClassVar[list[str]] = ["openai_computer"]
    config_cls: ClassVar[type[BaseAgentConfig]] = OperatorConfig

    @classmethod
    def agent_type(cls) -> AgentType:
        """Return the AgentType for Operator."""
        return AgentType.OPERATOR

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

    def _build_native_tool(self, tool: types.Tool, spec: NativeToolSpec) -> ToolParam | None:
        """Override to handle computer tools specially for Operator API."""
        # Use Operator's computer_use_preview for the designated computer tool
        if tool.name == self._operator_computer_tool_name:
            return ComputerUsePreviewToolParam(
                type="computer_use_preview",
                display_width=self._operator_display_width,
                display_height=self._operator_display_height,
                environment=self._operator_environment,
            )
        # Skip other computer tools (only one computer tool allowed)
        if tool.name == "computer" or tool.name.endswith("_computer"):
            return None
        # Delegate to parent for shell, apply_patch, etc.
        return super()._build_native_tool(tool, spec)

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

    _LEGACY_COMPUTER_NAMES = ("openai_computer",)

    def _legacy_native_spec_fallback(self, tool: types.Tool) -> NativeToolSpec | None:
        """Detect Operator native tools by name for backwards compatibility.

        Each tuple is ordered by preference — first name that exists wins.
        Only returns a spec if this tool IS that preferred match.
        """
        available = {t.name for t in (self._available_tools or [])} | {tool.name}
        preferred = lambda names: next((n for n in names if n in available), None) == tool.name

        if preferred(self._LEGACY_COMPUTER_NAMES):
            logger.debug("Legacy fallback: detected %s as computer tool", tool.name)
            return NativeToolSpec(
                api_type="computer_use_preview",
                api_name="computer",
                role="computer",
            )

        return super()._legacy_native_spec_fallback(tool)
