"""Gemini Computer Use preset agent.

The native Computer Use implementation lives in GeminiAgent. This class only
keeps the gemini_cua agent type/default model preset.
"""

from __future__ import annotations

from typing import Any, ClassVar

from hud.tools.computer.settings import computer_settings
from hud.types import AgentType, BaseAgentConfig
from hud.utils.types import with_signature

from .base import MCPAgent
from .gemini import GeminiAgent
from .types import GeminiCUAConfig, GeminiCUACreateParams


class GeminiCUAAgent(GeminiAgent):
    """
    Gemini Computer Use Agent that extends GeminiAgent with computer use capabilities.

    This agent uses Gemini's native computer use capabilities but executes
    tools through MCP servers instead of direct implementation.
    """

    metadata: ClassVar[dict[str, Any] | None] = {
        "display_width": computer_settings.GEMINI_COMPUTER_WIDTH,
        "display_height": computer_settings.GEMINI_COMPUTER_HEIGHT,
    }
    required_tools: ClassVar[list[str]] = ["gemini_computer"]
    config_cls: ClassVar[type[BaseAgentConfig]] = GeminiCUAConfig

    @classmethod
    def agent_type(cls) -> AgentType:
        """Return the AgentType for Gemini CUA."""
        return AgentType.GEMINI_CUA

    @with_signature(GeminiCUACreateParams)
    @classmethod
    def create(cls, **kwargs: Any) -> GeminiCUAAgent:  # pyright: ignore[reportIncompatibleMethodOverride]
        return MCPAgent.create.__func__(cls, **kwargs)  # type: ignore[return-value]
