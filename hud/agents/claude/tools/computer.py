"""Claude's ``computer`` tool, executed over RFB."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from hud.agents.computer.claude import ACTIONS, ClaudeComputer
from hud.agents.tools.rfb import RFBTool

from .base import ClaudeToolSpec, is_anthropic_model

if TYPE_CHECKING:
    from anthropic.types.beta import (
        BetaComputerToolset20260801Param,
        BetaToolComputerUse20250124Param,
        BetaToolComputerUse20251124Param,
        BetaToolUnionParam,
    )

    from hud.types import ToolResult


# Current Claude models take the computer toolset; only the older models listed in
# ``CLAUDE_COMPUTER_SPECS`` take a single computer tool.
COMPUTER_TOOLSET_SPEC = ClaudeToolSpec(api_type="computer_toolset_20260801", api_name="computer")

CLAUDE_COMPUTER_SPECS: tuple[ClaudeToolSpec, ...] = (
    ClaudeToolSpec(
        api_type="computer_20251124",
        api_name="computer",
        beta="computer-use-2025-11-24",
        supported_models=(
            "*claude-opus-4-5*",
            "*claude-opus-4-6*",
            "*claude-sonnet-4-6*",
            "*claude-opus-4-7*",
            "*claude-opus-4-8*",
        ),
    ),
    ClaudeToolSpec(
        api_type="computer_20250124",
        api_name="computer",
        beta="computer-use-2025-01-24",
        # Claude Opus 4 and Sonnet 4 go by bare, ``-0`` alias, dated, and Vertex ``@`` ids.
        supported_models=(
            "*claude-sonnet-4-5*",
            "*claude-haiku-4-5*",
            "*claude-opus-4-1*",
            "*claude-opus-4",
            "*claude-opus-4-0*",
            "*claude-opus-4-2025*",
            "*claude-opus-4@*",
            "*claude-sonnet-4",
            "*claude-sonnet-4-0*",
            "*claude-sonnet-4-2025*",
            "*claude-sonnet-4@*",
        ),
    ),
)

GENERIC_COMPUTER_SPEC = ClaudeToolSpec(
    api_type="function",
    api_name="computer",
)

_COMPUTER_INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "action": {
            "type": "string",
            "enum": list(ACTIONS),
            "description": "The action to perform on the screen.",
        },
        "coordinate": {
            "type": "array",
            "items": {"type": "integer"},
            "minItems": 2,
            "maxItems": 2,
            "description": "[x, y] pixel coordinate for click/move/scroll, or drag end point.",
        },
        "start_coordinate": {
            "type": "array",
            "items": {"type": "integer"},
            "minItems": 2,
            "maxItems": 2,
            "description": "[x, y] drag start point for `left_click_drag`.",
        },
        "text": {
            "type": "string",
            "description": (
                "Text to type for `type`; key chord (xdotool syntax, e.g. `ctrl+s`) for "
                "`key`/`hold_key`; optional modifier keys held during clicks and scrolls."
            ),
        },
        "duration": {
            "type": "number",
            "description": "Seconds to hold for `hold_key` or pause for `wait`.",
        },
        "scroll_direction": {
            "type": "string",
            "enum": ["up", "down", "left", "right"],
            "description": "Direction for `scroll`.",
        },
        "scroll_amount": {
            "type": "integer",
            "description": "Number of scroll clicks for `scroll`.",
        },
        "repeat": {
            "type": "integer",
            "description": "Times to repeat the key chord for `key`.",
        },
        "region": {
            "type": "array",
            "items": {"type": "integer"},
            "minItems": 4,
            "maxItems": 4,
            "description": "[x0, y0, x1, y1] region to magnify for `zoom`.",
        },
    },
    "required": ["action"],
}


class ClaudeComputerTool(RFBTool):
    """Claude's ``computer`` tool over RFB.

    The computer toolset for current Claude models, a versioned native schema for the older
    models in ``CLAUDE_COMPUTER_SPECS``, and a function schema for other models.
    """

    name = "computer"

    @classmethod
    def default_spec(cls, model: str) -> ClaudeToolSpec | None:
        if not is_anthropic_model(model):
            return GENERIC_COMPUTER_SPEC
        for candidate in CLAUDE_COMPUTER_SPECS:
            if candidate.supports_model(model):
                return candidate
        return COMPUTER_TOOLSET_SPEC

    @property
    def is_toolset(self) -> bool:
        """Whether the model calls this tool as the computer toolset, one tool per action."""
        return self.spec.api_type == COMPUTER_TOOLSET_SPEC.api_type

    def to_params(
        self,
    ) -> (
        BetaComputerToolset20260801Param
        | BetaToolComputerUse20250124Param
        | BetaToolComputerUse20251124Param
        | BetaToolUnionParam
    ):
        if self.is_toolset:
            # The model calls each member as its own tool and sizes clicks from screenshots.
            return {"type": "computer_toolset_20260801"}
        if self.spec.api_type == "function":
            return cast(
                "BetaToolUnionParam",
                {
                    "name": self.name,
                    "description": (
                        "Control the remote computer's screen, mouse, and keyboard. "
                        f"The display is {self.client.width}x{self.client.height} pixels. "
                        "Most actions return a screenshot of the result."
                    ),
                    "input_schema": _COMPUTER_INPUT_SCHEMA,
                },
            )
        if self.spec.api_type == "computer_20251124":
            return cast(
                "BetaToolComputerUse20251124Param",
                {
                    "type": "computer_20251124",
                    "name": self.name,
                    "display_width_px": self.client.width,
                    "display_height_px": self.client.height,
                    "display_number": 1,
                    "enable_zoom": True,
                },
            )
        return cast(
            "BetaToolComputerUse20250124Param",
            {
                "type": "computer_20250124",
                "name": self.name,
                "display_width_px": self.client.width,
                "display_height_px": self.client.height,
                "display_number": 1,
            },
        )

    async def execute(self, arguments: dict[str, Any]) -> ToolResult:
        return await ClaudeComputer(self).run(arguments)
