"""Move tool for the 2048 game."""

import logging
from typing import Any
from mcp.types import TextContent, ContentBlock
from hud.tools.base import BaseTool

logger = logging.getLogger(__name__)


class MoveTool(BaseTool):
    """Tool for making moves in the 2048 game."""

    def __init__(self, env: Any = None):
        """Initialize the move tool.

        Args:
            context: The game instance
        """
        super().__init__(
            env=env,
            name="move",
            title="Move Tiles",
            description="Make a move in the 2048 game by sliding tiles in a direction",
        )

    async def __call__(self, direction: str) -> list[ContentBlock]:
        """Make a move in the 2048 game.

        Args:
            direction: The direction to move ('up', 'down', 'left', 'right')
        """
        if self.env is None:
            return [TextContent(text="❌ Game not initialized. Run setup first.", type="text")]

        direction = direction.lower()
        if direction not in ["up", "down", "left", "right"]:
            return [
                TextContent(
                    text=f"❌ Invalid direction: {direction}. Use: up, down, left, right",
                    type="text",
                )
            ]

        # Make the move using context (the game)
        moved = self.env.move(direction)

        if not moved:
            return [
                TextContent(
                    text=f"❌ Cannot move {direction} - no valid moves in that direction",
                    type="text",
                )
            ]

        # Get game state
        state = self.env.get_state()

        # Format response
        board_str = self.env.get_board_ascii()
        text = f"✅ Moved {direction}\n"
        text += f"Score: {state['score']} | Moves: {state['moves']}\n"
        text += f"{board_str}"

        if state["game_over"]:
            text += "\n🎮 GAME OVER!"

        return [TextContent(text=text, type="text")]
