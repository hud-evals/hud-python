"""Board-size setup function for 2048."""

from fastmcp import Context
from mcp.types import TextContent, ContentBlock
from . import setup


@setup.tool("board")
async def setup_board(ctx: Context, board_size: int = 4) -> list[ContentBlock]:
    """Initialize a new game with the specified board size."""
    game = setup.env
    game.reset(size=board_size)

    # Get the initial board state to show the agent
    board_display = game.get_board_ascii()

    # Return the initial board display
    return [
        TextContent(
            text=f"{board_size}x{board_size} game initialized\n\n{board_display}", type="text"
        )
    ]
