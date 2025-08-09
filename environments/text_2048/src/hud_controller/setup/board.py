"""Board-size setup function for 2048."""

from fastmcp import Context
from hud.tools.types import SetupResult
from . import setup


@setup.tool("board")
async def setup_board(ctx: Context, board_size: int = 4):
    """Initialize a new game with the specified board size."""
    game = setup.env
    game.reset(size=board_size)
    return SetupResult(
        content=f"{board_size}x{board_size} game initialized",
    )
