"""Board-size setup function for 2048."""

from __future__ import annotations

import json
from contextlib import contextmanager
from typing import Any

from mcp.types import ContentBlock, TextContent

try:
    from hud.telemetry import trace
except ModuleNotFoundError:  # pragma: no cover - optional dependency safeguard

    @contextmanager
    def trace(*_args: Any, **_kwargs: Any) -> Any:  # type: ignore[misc]
        yield

from . import setup

DEFAULT_BOARD_SIZE = 4
MIN_BOARD_SIZE = 3
MAX_BOARD_SIZE = 8


def _normalize_board_size(value: Any) -> tuple[int, str | None]:
    """Convert ``value`` into a clamped integer plus a note if coercion was needed."""
    note: str | None = None
    try:
        size = int(value)
    except (TypeError, ValueError):
        size = DEFAULT_BOARD_SIZE
        note = (
            f"Invalid board size {value!r}; defaulting to {DEFAULT_BOARD_SIZE}."
        )
        return size, note

    if size < MIN_BOARD_SIZE:
        note = (
            f"Requested board size {size} is below {MIN_BOARD_SIZE}; "
            f"using {MIN_BOARD_SIZE}."
        )
        return MIN_BOARD_SIZE, note

    if size > MAX_BOARD_SIZE:
        note = (
            f"Requested board size {size} exceeds {MAX_BOARD_SIZE}; "
            f"using {MAX_BOARD_SIZE}."
        )
        return MAX_BOARD_SIZE, note

    return size, None


@setup.tool("board")
async def setup_board(board_size: int = DEFAULT_BOARD_SIZE) -> list[ContentBlock]:
    """Initialize a new game with the specified board size."""
    normalized_size, validation_note = _normalize_board_size(board_size)
    game = setup.env

    with trace(
        "text-2048 setup",
        attrs={
            "requested_board_size": board_size,
            "board_size": normalized_size,
            "validation_note": validation_note or "",
        },
    ):
        game.reset(size=normalized_size)
        board_display = game.get_board_ascii()
        state_payload = {
            "requested_board_size": board_size,
            "board_size": normalized_size,
            "state": game.get_state(),
        }

    lines = [f"{normalized_size}x{normalized_size} game initialized"]
    if validation_note:
        lines.append(validation_note)
    lines.append("")
    lines.append(board_display)

    return [
        TextContent(text="\n".join(lines), type="text"),
        TextContent(text=json.dumps(state_payload), type="text"),
    ]
