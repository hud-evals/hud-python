"""Connect Four self-play environment.

A step up from tic-tac-toe: a 6x7 board, drop-a-disc moves, and decisive
outcomes (draws are rare on a full 42-cell board), so the reward keeps a real
spread as the policy improves instead of collapsing to all-draws.

Starting order is randomized per task (seed % 2 decides who drops first). The
outer agent plays one side for a full game; the inner model (same slug) plays
the other. Reward is always from the outer agent's perspective: win=1.0,
draw=0.5, loss=0.0.

Inner-model token data (prompt_token_ids, token_ids, logprobs) is captured from
the HUD gateway response and stored in EvaluationResult.info so the training
loop can train on both sides of each game at once.
"""

from __future__ import annotations

import asyncio
import re
import socket
import time
from typing import Any

from fastmcp import FastMCP

from hud.capabilities import Capability
from hud.environment import Environment
from hud.graders import EvaluationResult

_INNER_MODEL: str = "c4-selfplay"
_OUTER_MARK: str = "X"  # set per game; "X" drops first, "O" drops second

# Per-game inner model samples (reset at game start, read at game end).
_inner_samples: list[dict[str, Any]] = []

# Shared rules text — the SAME framing for both sides, so the matchup is fair
# self-play between equal copies (the inner model used to get a terse, rule-free
# prompt and only 8 output tokens, which handed the outer agent a systematic edge).
_RULES = (
    "Connect Four on a 6-row by 7-column board. Drop a disc into a column (0-6); "
    "it falls to the lowest empty cell. Win by connecting four of your discs in a "
    "row — horizontally, vertically, or diagonally."
)
# Output budget for a move. Used for the inner model; the outer agent reasons under
# the harness with no tighter cap, so this keeps the two sides comparable.
_MOVE_MAX_TOKENS = 64

# ── game logic ─────────────────────────────────────────────────────────────────

ROWS, COLS = 6, 7
_DIRS = ((0, 1), (1, 0), (1, 1), (1, -1))  # right, down, down-right, down-left


class ConnectFour:
    def __init__(self) -> None:
        self.board: list[list[str | None]] = [[None] * COLS for _ in range(ROWS)]
        self.current: str = "X"

    def reset(self) -> None:
        self.board = [[None] * COLS for _ in range(ROWS)]
        self.current = "X"

    def available(self) -> list[int]:
        """Columns that aren't full (top cell empty)."""
        return [c for c in range(COLS) if self.board[0][c] is None]

    def drop(self, col: int, mark: str) -> int:
        """Drop a disc into the lowest empty row of a column. Returns the row it
        landed in, or -1 if the column is full."""
        for r in range(ROWS - 1, -1, -1):
            if self.board[r][col] is None:
                self.board[r][col] = mark
                self.current = "O" if mark == "X" else "X"
                return r
        return -1

    def winner(self) -> str | None:
        b = self.board
        for r in range(ROWS):
            for c in range(COLS):
                mark = b[r][c]
                if mark is None:
                    continue
                for dr, dc in _DIRS:
                    rr, cc, count = r, c, 0
                    while 0 <= rr < ROWS and 0 <= cc < COLS and b[rr][cc] == mark:
                        count += 1
                        rr += dr
                        cc += dc
                    if count >= 4:
                        return mark
        return None

    def over(self) -> bool:
        return self.winner() is not None or not self.available()

    def render(self) -> str:
        header = " " + " ".join(str(c) for c in range(COLS))
        grid = [" " + " ".join(self.board[r][c] or "." for c in range(COLS)) for r in range(ROWS)]
        lines = [header, *grid]
        w = self.winner()
        if w:
            lines.append(f"Winner: {w}")
        elif not self.available():
            lines.append("Draw")
        else:
            lines.append(
                f"Current player: {self.current}  |  Available columns: {self.available()}"
            )
        return "\n".join(lines)


game = ConnectFour()

# ── MCP server ─────────────────────────────────────────────────────────────────


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


_PORT = _free_port()
server = FastMCP(name="connect-four")


async def _inner_move(inner_mark: str) -> int:
    """Ask the inner model to pick a column. Falls back to first available.

    Also captures token-level training data (prompt_token_ids, token_ids,
    logprobs) into _inner_samples so the training loop can train on both sides
    of each game with a flipped reward.
    """
    from hud.utils.gateway import build_gateway_client

    client = build_gateway_client("openai")
    available = game.available()

    try:
        resp = await client.chat.completions.create(
            model=_INNER_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": f"You are playing Connect Four as {inner_mark}. {_RULES}",
                },
                {
                    "role": "user",
                    "content": (
                        f"{game.render()}\n\n"
                        "Choose your move. Reason briefly if you want, then end your reply "
                        "with the column number to drop into."
                    ),
                },
            ],
            max_tokens=_MOVE_MAX_TOKENS,
            logprobs=True,
            extra_body={"return_token_ids": True},
        )
        choice = resp.choices[0]
        # HUD gateway returns these as non-standard attributes when return_token_ids=True
        prompt_ids = getattr(choice, "prompt_token_ids", None)
        token_ids = getattr(choice, "token_ids", None)
        if prompt_ids is not None and token_ids is not None:
            content_lp = choice.logprobs.content if choice.logprobs else None
            _inner_samples.append(
                {
                    "prompt_token_ids": list(prompt_ids),
                    "output_token_ids": list(token_ids),
                    "output_logprobs": [tok.logprob for tok in content_lp] if content_lp else [],
                }
            )
        # The model may reason before answering, so take the LAST valid column it
        # names, not the first integer it mentions.
        text = choice.message.content or ""
        for tok in reversed(re.findall(r"\d+", text)):
            col = int(tok)
            if col in available:
                return col
    except Exception:
        pass

    return available[0]


@server.tool
async def make_move(column: int) -> str:
    """Drop your disc into a column (0-6), then the inner model responds.

    Columns are numbered left to right, 0-6. Discs stack from the bottom.
    Returns the board after both moves. Keep calling until you see "Winner" or
    "Draw".
    """
    if game.over():
        return f"Game is already over.\n{game.render()}"

    outer_mark = _OUTER_MARK
    inner_mark = "O" if outer_mark == "X" else "X"

    if game.current != outer_mark:
        return f"It's {game.current}'s turn (inner model), not yours. Board:\n{game.render()}"

    if column not in game.available():
        return f"Column {column} is full or invalid. Available: {game.available()}\n{game.render()}"

    game.drop(column, outer_mark)
    if game.over():
        return game.render()

    col = await _inner_move(inner_mark)
    game.drop(col, inner_mark)

    return game.render()


@server.tool
def get_state() -> str:
    """Return the current board, whose turn it is, and available columns."""
    return game.render()


# ── environment ────────────────────────────────────────────────────────────────

env = Environment(name="connect-four-selfplay")
_server_task: asyncio.Task[None] | None = None


async def _listening(host: str, port: int, timeout: float = 10.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), 0.2):
                return
        except OSError:
            await asyncio.sleep(0.1)
    raise RuntimeError(f"nothing listening on {host}:{port}")


@env.initialize
async def _up() -> None:
    global _server_task
    if _server_task is None:
        _server_task = asyncio.create_task(
            server.run_async(transport="http", host="127.0.0.1", port=_PORT)
        )
        await _listening("127.0.0.1", _PORT)
    env.add_capability(Capability.mcp(name="tools", url=f"http://127.0.0.1:{_PORT}/mcp"))


@env.shutdown
async def _down() -> None:
    global _server_task
    if _server_task is not None:
        _server_task.cancel()
        _server_task = None


@env.template()
async def play_self(model: str = _INNER_MODEL, seed: int = 0) -> None:
    """Self-play game. seed % 2 decides who drops first: even → outer is X, odd → outer is O."""
    global _INNER_MODEL, _OUTER_MARK, _inner_samples
    _INNER_MODEL = model
    _OUTER_MARK = "X" if seed % 2 == 0 else "O"
    inner_mark = "O" if _OUTER_MARK == "X" else "X"

    game.reset()
    _inner_samples = []  # fresh per game

    # If the inner model goes first (outer is O), let it make the opening move now.
    if _OUTER_MARK == "O":
        opening = await _inner_move("X")
        game.drop(opening, "X")

    yield (
        f"You are playing Connect Four as {_OUTER_MARK} against {model} playing {inner_mark}.\n"
        f"{_RULES}\n"
        f"{'You go first.' if _OUTER_MARK == 'X' else 'The opponent opened — it is now your turn.'}\n"
        "Make each of your moves by calling make_move(column); after your move the "
        "opponent responds automatically.\n"
        "Keep playing until you see 'Winner' or 'Draw'.\n\n"
        f"Current board:\n{game.render()}"
    )

    w = game.winner()
    reward = 1.0 if w == _OUTER_MARK else (0.0 if w is not None else 0.5)

    yield EvaluationResult(
        reward=reward,
        content=f"Winner: {w or 'Draw'}",
        info={
            "winner": w,
            "outer_mark": _OUTER_MARK,
            "board": game.board,
            "model": model,
            "inner_samples": _inner_samples,  # token data for symmetric training
        },
    )


tasks = [play_self(model="c4-selfplay", seed=s) for s in range(2)]
