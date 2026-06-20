"""Tic-tac-toe self-play environment.

Starting order is randomized per task (seed % 2 determines who goes first).
The outer agent always plays the same role for a full game; the inner model
(same slug) plays the other side. Reward is always from the outer agent's
perspective: win=1.0, draw=0.5, loss=0.0.

Inner model token data (prompt_token_ids, token_ids, logprobs) is captured
from the HUD gateway response and stored in EvaluationResult.info so the
training loop can train on both sides of each game simultaneously.
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

_INNER_MODEL: str = "ttt-selfplay-389d2c"
_OUTER_MARK: str = "X"  # set per game; "X" goes first, "O" goes second

# Per-game inner model samples (reset at game start, read at game end).
_inner_samples: list[dict[str, Any]] = []

# ── game logic ─────────────────────────────────────────────────────────────────

_WINS = [
    (0, 1, 2),
    (3, 4, 5),
    (6, 7, 8),  # rows
    (0, 3, 6),
    (1, 4, 7),
    (2, 5, 8),  # cols
    (0, 4, 8),
    (2, 4, 6),  # diagonals
]


class TicTacToe:
    def __init__(self) -> None:
        self.board: list[str | None] = [None] * 9
        self.current: str = "X"

    def reset(self) -> None:
        self.board = [None] * 9
        self.current = "X"

    def available(self) -> list[int]:
        return [i for i, v in enumerate(self.board) if v is None]

    def winner(self) -> str | None:
        for a, b, c in _WINS:
            if self.board[a] and self.board[a] == self.board[b] == self.board[c]:
                return self.board[a]
        return None

    def over(self) -> bool:
        return self.winner() is not None or not self.available()

    def apply(self, pos: int, mark: str) -> None:
        self.board[pos] = mark
        self.current = "O" if mark == "X" else "X"

    def render(self) -> str:
        def cell(i: int) -> str:
            return self.board[i] or str(i)

        rows = [
            f" {cell(0)} | {cell(1)} | {cell(2)} ",
            "---+---+---",
            f" {cell(3)} | {cell(4)} | {cell(5)} ",
            "---+---+---",
            f" {cell(6)} | {cell(7)} | {cell(8)} ",
        ]
        w = self.winner()
        if w:
            rows.append(f"Winner: {w}")
        elif not self.available():
            rows.append("Draw")
        else:
            rows.append(f"Current player: {self.current}  |  Available: {self.available()}")
        return "\n".join(rows)


game = TicTacToe()

# ── MCP server ─────────────────────────────────────────────────────────────────


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


_PORT = _free_port()
server = FastMCP(name="tictactoe")


async def _inner_move(inner_mark: str) -> int:
    """Ask the inner model to pick a move. Falls back to first available.

    Also captures token-level training data (prompt_token_ids, token_ids,
    logprobs) into _inner_samples so the training loop can train on both
    sides of each game with a flipped reward.
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
                    "content": (
                        f"You are playing tic-tac-toe as {inner_mark}. "
                        "Reply with ONLY a single integer from the list of available positions."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Board:\n{game.render()}\n\n"
                        f"Available positions: {available}\n"
                        "Your move (integer only):"
                    ),
                },
            ],
            max_tokens=8,
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
        text = choice.message.content or ""
        nums = re.findall(r"\d+", text)
        if nums:
            pos = int(nums[0])
            if pos in available:
                return pos
    except Exception:
        pass

    return available[0]


@server.tool
async def make_move(position: int) -> str:
    """Place your mark at position 0–8, then the inner model responds.

    Positions:
      0 | 1 | 2
      3 | 4 | 5
      6 | 7 | 8

    Returns the board after both moves. Keep calling until you see "Winner" or "Draw".
    """
    if game.over():
        return f"Game is already over.\n{game.render()}"

    outer_mark = _OUTER_MARK
    inner_mark = "O" if outer_mark == "X" else "X"

    if game.current != outer_mark:
        return f"It's {game.current}'s turn (inner model), not yours. Board:\n{game.render()}"

    if position not in game.available():
        return f"Position {position} is taken. Available: {game.available()}\n{game.render()}"

    game.apply(position, outer_mark)
    if game.over():
        return game.render()

    pos = await _inner_move(inner_mark)
    game.apply(pos, inner_mark)

    return game.render()


@server.tool
def get_state() -> str:
    """Return the current board, whose turn it is, and available positions."""
    return game.render()


# ── environment ────────────────────────────────────────────────────────────────

env = Environment(name="tictactoe-selfplay")
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
    """Self-play game. seed % 2 decides starting order: even → outer is X, odd → outer is O."""
    global _INNER_MODEL, _OUTER_MARK, _inner_samples
    _INNER_MODEL = model
    _OUTER_MARK = "X" if seed % 2 == 0 else "O"
    inner_mark = "O" if _OUTER_MARK == "X" else "X"

    game.reset()
    _inner_samples = []  # fresh per game

    # If the inner model goes first (outer is O), let it make the opening move now.
    if _OUTER_MARK == "O":
        opening = await _inner_move("X")
        game.apply(opening, "X")

    yield (
        f"You are playing tic-tac-toe as {_OUTER_MARK} against {model} playing {inner_mark}.\n"
        f"{'You go first.' if _OUTER_MARK == 'X' else 'The opponent opened — it is now your turn.'}\n"
        "Call make_move(position) with a position 0–8 for each of your turns.\n"
        "After your move, the opponent responds automatically.\n\n"
        "Positions:\n  0 | 1 | 2\n  3 | 4 | 5\n  6 | 7 | 8\n\n"
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


tasks = [play_self(model="ttt-selfplay-389d2c", seed=s) for s in range(2)]
