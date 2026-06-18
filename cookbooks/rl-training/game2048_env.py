"""A 2048 game as a multi-turn HUD environment the LLM plays move-by-move.

The board lives in a module-level ``Game2048`` (one env process per rollout, so
state is per-game). A FastMCP server exposes a single ``move(direction)`` tool;
the agent calls it each turn and sees the updated board. The task template yields
the opening prompt, lets the agent run its tool loop, then grades from the final
board (max tile reached) — it does not read the agent's text answer.

Multi-turn note: each ``move`` is one agent turn, so a rollout produces a
multi-turn trajectory; with ``return_token_ids`` every turn carries a token-level
``Sample``, which is exactly the trainable unit (``turns_to_trajectory`` builds a
multi-transition trajectory from it).

Run a single game with ``play_2048.py``, or serve standalone: ``hud serve game2048_env.py``.
"""

from __future__ import annotations

import asyncio
import math
import random
import socket
import time

from fastmcp import FastMCP

from hud.capabilities import Capability
from hud.environment import Environment
from hud.graders import EvaluationResult

_PORT = 8047
_SIZE = 4
_MOVES = {"up", "down", "left", "right"}


class Game2048:
    """Minimal 2048: 4x4 board, merge-on-move, random 2/4 spawns."""

    def __init__(self) -> None:
        self.board: list[list[int]] = [[0] * _SIZE for _ in range(_SIZE)]
        self.score = 0
        self.reset()

    def reset(self) -> None:
        self.board = [[0] * _SIZE for _ in range(_SIZE)]
        self.score = 0
        self._spawn()
        self._spawn()

    def _spawn(self) -> None:
        empty = [(r, c) for r in range(_SIZE) for c in range(_SIZE) if self.board[r][c] == 0]
        if empty:
            r, c = random.choice(empty)
            self.board[r][c] = 4 if random.random() < 0.1 else 2

    @staticmethod
    def _merge_left(row: list[int]) -> tuple[list[int], int]:
        """Collapse a single row to the left, returning (new_row, gained_score)."""
        tight = [v for v in row if v != 0]
        out: list[int] = []
        gained = 0
        i = 0
        while i < len(tight):
            if i + 1 < len(tight) and tight[i] == tight[i + 1]:
                merged = tight[i] * 2
                out.append(merged)
                gained += merged
                i += 2
            else:
                out.append(tight[i])
                i += 1
        out.extend([0] * (_SIZE - len(out)))
        return out, gained

    def _transform(self, direction: str) -> list[list[int]]:
        b = self.board
        if direction == "left":
            return [row[:] for row in b]
        if direction == "right":
            return [row[::-1] for row in b]
        if direction == "up":
            return [[b[r][c] for r in range(_SIZE)] for c in range(_SIZE)]
        # down
        return [[b[_SIZE - 1 - r][c] for r in range(_SIZE)] for c in range(_SIZE)]

    def _untransform(self, direction: str, grid: list[list[int]]) -> list[list[int]]:
        if direction == "left":
            return grid
        if direction == "right":
            return [row[::-1] for row in grid]
        if direction == "up":
            return [[grid[c][r] for c in range(_SIZE)] for r in range(_SIZE)]
        return [[grid[c][_SIZE - 1 - r] for c in range(_SIZE)] for r in range(_SIZE)]

    def move(self, direction: str) -> bool:
        """Apply a move; return True if the board changed (and a tile spawned)."""
        grid = self._transform(direction)
        moved = False
        new_grid: list[list[int]] = []
        for row in grid:
            new_row, gained = self._merge_left(row)
            self.score += gained
            if new_row != row:
                moved = True
            new_grid.append(new_row)
        if moved:
            self.board = self._untransform(direction, new_grid)
            self._spawn()
        return moved

    def max_tile(self) -> int:
        return max(max(row) for row in self.board)

    def game_over(self) -> bool:
        if any(0 in row for row in self.board):
            return False
        return not any(
            self._transform(d) != [r for r, _ in (self._merge_left(row) for row in self._transform(d))]
            for d in _MOVES
        )

    def render(self) -> str:
        width = max(len(str(self.max_tile())), 4)
        rows = [" ".join(f"{v or '.':>{width}}" for v in row) for row in self.board]
        return "\n".join(rows) + f"\nscore={self.score} max_tile={self.max_tile()}"


game = Game2048()
server = FastMCP(name="game2048")


@server.tool
def move(direction: str) -> str:
    """Slide the board: ``up``, ``down``, ``left``, or ``right``. Returns the board."""
    d = direction.strip().lower()
    if d not in _MOVES:
        return f"invalid direction {direction!r}; use one of up/down/left/right\n{game.render()}"
    changed = game.move(d)
    note = "" if changed else " (no tiles moved — try another direction)"
    over = "\nGAME OVER" if game.game_over() else ""
    return f"{game.render()}{note}{over}"


env = Environment(name="game2048")
_task: asyncio.Task[None] | None = None


async def _listening(host: str, port: int, timeout: float = 10.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), 0.2):
                return
        except OSError:
            await asyncio.sleep(0.1)
    raise RuntimeError(f"FastMCP server not listening on {host}:{port}")


@env.initialize
async def _up() -> None:
    global _task
    if _task is None:
        _task = asyncio.create_task(
            server.run_async(transport="http", host="127.0.0.1", port=_PORT)
        )
        await _listening("127.0.0.1", _PORT)
    env.add_capability(Capability.mcp(name="tools", url=f"http://127.0.0.1:{_PORT}/mcp"))


@env.shutdown
async def _down() -> None:
    global _task
    if _task is not None:
        _task.cancel()
        _task = None


@env.template()
async def play(target: int = 256):
    """Play one game; reward scales with the highest tile reached (target = win)."""
    game.reset()
    yield (
        "You are playing 2048 on a 4x4 grid. Each turn call the `move` tool with a "
        "direction (up/down/left/right) to slide and merge tiles. Keep playing to "
        f"build the largest tile you can (aim for {target}). The current board:\n\n"
        f"{game.render()}"
    )

    max_tile = game.max_tile()
    # Reward: normalized log2 progress from the start tile (2) to the target.
    reward = (math.log2(max_tile) - 1) / (math.log2(target) - 1)
    yield EvaluationResult(
        reward=max(0.0, min(1.0, reward)),
        content=str(max_tile),
        info={"max_tile": max_tile, "score": game.score, "target": target},
    )
