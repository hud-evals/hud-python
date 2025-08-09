"""Evaluator for highest tile."""

from fastmcp import Context
from hud.tools.types import EvaluationResult
from . import evaluate


@evaluate.tool("max_number")
async def evaluate_max_number(ctx: Context, target: int | None = None):
    game = evaluate.env
    highest_tile = game.get_state().get("highest_tile", 0)
    if target is None:
        reward = highest_tile
        done = True
    else:
        reward = min(1.0, highest_tile / target)
        done = highest_tile >= target
    return EvaluationResult(
        reward=reward,
        done=done,
        content=f"Target: {target}, Highest: {highest_tile}",
        info={"target": target, "highest_tile": highest_tile},
    )
