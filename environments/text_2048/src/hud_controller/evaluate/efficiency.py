"""Evaluator for move efficiency."""

from fastmcp import Context
from hud.tools.types import EvaluationResult
from . import evaluate


@evaluate.tool(name="efficiency", description="Evaluate game efficiency based on score/moves ratio")
async def evaluate_efficiency(ctx: Context, min_ratio: float):
    game = evaluate.env
    state = game.get_state()
    score = state.get("score", 0)
    moves = state.get("moves", 0)

    # Calculate the efficiency ratio directly (no logarithmic scaling)
    ratio = score / moves if moves > 0 else 0.0

    # Linear reward: proportional to ratio / min_ratio, capped at 1.0
    reward = min(1.0, ratio / min_ratio) if min_ratio > 0 else 0.0
    done = ratio >= min_ratio

    return EvaluationResult(
        reward=reward,
        done=done,
        content=f"Efficiency: {ratio:.2f} (target: {min_ratio})",
        info={"score": score, "moves": moves, "ratio": ratio},
    )
