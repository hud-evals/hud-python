"""Validation eval: trained c4-selfplay vs a fixed opponent.

Self-play reward is always ~0.5 by construction, so it can't tell you whether
the model is actually getting stronger. This script fixes the opponent to a
static model and measures win rate — if training is working, win rate vs the
same opponent should climb over time.

Default opponent: Qwen3 (separate Tinker-hosted model, never changes with RL steps).
You can swap in any gateway model slug with --opponent.

Usage:
    uv run python cookbooks/connect4-selfplay/validate.py
    uv run python cookbooks/connect4-selfplay/validate.py --games 20
    uv run python cookbooks/connect4-selfplay/validate.py --opponent Qwen3 --games 10
"""

from __future__ import annotations

import argparse
import asyncio

TRAINED_MODEL = "c4-selfplay"
DEFAULT_OPPONENT = "Qwen/Qwen3.5-4B"
DEFAULT_GAMES = 10  # even number keeps X/O starts balanced


async def run_validation(trained: str, opponent: str, games: int) -> None:
    from hud.agents import create_agent
    from hud.eval import Taskset

    from env import play_self

    # Alternate who goes first: even seeds → trained is X (first), odd → trained is O.
    tasks = [play_self(model=opponent, seed=s) for s in range(games)]
    taskset = Taskset("c4-validate", tasks)
    agent = create_agent(trained)

    print(f"Validation: {trained} (outer) vs {opponent} (inner fixed)")
    print(f"Games: {games}  ({games // 2} as X, {games - games // 2} as O)\n")

    job = await taskset.run(agent)

    wins = sum(1 for r in job.runs if r.reward == 1.0)
    losses = sum(1 for r in job.runs if r.reward == 0.0)
    draws = sum(1 for r in job.runs if r.reward == 0.5)
    mean = job.reward

    print(f"Result vs {opponent}:")
    print(f"  {wins}W / {draws}D / {losses}L  (mean reward: {mean:.3f})")
    rate = wins / games if games else 0.0
    verdict = "above baseline" if rate > 0.5 else "at or below baseline"
    print(f"  Win rate: {rate:.1%}  ({verdict})")

    # Per-game breakdown so you can spot which seeds the model struggles on.
    print("\nPer-game:")
    for i, run in enumerate(job.runs):
        side = "X" if i % 2 == 0 else "O"
        outcome = {1.0: "WIN ", 0.5: "DRAW", 0.0: "LOSS"}.get(run.reward, f"{run.reward:.2f}")
        print(f"  game {i + 1:2d} (outer={side}): {outcome}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate c4-selfplay against a fixed opponent")
    parser.add_argument("--trained", default=TRAINED_MODEL, help="Trained model slug")
    parser.add_argument("--opponent", default=DEFAULT_OPPONENT, help="Fixed opponent model slug")
    parser.add_argument("--games", type=int, default=DEFAULT_GAMES, help="Number of games")
    args = parser.parse_args()

    if args.games % 2 != 0:
        print("Warning: odd number of games — outer won't have equal X/O starts")

    asyncio.run(run_validation(args.trained, args.opponent, args.games))


if __name__ == "__main__":
    main()
