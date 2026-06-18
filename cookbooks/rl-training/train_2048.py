"""On-policy RL on a multi-turn game: train a model to play 2048.

Same five-line loop as ``simple_train.py`` — the training code is identical; only
the rollout source changes. Here each rollout is a whole *game*: the agent calls
the ``move`` tool many times (up to ``--moves`` turns), so one run is a multi-turn
trajectory and every turn contributes token-level data to ``forward_backward``.

GRPO grouping: 2048 spawns tiles randomly, so the ``group`` replays of the single
play task are genuinely different games — their reward spread (how high a tile
each reached) is the advantage signal.

    HUD_MODEL=<trainable-gateway-model> uv run train_2048.py --steps 15 --moves 12
"""

from __future__ import annotations

import argparse
import asyncio
import math
import os
import time

from dotenv import load_dotenv

from hud import TrainingClient
from hud.agents import create_agent
from hud.agents.types import AgentStep
from hud.eval import Job, LocalRuntime, Taskset

from game2048_env import play


def _output_tokens(runs: list) -> int:
    return sum(
        len(sample.output_token_ids)
        for run in runs
        for sample in run.trace.collect(
            lambda s: s.sample if isinstance(s, AgentStep) and s.sample else None
        )
    )


async def main(
    *,
    steps: int,
    group: int,
    moves: int,
    target: int,
    learning_rate: float,
    max_concurrent: int,
    rollout_timeout: float,
) -> None:
    model = os.environ["HUD_MODEL"]

    # max_steps caps the moves per game; return_token_ids records the per-turn
    # token-level Samples that TrainingClient trains on.
    agent = create_agent(
        model,
        max_steps=moves,
        completion_kwargs={"max_tokens": 512, "extra_body": {"return_token_ids": True}},
    )
    trainer = TrainingClient(model)
    # One play task; the job's `group` replays it into a GRPO group of games.
    taskset = Taskset("2048", [play(target=target)])
    runtime = LocalRuntime("game2048_env.py")

    session = await Job.start("game2048-rl", group=group)
    for step in range(steps):
        batch_start = len(session.runs)

        t0 = time.perf_counter()
        await taskset.run(
            agent,
            runtime=runtime,
            job=session,
            max_concurrent=max_concurrent,
            rollout_timeout=rollout_timeout,  # a wedged game can't stall the batch
        )
        rollout_s = time.perf_counter() - t0
        batch = session.runs[batch_start:]
        tokens = _output_tokens(batch)
        failed = sum(1 for run in batch if run.trace.status == "error")

        t1 = time.perf_counter()
        fb = await trainer.forward_backward(
            batch,
            loss_fn="importance_sampling",
            group_size=group,
        )
        result = await trainer.optim_step(learning_rate=learning_rate)
        train_s = time.perf_counter() - t1

        rewards = [run.reward for run in batch]
        mean_reward = sum(rewards) / len(rewards)
        # The grade boundary returns only the score, so invert the env's reward
        # (normalized log2 tile progress) to recover the best tile reached.
        best_tile = round(2 ** (max(rewards) * (math.log2(target) - 1) + 1))
        tok_per_s = tokens / rollout_s if rollout_s > 0 else 0.0
        loss = fb.metrics.get("loss:sum", float("nan"))
        print(
            f"step {step:2d} | reward {mean_reward:.3f} best_tile {best_tile:4d} "
            f"| rollout {rollout_s:5.1f}s {tokens:6d}tok {tok_per_s:4.0f}tok/s "
            f"| train {train_s:5.1f}s loss {loss:+.4f} "
            f"| optim {result.step} datums {fb.num_datums} failed {failed}/{len(batch)}",
            flush=True,
        )


if __name__ == "__main__":
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=15)
    parser.add_argument("--group", type=int, default=8, help="games per step (GRPO group)")
    parser.add_argument("--moves", type=int, default=12, help="max moves (turns) per game")
    parser.add_argument("--target", type=int, default=256, help="win tile (reward scale)")
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--max-concurrent", type=int, default=8)
    parser.add_argument(
        "--timeout", type=float, default=300.0, help="per-game wall-clock cap (s)"
    )
    args = parser.parse_args()
    asyncio.run(
        main(
            steps=args.steps,
            group=args.group,
            moves=args.moves,
            target=args.target,
            learning_rate=args.learning_rate,
            max_concurrent=args.max_concurrent,
            rollout_timeout=args.timeout,
        )
    )
