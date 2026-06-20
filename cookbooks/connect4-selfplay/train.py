"""Self-play Connect Four training loop.

Each step runs a batch of games (even seeds → outer=X, odd → outer=O for
symmetric coverage) then trains on BOTH sides of every game at once:

  - Outer agent trajectory: reward = game outcome from outer's perspective
  - Inner model trajectory: reward = 1 - outer_reward (symmetric flip)

Both go into a single forward_backward call with PPO loss (clips the IS ratio,
so one lucky game can't blow up the update), grouped in pairs so the GRPO
advantage is computed within each game.

This uses the public training API directly: forward_backward accepts Run and
TrajectoryPayload mixed, so no private helpers are needed.

Setup:
    hud models fork Qwen/Qwen3.5-4B --name c4-selfplay

Run:
    python train.py --model c4-selfplay-<id>
"""

from __future__ import annotations

import argparse
import asyncio

from hud import Run, TrainingClient
from hud.agents import create_agent
from hud.eval import Job, Taskset
from hud.train import TrajectoryPayload, TrajectorySample

from env import play_self


def make_tasks(model: str) -> Taskset:
    # 8 seeds: even → outer=X, odd → outer=O (symmetric first-move coverage)
    return Taskset("c4-self-play", [play_self(model=model, seed=i) for i in range(8)])


async def main(model: str, steps: int, group: int, lr: float) -> None:
    # return_token_ids: gateway returns token ids + per-token logprobs for training
    agent = create_agent(
        model,
        completion_kwargs={"extra_body": {"return_token_ids": True}},
    )
    trainer = TrainingClient(model)
    tasks = make_tasks(model)
    session = await Job.start(model, group=group)

    for step in range(steps):
        batch_start = len(session.runs)
        await tasks.run(agent, job=session)
        batch = session.runs[batch_start:]

        # One outer + one inner trajectory per game. forward_backward takes the
        # Run directly (its tokens + reward); the inner side is a hand-built
        # TrajectoryPayload with the flipped reward.
        combined: list[Run | TrajectoryPayload] = []
        inner_count = 0

        for run in batch:
            combined.append(run)  # outer side, reward from outer's perspective

            inner_dicts = run.grade.info.get("inner_samples", [])
            inner_turns = [
                TrajectorySample(
                    prompt_token_ids=s["prompt_token_ids"],
                    output_token_ids=s["output_token_ids"],
                    output_logprobs=s.get("output_logprobs", []),
                )
                for s in inner_dicts
                if s.get("output_token_ids")
            ]
            if inner_turns:
                inner_count += 1
                combined.append(TrajectoryPayload(samples=inner_turns, reward=1.0 - run.reward))

        # group_size=2 pairs each outer with its inner (advantage = r_outer - 0.5
        # per game). If any game is missing its inner side the pairing breaks, so
        # fall back to a single group rather than a misaligned one.
        effective_group = 2 if inner_count == len(batch) else None

        await trainer.forward_backward(combined, loss_fn="ppo", group_size=effective_group)
        result = await trainer.optim_step(learning_rate=lr)

        rewards = [r.reward for r in batch]
        mean_r = sum(rewards) / len(rewards) if rewards else float("nan")
        wins = sum(1 for r in rewards if r == 1.0)
        draws = sum(1 for r in rewards if r == 0.5)
        losses = sum(1 for r in rewards if r == 0.0)
        print(
            f"step {step + 1}/{steps}  "
            f"mean={mean_r:.3f}  outer-wins={wins}  draws={draws}  outer-losses={losses}  "
            f"inner-trajectories={inner_count}/{len(batch)}"
        )
        print(f"  -> checkpoint {result.step}  sampler={result.sampler_path}")

    # Server-side view of the run (reward spread etc.), via the checkpoints API.
    print("\nlast checkpoints (server metrics):")
    for c in (await trainer.checkpoints())[-min(steps, 5) :]:
        std = c.metrics.get("reward_std")
        print(f"  {c.name}  reward={c.mean_reward}  std={std}  loss={c.loss_fn}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="c4-selfplay", help="trainable model slug")
    parser.add_argument("--steps", type=int, default=20, help="optimizer steps")
    parser.add_argument("--group", type=int, default=4, help="GRPO group size (rollouts per task)")
    parser.add_argument("--lr", type=float, default=1e-5, help="learning rate")
    args = parser.parse_args()
    asyncio.run(main(args.model, args.steps, args.group, args.lr))
