"""Self-play tic-tac-toe training loop.

Each step runs 8 games (outer=X for seeds 0,2,4,6 and outer=O for seeds 1,3,5,7)
then trains on BOTH sides of every game simultaneously:

  - Outer agent trajectory: reward = game outcome from outer's perspective
  - Inner model trajectory: reward = 1 - outer_reward (symmetric flip)

Both are included in a single forward-backward call with PPO loss (epsilon=0.2),
which clips the IS ratio and prevents destructive updates from a single hot game.

Setup:
    hud models fork Qwen/Qwen3.5-4B --name ttt-selfplay

Run:
    HUD_RL_URL=http://localhost:8003 python train.py --model ttt-selfplay-389d2c
"""

from __future__ import annotations

import argparse
import asyncio

from hud import TrainingClient
from hud.agents import create_agent
from hud.eval import Job, Taskset
from hud.train.client import _run_to_input
from hud.train.types import ForwardBackwardRequest, TrajectoryPayload, TrajectorySample

from env import play_self


def make_tasks(model: str) -> Taskset:
    # 8 seeds: even seeds → outer=X, odd seeds → outer=O (symmetric coverage)
    return Taskset("ttt-self-play", [play_self(model=model, seed=i) for i in range(8)])


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

        # --- Build combined inputs: one outer + one inner payload per game ---
        # Outer trajectory: run's token trace, reward from outer's perspective.
        # Inner trajectory: inner model tokens captured in env, reward flipped.
        combined: list[str | TrajectoryPayload] = []
        inner_count = 0

        for run in batch:
            combined.append(_run_to_input(run))

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
                # Symmetric reward: inner model wins what outer loses
                combined.append(TrajectoryPayload(
                    samples=inner_turns,
                    reward=1.0 - run.reward,
                ))

        # group_size=2 pairs each outer with its inner (symmetric GRPO advantage:
        # advantage = reward - mean([r_outer, r_inner]) = r_outer - 0.5 per game).
        # If no inner samples were captured, group_size=None puts all in one group.
        effective_group = 2 if inner_count == len(batch) else None

        fb_req = ForwardBackwardRequest(
            inputs=combined,
            loss_fn="ppo",
            # Tinker's deployed PPOLoss rejects an `epsilon` kwarg (the SDK
            # docstring's `{"epsilon": 0.2}` example is stale); use PPO defaults.
            group_size=effective_group,
        )
        await trainer._post("forward-backward", fb_req.model_dump())
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="ttt-selfplay-389d2c", help="trainable model slug")
    parser.add_argument("--steps", type=int, default=20, help="optimizer steps")
    parser.add_argument("--group", type=int, default=8, help="GRPO group size (rollouts per task)")
    parser.add_argument("--lr", type=float, default=1e-5, help="learning rate")
    args = parser.parse_args()
    asyncio.run(main(args.model, args.steps, args.group, args.lr))
