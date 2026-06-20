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
    python train.py --model c4-selfplay --steps 20 --lr 3e-6 --validate-every 5
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import time

from hud import Run, TrainingClient
from hud.agents import create_agent
from hud.eval import Job, Taskset
from hud.train import TrajectoryPayload, TrajectorySample
from hud.utils.exceptions import HudNetworkError, HudTimeoutError

from env import play_self


logger = logging.getLogger(__name__)


async def _optim_step_with_retry(trainer: TrainingClient, *, lr: float, retries: int = 3) -> any:
    """Retry optim_step only on network/timeout errors.

    optim_step is state-mutating (saves a checkpoint). Retrying on server-side
    errors (5xx) can cause duplicate checkpoint saves, which poisons the Tinker
    training run. Only network-level failures (connection drop, timeout) are safe
    to retry.
    """
    for attempt in range(1, retries + 1):
        try:
            return await trainer.optim_step(learning_rate=lr)
        except (HudNetworkError, HudTimeoutError) as exc:
            if attempt == retries:
                raise
            wait = 2**attempt
            logger.warning(
                "optim_step network error (attempt %d/%d): %s — retrying in %ds",
                attempt,
                retries,
                exc,
                wait,
            )
            await asyncio.sleep(wait)


def make_tasks(model: str) -> Taskset:
    # 8 seeds: even → outer=X, odd → outer=O (symmetric first-move coverage)
    return Taskset("c4-self-play", [play_self(model=model, seed=i) for i in range(8)])


async def _run_validation(trained: str, opponent: str, games: int, step: int) -> float:
    """Run a quick validation match and return mean reward (win rate proxy)."""
    val_agent = create_agent(trained, max_steps=30)
    val_tasks = Taskset("c4-validate", [play_self(model=opponent, seed=s) for s in range(games)])
    job = await val_tasks.run(val_agent)
    wins = sum(1 for r in job.runs if r.reward == 1.0)
    draws = sum(1 for r in job.runs if r.reward == 0.5)
    losses = sum(1 for r in job.runs if r.reward == 0.0)
    mean = job.reward
    x_wins = sum(1 for i, r in enumerate(job.runs) if r.reward == 1.0 and i % 2 == 0)
    o_wins = sum(1 for i, r in enumerate(job.runs) if r.reward == 1.0 and i % 2 != 0)
    print(
        f"  [val@{step}] vs {opponent}: {wins}W/{draws}D/{losses}L  "
        f"mean={mean:.3f}  X-wins={x_wins}/{games // 2}  O-wins={o_wins}/{games - games // 2}"
    )
    return mean


async def main(
    model: str,
    steps: int,
    group: int,
    lr: float,
    validate_every: int,
    val_opponent: str,
    val_games: int,
) -> None:
    # Keep reasoning short so the agent doesn't burn its step budget on analysis.
    # Each turn: one sentence of reasoning, then call make_move immediately.
    system_prompt = (
        "You are playing Connect Four. Think in ONE sentence, then immediately call make_move. "
        "Do not write long analysis. Just pick the best column and call the tool."
    )
    # return_token_ids: gateway returns token ids + per-token logprobs for training
    agent = create_agent(
        model,
        max_steps=30,
        system_prompt=system_prompt,
        completion_kwargs={"extra_body": {"return_token_ids": True}},
    )
    trainer = TrainingClient(model)
    tasks = make_tasks(model)
    session = await Job.start("c4-selfplay", group=group)

    val_curve: list[tuple[int, float]] = []

    for step in range(steps):
        t_step = time.perf_counter()

        batch_start = len(session.runs)
        t0 = time.perf_counter()
        await tasks.run(agent, job=session)
        t_rollout = time.perf_counter() - t0
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

        if effective_group is None:
            print(
                f"  warning: {len(batch) - inner_count}/{len(batch)} games missing inner samples — group_size=None"
            )
        print(
            f"  rollout done: {t_rollout:.1f}s for {len(batch)} games ({len(combined)} trajectories)"
        )

        t0 = time.perf_counter()
        await trainer.forward_backward(combined, loss_fn="ppo", group_size=effective_group)
        t_fwdbwd = time.perf_counter() - t0

        t0 = time.perf_counter()
        result = await _optim_step_with_retry(trainer, lr=lr)
        t_optim = time.perf_counter() - t0

        rewards = [r.reward for r in batch]
        mean_r = sum(rewards) / len(rewards) if rewards else float("nan")
        reward_std = (
            (sum((r - mean_r) ** 2 for r in rewards) / len(rewards)) ** 0.5 if rewards else 0.0
        )
        wins = sum(1 for r in rewards if r == 1.0)
        draws = sum(1 for r in rewards if r == 0.5)
        losses = sum(1 for r in rewards if r == 0.0)
        print(
            f"step {step + 1}/{steps}  "
            f"mean={mean_r:.3f}  std={reward_std:.3f}  "
            f"outer-wins={wins}  draws={draws}  outer-losses={losses}  "
            f"inner-trajectories={inner_count}/{len(batch)}"
        )
        t_total = time.perf_counter() - t_step
        print(f"  -> checkpoint {result.step}  sampler={result.sampler_path}")
        print(
            f"  timing: rollout={t_rollout:.1f}s  fwd_bwd={t_fwdbwd:.1f}s  optim={t_optim:.1f}s  total={t_total:.1f}s"
        )

        if validate_every > 0 and (step + 1) % validate_every == 0:
            mean_val = await _run_validation(model, val_opponent, val_games, step + 1)
            val_curve.append((step + 1, mean_val))

    # Server-side view of the run (reward spread etc.), via the checkpoints API.
    print("\nlast checkpoints (server metrics):")
    for c in (await trainer.checkpoints())[-min(steps, 5) :]:
        std = c.metrics.get("reward_std")
        print(f"  {c.name}  reward={c.mean_reward}  std={std}  loss={c.loss_fn}")

    if val_curve:
        print("\nvalidation curve (step → mean reward vs opponent):")
        for s, r in val_curve:
            bar = "#" * int(r * 20)
            print(f"  step {s:3d}: {r:.3f}  {bar}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="c4-selfplay", help="trainable model slug")
    parser.add_argument("--steps", type=int, default=20, help="optimizer steps")
    parser.add_argument("--group", type=int, default=4, help="GRPO group size (rollouts per task)")
    parser.add_argument("--lr", type=float, default=1e-5, help="learning rate")
    parser.add_argument(
        "--validate-every",
        type=int,
        default=5,
        metavar="N",
        help="run validation vs opponent every N steps (0 = off)",
    )
    parser.add_argument(
        "--val-opponent", default="Qwen/Qwen3.5-4B", help="fixed opponent for validation"
    )
    parser.add_argument(
        "--val-games", type=int, default=10, help="games per validation run (even = balanced X/O)"
    )
    args = parser.parse_args()
    asyncio.run(
        main(
            args.model,
            args.steps,
            args.group,
            args.lr,
            args.validate_every,
            args.val_opponent,
            args.val_games,
        )
    )
