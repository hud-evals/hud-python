"""On-policy RL with a custom, client-side loss: GLM-5.2 double-sided IS.

Same loop as ``simple_train.py``, but instead of a built-in ``loss_fn`` the
policy-gradient loss is written here in torch and run client-side via
``trainer.forward_backward_custom``. The service runs the current-policy forward
pass and returns per-token tensors (:class:`DatumTensors`); this loss turns them
into per-token gradients, which the service applies. ``optim_step`` then
checkpoints and promotes as usual.

The loss is GLM-5.2's *direct double-sided importance sampling* (see the README):
reuse the rollout logprobs as the behavior proxy, form the token ratio
``r = exp(logπ_θ − logπ_rollout)``, **hard-mask** tokens whose ratio leaves the
trust region (zero gradient, not clipped), and normalize at the token level so
long and short trajectories contribute evenly.

    uv run ppo_custom_loss.py --steps 10   # set MODEL below (pick one with `hud models`)

Requires torch (declared in this cookbook's pyproject; in the SDK it is the
``hud-python[train]`` extra).
"""

from __future__ import annotations

import argparse
import asyncio

import torch
from dotenv import load_dotenv

from common import load_taskset_and_runtime
from hud import TrainingClient
from hud.agents import create_agent
from hud.eval import Job
from hud.train import DatumTensors

# The trainable gateway model to sample from and train, in place.
# Pick one with `hud models` and paste its id here.
MODEL = "<trainable-model>"


def glm_double_sided_is(
    data: list[DatumTensors],
    logprobs: list[torch.Tensor],
    *,
    eps_low: float = 0.2,
    eps_high: float = 0.28,
) -> tuple[torch.Tensor, dict[str, float]]:
    """GLM-5.2 direct double-sided importance-sampling policy-gradient loss.

    ``logprobs[i]`` are the current-policy (π_θ) per-token logprobs for datum
    ``i`` as differentiable leaves — build the loss out of *these* tensors so the
    gradient flows back. Everything else (rollout logprobs q, action mask,
    reward) is a constant carried on the matching :class:`DatumTensors`.
    """
    # Per-group (GRPO) baseline: advantage = reward − group mean. GLM-5.2 uses a
    # learned critic here (README, Option A); the group baseline is the
    # critic-free stand-in so the focus stays on the IS ratio / mask / norm.
    # ``group_idx`` is None when training ungrouped — then all datums share one
    # baseline (a single-rollout / batch-mean formulation).
    group_rewards: dict[int | None, list[float]] = {}
    for datum in data:
        group_rewards.setdefault(datum.group_idx, []).append(datum.reward)
    baseline = {group: sum(rs) / len(rs) for group, rs in group_rewards.items()}

    total = logprobs[0].new_zeros(())
    trained_tokens = logprobs[0].new_zeros(())
    masked_tokens = 0.0

    for datum, policy_logprobs in zip(data, logprobs, strict=True):
        rollout_logprobs = torch.tensor(datum.sampling_logprobs, dtype=torch.float32)
        action_mask = torch.tensor(datum.mask, dtype=torch.float32)
        advantage = datum.reward - baseline[datum.group_idx]

        # r_t = π_θ / π_rollout, reusing rollout logprobs as the behavior proxy
        # (no separate π_old pass — GLM's simplification).
        ratio = torch.exp(policy_logprobs - rollout_logprobs)

        # Double-sided HARD mask: tokens outside [1 − eps_low, 1 + eps_high] get
        # zero gradient. The comparison is non-differentiable, so this is a true
        # mask (set to zero), not PPO's clip.
        in_region = (ratio >= 1.0 - eps_low) & (ratio <= 1.0 + eps_high)
        token_mask = action_mask * in_region.float()

        total = total - (ratio * advantage * token_mask).sum()
        trained_tokens = trained_tokens + token_mask.sum()
        masked_tokens += float((action_mask.sum() - token_mask.sum()).item())

    # Token-level normalization: divide by the number of trained tokens, not by
    # trajectory count, so length imbalance does not skew the update.
    loss = total / trained_tokens.clamp_min(1.0)
    return loss, {
        "trained_tokens": float(trained_tokens.item()),
        "masked_tokens": masked_tokens,
    }


async def main(*, steps: int, group: int, learning_rate: float, max_concurrent: int) -> None:
    model = MODEL  # the trainable gateway model (set at the top of this file)

    # Training rollout: capture token ids + logprobs onto each turn's Sample;
    # room for chain-of-thought (the task needs scratch work).
    agent = create_agent(
        model,
        completion_kwargs={"max_tokens": 1024, "extra_body": {"return_token_ids": True}},
    )
    trainer = TrainingClient(model)
    # A deployed taskset on remote HUD boxes (HUD_TASKSET), or the local env.
    taskset, runtime = load_taskset_and_runtime()

    session = await Job.start("arith-rl-ppo", group=group)
    for step in range(steps):
        batch_start = len(session.runs)
        await taskset.run(agent, runtime=runtime, job=session, max_concurrent=max_concurrent)
        batch = session.runs[batch_start:]

        # forward (server) -> glm loss (here, torch) -> backward (server)
        fb = await trainer.forward_backward_custom(batch, glm_double_sided_is, group_size=group)
        result = await trainer.optim_step(learning_rate=learning_rate)

        mean_reward = sum(run.reward for run in batch) / len(batch)
        print(
            f"step {step}: mean_reward={mean_reward:.3f} "
            f"masked_tokens={fb.metrics.get('masked_tokens', 0.0):.0f} "
            f"optim_step={result.step} -> {result.sampler_path}",
            flush=True,
        )


if __name__ == "__main__":
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--group", type=int, default=8, help="rollouts per task (GRPO group)")
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--max-concurrent", type=int, default=8)
    args = parser.parse_args()
    asyncio.run(
        main(
            steps=args.steps,
            group=args.group,
            learning_rate=args.learning_rate,
            max_concurrent=args.max_concurrent,
        )
    )
