"""Simple on-policy RL: roll out a taskset, train with a built-in loss, repeat.

The whole loop is five lines: run the taskset under one long-lived job, take the
batch of fresh runs, and hand them to ``trainer.step``. ``step`` does one
``forward_backward`` with a server-side loss (importance sampling here) followed
by one ``optim_step`` — which checkpoints and promotes the new weights behind the
*same* model string, so the next rollout samples the updated policy.

Runs are passed directly: ``TrainingClient`` reads each ``Run``'s trajectory and
reward. (Pass ``run.trace_id`` strings instead to train on trajectories the
platform already holds.)

    HUD_MODEL=<trainable-gateway-model> uv run simple_train.py --steps 10
"""

from __future__ import annotations

import argparse
import asyncio
import os
import time

from dotenv import load_dotenv

from common import load_taskset_and_runtime
from hud import TrainingClient
from hud.agents import create_agent
from hud.agents.types import AgentStep
from hud.eval import Job


def _output_tokens(runs: list) -> int:
    """Total generated tokens across a batch of runs (a throughput numerator)."""
    return sum(
        len(sample.output_token_ids)
        for run in runs
        for sample in run.trace.collect(
            lambda s: s.sample if isinstance(s, AgentStep) and s.sample else None
        )
    )


async def main(*, steps: int, group: int, learning_rate: float, max_concurrent: int) -> None:
    model = os.environ["HUD_MODEL"]  # a trainable gateway model string

    # return_token_ids tells the gateway/agent this is a training rollout: the
    # response carries token ids + per-token logprobs, which the agent records on
    # each turn's trace Sample — the token-level data TrainingClient trains on.
    # Allow room for chain-of-thought: this is a reasoning model, and the task
    # (3-digit x 2-digit) needs scratch work — it just has to be hard enough to be
    # right only sometimes (the GRPO signal).
    agent = create_agent(
        model,
        completion_kwargs={"max_tokens": 1024, "extra_body": {"return_token_ids": True}},
    )
    trainer = TrainingClient(model)
    # A deployed taskset on remote HUD boxes (HUD_TASKSET), or the local env.
    taskset, runtime = load_taskset_and_runtime()

    # One job spans the whole session; each iteration appends its batch of runs.
    session = await Job.start("arith-rl-simple", group=group)
    for step in range(steps):
        batch_start = len(session.runs)

        # --- rollout phase (sampling throughput) ---
        t0 = time.perf_counter()
        await taskset.run(agent, runtime=runtime, job=session, max_concurrent=max_concurrent)
        rollout_s = time.perf_counter() - t0
        batch = session.runs[batch_start:]
        tokens = _output_tokens(batch)

        # --- train phase (forward_backward + optim_step, split out for metrics) ---
        t1 = time.perf_counter()
        fb = await trainer.forward_backward(
            batch,
            loss_fn="importance_sampling",
            group_size=group,  # each task's `group` repeats form one GRPO group
        )
        result = await trainer.optim_step(learning_rate=learning_rate)
        train_s = time.perf_counter() - t1

        mean_reward = sum(run.reward for run in batch) / len(batch)
        solved = sum(1 for run in batch if run.reward > 0)
        tok_per_s = tokens / rollout_s if rollout_s > 0 else 0.0
        loss = fb.metrics.get("loss:sum", float("nan"))
        print(
            f"step {step:2d} | reward {mean_reward:.3f} ({solved}/{len(batch)}) "
            f"| rollout {rollout_s:5.1f}s {tokens:5d}tok {tok_per_s:4.0f}tok/s "
            f"| train {train_s:5.1f}s loss {loss:+.4f} "
            f"| optim {result.step} datums {fb.num_datums}",
            flush=True,
        )


if __name__ == "__main__":
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--group", type=int, default=8, help="rollouts per task (GRPO group)")
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument(
        "--max-concurrent", type=int, default=8, help="cap on simultaneous rollouts"
    )
    args = parser.parse_args()
    asyncio.run(
        main(
            steps=args.steps,
            group=args.group,
            learning_rate=args.learning_rate,
            max_concurrent=args.max_concurrent,
        )
    )
