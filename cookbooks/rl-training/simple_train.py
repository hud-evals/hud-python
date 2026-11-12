"""Simple on-policy RL: roll out a taskset, train with a built-in loss, repeat.

Each step runs the taskset under one long-lived job, takes the batch of fresh
runs, and trains on it: one ``forward_backward`` with a server-side loss
(importance sampling here), then one ``optim_step``, which checkpoints and
promotes the new weights behind the *same* model string, so the next rollout
samples the updated policy.

Traces are passed directly: ``TrainingClient`` reads each trace's samples and
reward. (Pass ``trace.trace_id`` strings instead to train on trajectories the
platform already holds.)

    uv run simple_train.py --steps 10   # set MODEL below (pick one with `hud models list`)
"""

from __future__ import annotations

import argparse
import asyncio
import time

from dotenv import load_dotenv

from common import load_taskset_and_runtime
from hud import Job
from hud.agents import create_agent
from hud.agents.types import AgentStep
from hud.train import TrainingClient

# The trainable gateway model to sample from and train, in place.
# Pick one with `hud models list` and paste its id here.
MODEL = "Qwen3 4B Instruct 2507 (Tinker)"


def _output_tokens(traces: list) -> int:
    """Total generated tokens across a batch of runs (a throughput numerator)."""
    return sum(
        len(sample.output_token_ids)
        for trace in traces
        for sample in trace.collect(
            lambda s: s.sample if isinstance(s, AgentStep) and s.sample else None
        )
    )


async def main(*, steps: int, group: int, learning_rate: float, max_concurrent: int) -> None:
    model = MODEL

    # return_token_ids tells the gateway/agent this is a training rollout: the
    # response carries token ids + per-token logprobs, which the agent records on
    # each turn's trace Sample — the token-level data TrainingClient trains on.
    # Leave room for scratch work: the 3-digit x 2-digit task is meant to be right
    # only sometimes, which is the GRPO signal.
    agent = create_agent(
        model,
        completion_kwargs={"max_tokens": 1024, "extra_body": {"return_token_ids": True}},
    )
    trainer = TrainingClient(model)
    # The deployed TASKSET (see common.py) against HUD-hosted envs, or the bundled local env.
    taskset, runtime = load_taskset_and_runtime()

    # One job spans the whole session; each iteration appends its batch of runs.
    session = await Job.start("arith-rl-simple", group=group)
    for step in range(steps):
        batch_start = len(session.traces)

        t0 = time.perf_counter()
        await taskset.run(agent, runtime=runtime, job=session, max_concurrent=max_concurrent)
        rollout_s = time.perf_counter() - t0
        batch = session.traces[batch_start:]
        tokens = _output_tokens(batch)

        # trainer.step would run both; they are split here to time each.
        t1 = time.perf_counter()
        fb = await trainer.forward_backward(
            batch,
            loss_fn="importance_sampling",
            group_size=group,  # each task's `group` repeats form one GRPO group
        )
        result = await trainer.optim_step(learning_rate=learning_rate)
        train_s = time.perf_counter() - t1

        mean_reward = sum(trace.reward for trace in batch) / len(batch)
        solved = sum(1 for trace in batch if trace.reward > 0)
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
