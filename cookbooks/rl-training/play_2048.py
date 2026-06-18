"""Validate the 2048 env end-to-end: one game, multi-turn, trainable traces.

Drives a single rollout of ``game2048_env.play`` with a tool-using agent (raised
max_steps so it can make many moves), then reports the outcome and — crucially —
how many turns carried token-level Samples. That proves a multi-turn game
produces the trainable trajectory the RL pipeline consumes.

    HUD_MODEL=<trainable-model> uv run play_2048.py --target 256 --max-steps 30
"""

from __future__ import annotations

import argparse
import asyncio
import os

from dotenv import load_dotenv

from hud.agents import create_agent
from hud.agents.types import AgentStep
from hud.eval import LocalRuntime

from game2048_env import play


async def main(*, target: int, max_steps: int) -> None:
    model = os.environ["HUD_MODEL"]
    agent = create_agent(
        model,
        max_steps=max_steps,
        completion_kwargs={"extra_body": {"return_token_ids": True}},
    )

    print(f"playing one game (target={target}, max_steps={max_steps})...", flush=True)
    job = await play(target=target).run(agent, runtime=LocalRuntime("game2048_env.py"))
    run = job.runs[0]

    samples = run.trace.collect(
        lambda s: s.sample if isinstance(s, AgentStep) and s.sample else None
    )
    trainable = [s for s in samples if s.output_token_ids]
    moves = sum(1 for step in run.trace.steps if isinstance(step, AgentStep) and step.tool_calls)
    print(f"reward={run.reward:.3f} status={run.trace.status}", flush=True)
    print(
        f"agent turns={len(samples)} (with tool calls={moves}) "
        f"trainable turns={len(trainable)} "
        f"tokens={sum(len(s.output_token_ids) for s in trainable)}"
    )
    print(f"final: {run.evaluation}")


if __name__ == "__main__":
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=int, default=256)
    parser.add_argument("--max-steps", type=int, default=30)
    args = parser.parse_args()
    asyncio.run(main(target=args.target, max_steps=args.max_steps))
