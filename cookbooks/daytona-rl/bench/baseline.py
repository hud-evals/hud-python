"""Probe the task's difficulty before committing to a training run.

    uv run baseline.py                       # 24 variants, 1 rollout each
    uv run baseline.py --variants 24 --group 2

Runs the same agent config training would use, with no optimizer. The decision
this informs: a ~20-40% pass rate leaves room for GRPO to learn. Near 0% or near
100% means retune the task instead of spending hours discovering the gradient was
dead.
"""

from __future__ import annotations

import argparse
import asyncio
import time
from collections import Counter

from daytona import Image
from hud.agents import create_agent
from hud.eval import DaytonaRuntime, Job, Taskset

import bugs
from env import fix_calc
from snapshot import snapshot_name

MODEL = "daytona-calc-3"


async def main(*, variants: int, group: int, max_concurrent: int, start: int, model: str) -> None:
    agent = create_agent(
        model,
        completion_kwargs={"max_tokens": 2048, "extra_body": {"return_token_ids": True}},
    )
    variant_ids = list(range(start, start + variants))
    taskset = Taskset("calc", [fix_calc(variant=v) for v in variant_ids])
    runtime = DaytonaRuntime(snapshot_name(), image=Image.from_dockerfile("Dockerfile.hud"))

    print(
        f"model={model} variants={variant_ids} group={group} rollouts={len(variant_ids) * group} tests={bugs.test_count()}"
    )
    session = await Job.start(f"calc-baseline-{start}-{start + variants - 1}", group=group)
    t0 = time.perf_counter()
    await taskset.run(agent, runtime=runtime, job=session, max_concurrent=max_concurrent)
    wall = time.perf_counter() - t0

    runs = session.runs
    rewards = [r.reward for r in runs]
    solved = sum(1 for r in rewards if r == 1.0)

    by_k: Counter = Counter()
    solved_by_k: Counter = Counter()
    unattributed = 0
    for run in runs:
        variant = getattr(run, "_args", {}).get("variant")
        if variant is None:
            unattributed += 1
            continue
        k = len(bugs.broken_for(variant))
        by_k[k] += 1
        if run.reward == 1.0:
            solved_by_k[k] += 1

    launched = len(runs) - unattributed
    print(f"\nwall {wall:.1f}s | runs {len(runs)} | launched {launched}")

    if unattributed:
        print(
            f"\n{unattributed}/{len(runs)} rollouts never launched — no difficulty "
            f"signal here. Check the env starts in the sandbox (a missing file in "
            f"Dockerfile.hud shows up as \"env closed connection during 'hello'\")."
        )
        return

    print(f"pass rate: {solved}/{launched} = {solved / launched:.1%}")
    for k in sorted(by_k):
        print(f"  k={k} bugs: {solved_by_k[k]}/{by_k[k]} = {solved_by_k[k] / by_k[k]:.1%}")

    rate = solved / launched
    verdict = (
        "in range — a training run is justified"
        if 0.15 <= rate <= 0.55
        else "too easy — add bugs or harder ones"
        if rate > 0.55
        else "too hard — reduce k or drop the subtlest bugs"
    )
    print(f"\nverdict: {rate:.1%} {verdict}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--variants", type=int, default=24)
    p.add_argument("--group", type=int, default=1)
    p.add_argument("--max-concurrent", type=int, default=12)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--model", default=MODEL)
    a = p.parse_args()
    asyncio.run(
        main(
            variants=a.variants,
            group=a.group,
            max_concurrent=a.max_concurrent,
            start=a.start,
            model=a.model,
        )
    )
