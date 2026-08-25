"""The published training run: 10 steps of GRPO, each a 128-wide burst on Daytona.

    uv run train.py --steps 10

Shape per step: 16 bug variants x 8 attempts = 128 rollouts against a warm pool
of 128 (pool = batch, the sizing rule the guide gives). Variants 0-15 only;
16-23 are held out so the before/after pass rate is measured on bugs the model
never trained on.

Records what the guide needs: pass rate per step (the curve), within-group reward
spread (what GRPO actually consumes), wall-clock split rollouts vs training, and
the count of rollouts that never launched — an infra failure arrives as a
reward-0.0 run with no tokens, which is indistinguishable from a model failure
unless it is counted separately.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import resource
import signal
import statistics
import time
from pathlib import Path

from daytona import Image
from hud import TrainingClient
from hud.agents import create_agent
from hud.agents.types import AgentStep
from hud.eval import DaytonaRuntime, Job, Taskset

from env import fix_calc
from snapshot import snapshot_name
import pool

MODEL = "daytona-calc-3"
SNAPSHOT = snapshot_name()
TRAIN_VARIANTS = range(16)  # 16-23 held out for evaluation


def _samples(runs: list) -> list:
    return [
        sample
        for run in runs
        for sample in run.trace.collect(
            lambda s: s.sample if isinstance(s, AgentStep) and s.sample else None
        )
    ]


def _group_spread(runs: list) -> float | None:
    """Mean within-group reward stdev — the quantity GRPO turns into advantage."""
    groups: dict[str, list[float]] = {}
    for run in runs:
        gid = getattr(run, "group_id", None)
        if gid is not None:
            groups.setdefault(gid, []).append(run.reward or 0.0)
    spreads = [statistics.pstdev(v) for v in groups.values() if len(v) > 1]
    return round(statistics.mean(spreads), 4) if spreads else None


async def main(
    *, steps: int, group: int, lr: float, concurrent: int, chunk: int, out: Path
) -> None:
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    want = concurrent * 16 + 512
    if soft < want:
        resource.setrlimit(resource.RLIMIT_NOFILE, (min(want, hard), hard))

    agent = create_agent(
        MODEL,
        completion_kwargs={"max_tokens": 2048, "extra_body": {"return_token_ids": True}},
    )
    trainer = TrainingClient(MODEL)
    taskset = Taskset("calc", [fix_calc(variant=v) for v in TRAIN_VARIANTS])
    runtime = DaytonaRuntime(SNAPSHOT, image=Image.from_dockerfile("Dockerfile.hud"))

    # A killed process skips the finally and strands the whole pool (128 vCPU),
    # so cancel the loop on a signal and let the finally run.
    loop = asyncio.get_running_loop()
    stopping = asyncio.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        with contextlib.suppress(NotImplementedError):
            loop.add_signal_handler(sig, stopping.set)

    # Resume-safe: a killed run loses its pool but keeps its checkpoints, so
    # re-invoking must append to the curve rather than restart it at step 0.
    history: list[dict] = json.loads(out.read_text()) if out.exists() else []
    step_offset = (history[-1]["step"] + 1) if history else 0
    if step_offset:
        print(f"resuming after step {step_offset - 1} ({len(history)} recorded)", flush=True)
    # Daytona rejects a warm pool whose snapshot does not exist yet (400), and
    # DaytonaRuntime only builds the snapshot on its first acquisition — so take
    # one sandbox first to force the build, then pool against it.
    print("building snapshot (first run only)...", flush=True)
    async with runtime(fix_calc(variant=0)):
        pass

    await pool.drop(SNAPSHOT)
    await pool.create(SNAPSHOT, concurrent)
    try:
        filled = await pool.wait_full(SNAPSHOT, concurrent)
        if filled < concurrent:
            print(
                f"WARNING: pool reached {filled}/{concurrent}; rollouts above that "
                f"run cold and the pool-sized timings will not hold",
                flush=True,
            )
        else:
            print(f"pool full: {filled}/{concurrent}", flush=True)

        session = await Job.start("calc-rl-daytona-published", group=group)
        print(f"job {session.id} | model {MODEL} | variants {list(TRAIN_VARIANTS)}", flush=True)

        for step in range(step_offset, step_offset + steps):
            if stopping.is_set():
                print("stop requested — exiting cleanly", flush=True)
                break
            batch_start = len(session.runs)

            t0 = time.perf_counter()
            await taskset.run(
                agent, runtime=runtime, job=session, group=group, max_concurrent=concurrent
            )
            rollout_s = time.perf_counter() - t0
            batch = session.runs[batch_start:]

            samples = _samples(batch)
            rewards = [r.reward or 0.0 for r in batch]
            errored = sum(1 for r in batch if r.trace.is_error)
            solved = sum(1 for r in rewards if r > 0)
            row = {
                "step": step,
                "runs": len(batch),
                "solved": solved,
                "pass_rate": round(solved / len(batch), 4) if batch else None,
                "errored": errored,
                "group_spread": _group_spread(batch),
                "samples": len(samples),
                "tokens": sum(len(s.output_token_ids) for s in samples),
                "rollout_s": round(rollout_s, 1),
            }
            print(
                f"\nstep {step}: pass {solved}/{len(batch)} = {row['pass_rate']:.1%} | "
                f"errored {errored} | spread {row['group_spread']} | "
                f"rollout {rollout_s:.0f}s | {row['tokens']} tok",
                flush=True,
            )
            if not samples:
                print("  !! no token-level samples — cannot train", flush=True)
                history.append(row | {"aborted": "no_samples"})
                break

            # 128 runs with tokens inline is ~37MB and the endpoint rejects it
            # (413). Several forward_backward calls accumulate into one
            # optim_step, so send the batch in group-aligned chunks.
            t1 = time.perf_counter()
            loss_sum, datums = 0.0, 0
            for i in range(0, len(batch), chunk):
                part = batch[i : i + chunk]
                fb = await trainer.forward_backward(
                    part, loss_fn="importance_sampling", group_size=group
                )
                loss_sum += fb.metrics.get("loss:sum") or 0.0
                datums += fb.num_datums
            row["loss"] = round(loss_sum, 4)
            row["datums"] = datums
            row["chunks"] = -(-len(batch) // chunk)

            if len({round(r, 6) for r in rewards}) == 1:
                row["train_s"] = round(time.perf_counter() - t1, 1)
                row["optim_step"] = None
                print(f"  no reward spread in batch — skipping optim_step", flush=True)
            else:
                result = await trainer.optim_step(learning_rate=lr)
                row["train_s"] = round(time.perf_counter() - t1, 1)
                row["optim_step"] = result.step
                print(
                    f"  fwd/bwd+optim {row['train_s']}s | loss {row['loss']} | "
                    f"optim step {result.step}",
                    flush=True,
                )
            history.append(row)
            out.write_text(json.dumps(history, indent=1))
    finally:
        await pool.drop(SNAPSHOT)
        out.write_text(json.dumps(history, indent=1))
        print(f"\nwrote {out} ({len(history)} steps); pool dropped", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--group", type=int, default=8)
    p.add_argument("--concurrent", type=int, default=128)
    p.add_argument("--learning-rate", type=float, default=1e-5)
    p.add_argument("--chunk", type=int, default=16)
    p.add_argument("--out", default="train_run.json")
    a = p.parse_args()
    asyncio.run(
        main(
            steps=a.steps,
            group=a.group,
            lr=a.learning_rate,
            concurrent=a.concurrent,
            chunk=a.chunk,
            out=Path(a.out),
        )
    )
