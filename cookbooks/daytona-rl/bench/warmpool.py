"""A/B the Daytona warm-pool effect on sandbox spin-up.

    PYTHONPATH=. uv run bench/warmpool.py --n 8

Warm pools are absent from the high-level ``daytona`` SDK; they live in the
generated client (``WarmPoolsApi``). This measures one concurrent wave of N
acquisitions with no pool, then creates a pool of N, waits for it to fill, and
measures an identical wave. The difference shows whether pooled sandboxes serve
DaytonaRuntime's creates, which use ``ephemeral=True`` and ``auto_stop_interval=0``.
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
import time

from daytona import AsyncDaytona, Image
from hud.clients import connect
from hud.runtime import DaytonaRuntime

import pool
from env import fix_calc
from snapshot import snapshot_name


async def one(provider) -> tuple[float | None, str]:
    t0 = time.perf_counter()
    try:
        async with provider(fix_calc(variant=0)) as rt:
            elapsed = time.perf_counter() - t0
            async with connect(rt) as client:
                assert client.manifest is not None
            return elapsed, ""
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


async def wave(n: int, label: str) -> dict:
    provider = DaytonaRuntime(snapshot_name(), image=Image.from_dockerfile("Dockerfile.hud"))
    t0 = time.perf_counter()
    results = await asyncio.gather(*(one(provider) for _ in range(n)))
    wall = time.perf_counter() - t0

    spins = sorted(s for s, _ in results if s is not None)
    errors = [e for _, e in results if e]
    row = {
        "label": label,
        "n": n,
        "ok": len(spins),
        "failed": n - len(spins),
        "wall_s": round(wall, 1),
        "p50": round(statistics.median(spins), 2) if spins else None,
        "min": round(spins[0], 2) if spins else None,
        "max": round(spins[-1], 2) if spins else None,
    }
    print(f"[{label}] {row}", flush=True)
    for e in dict.fromkeys(errors):
        print(f"    {e[:180]}", flush=True)
    return row


async def main(n: int) -> None:
    snapshot = snapshot_name()
    print(f"snapshot: {snapshot}")

    cold = await wave(n, "no pool")

    print("\ncreating warm pool...")
    async with AsyncDaytona() as d:
        page = await d.snapshot.list()
        built = next(s.name for s in page.items if s.name.startswith(snapshot))
    print(f"pool target snapshot: {built}")

    # Match the pool's region to where the runtime actually creates sandboxes.
    provider = DaytonaRuntime(built, image=Image.from_dockerfile("Dockerfile.hud"))
    async with provider(fix_calc(variant=0)) as probe:
        target = await pool.region_of(probe.params["instance_id"])
    print(f"region: {target}")

    try:
        await pool.create(built, n, target)
        filled = await pool.wait_full(built, n, timeout_s=600)
        print(f"pool reported {filled}/{n}\n")
        warm = await wave(n, "warm pool")
    finally:
        await pool.drop(built)

    if cold["p50"] and warm["p50"]:
        delta = (cold["p50"] - warm["p50"]) / cold["p50"] * 100
        print(f"\np50 {cold['p50']}s -> {warm['p50']}s  ({delta:+.0f}%)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=8)
    a = p.parse_args()
    asyncio.run(main(a.n))
