"""A/B the Daytona warm-pool effect on sandbox spin-up.

    uv run warmpool.py --n 8

Warm pools are absent from the high-level `daytona` SDK; they live in the
generated client (`WarmPoolsApi`). This measures one concurrent wave of N
acquisitions with no pool, then creates a pool of N, waits for it to fill, and
measures an identical wave.

Open question the numbers answer: DaytonaRuntime creates sandboxes with
ephemeral=True and auto_stop_interval=0, so a pooled sandbox is only reused if
the pool matches those create parameters.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import statistics
import time

from daytona import Image
from daytona_api_client_async import ApiClient, Configuration, WarmPoolsApi
from daytona_api_client_async.models.create_warm_pool import CreateWarmPool
from hud.clients import connect
from hud.eval import DaytonaRuntime

import pool
from env import fix_calc
from snapshot import snapshot_name

HOST = "https://app.daytona.io/api"


def _api() -> ApiClient:
    return ApiClient(Configuration(host=HOST, access_token=os.environ["DAYTONA_API_KEY"]))


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


async def fill_pool(snapshot: str, n: int, timeout_s: int = 600) -> int:
    """Create the pool, then wait for sandboxes that actually exist.

    The pool's own ``current_size`` reports the target roughly 12s before the
    sandboxes are there, so gating on it starts the warm wave against a
    half-empty pool and understates the benefit. ``pool.wait_full`` counts real
    unclaimed sandboxes instead.
    """
    async with _api() as api:
        await WarmPoolsApi(api).create_warm_pool(
            CreateWarmPool(snapshot=snapshot, pool=n, target="eu")
        )
    return await pool.wait_full(snapshot, n, timeout_s=timeout_s)


async def drop_pool(snapshot: str) -> None:
    async with _api() as api:
        pools = WarmPoolsApi(api)
        for p in await pools.list_warm_pools():
            if p.snapshot == snapshot:
                await pools.delete_warm_pool(p.id)
                print(f"deleted warm pool {p.id}")


async def main(n: int) -> None:
    snapshot = snapshot_name()
    print(f"snapshot base: {snapshot}  (digest suffix added by DaytonaRuntime)")

    cold = await wave(n, "no pool")

    print("\ncreating warm pool...")
    from daytona import AsyncDaytona

    async with AsyncDaytona() as d:
        page = await d.snapshot.list()
        built = next(s.name for s in page.items if s.name.startswith(snapshot))
    print(f"pool target snapshot: {built}")

    try:
        filled = await fill_pool(built, n)
        print(f"pool reported {filled}/{n}\n")
        warm = await wave(n, "warm pool")
    finally:
        await drop_pool(built)

    if cold["p50"] and warm["p50"]:
        delta = (cold["p50"] - warm["p50"]) / cold["p50"] * 100
        print(f"\np50 {cold['p50']}s -> {warm['p50']}s  ({delta:+.0f}%)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=8)
    a = p.parse_args()
    asyncio.run(main(a.n))
