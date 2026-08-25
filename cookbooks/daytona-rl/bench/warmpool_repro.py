"""Warm pool A/B: sequential vs concurrent sandbox creation.

    export DAYTONA_API_KEY=...
    uv run --with daytona python warmpool_repro.py
    uv run --with daytona python warmpool_repro.py --snapshot my-snapshot --n 8

Measures how long `daytona.create` takes from an existing snapshot, four ways:
one-at-a-time and N-at-once, each with and without a warm pool of size N.

Creates a warm pool on the chosen snapshot and deletes it at the end, along with
every sandbox it starts. Defaults to a stock Daytona base image so it runs with
no setup.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import statistics
import time

from daytona import AsyncDaytona, CreateSandboxFromSnapshotParams
from daytona_api_client_async import ApiClient, Configuration, WarmPoolsApi
from daytona_api_client_async.models.create_warm_pool import CreateWarmPool

HOST = "https://app.daytona.io/api"


def _api() -> ApiClient:
    key = os.environ.get("DAYTONA_API_KEY")
    if not key:
        raise SystemExit("set DAYTONA_API_KEY")
    return ApiClient(Configuration(host=HOST, access_token=key))


async def _create_once(daytona: AsyncDaytona, snapshot: str, ephemeral: bool) -> float:
    """Time one create, then delete the sandbox."""
    started = time.perf_counter()
    sandbox = await daytona.create(
        CreateSandboxFromSnapshotParams(
            snapshot=snapshot, ephemeral=ephemeral, auto_stop_interval=0
        ),
        timeout=180,
    )
    elapsed = time.perf_counter() - started
    await daytona.delete(sandbox)
    return elapsed


async def sequential(snapshot: str, reps: int, ephemeral: bool) -> list[float]:
    out = []
    async with AsyncDaytona() as daytona:
        for _ in range(reps):
            out.append(await _create_once(daytona, snapshot, ephemeral))
            await asyncio.sleep(15)
    return out


async def burst(snapshot: str, n: int, ephemeral: bool) -> list[float]:
    async with AsyncDaytona() as daytona:
        return sorted(
            await asyncio.gather(*(_create_once(daytona, snapshot, ephemeral) for _ in range(n)))
        )


async def pool_size(snapshot: str) -> int | None:
    async with _api() as api:
        pools = await WarmPoolsApi(api).list_warm_pools()
        return next((p.current_size for p in pools if p.snapshot == snapshot), None)


async def make_pool(snapshot: str, n: int, region: str, timeout_s: int = 600) -> int | None:
    async with _api() as api:
        await WarmPoolsApi(api).create_warm_pool(
            CreateWarmPool(snapshot=snapshot, pool=n, target=region)
        )
    started = time.perf_counter()
    while time.perf_counter() - started < timeout_s:
        current = await pool_size(snapshot)
        if current is not None and current >= n:
            print(f"  pool filled {current}/{n} in {time.perf_counter() - started:.0f}s")
            return current
        await asyncio.sleep(5)
    return await pool_size(snapshot)


async def drop_pool(snapshot: str) -> None:
    async with _api() as api:
        pools = WarmPoolsApi(api)
        for pool in await pools.list_warm_pools():
            if pool.snapshot == snapshot:
                await pools.delete_warm_pool(pool.id)


def show(label: str, values: list[float]) -> None:
    print(
        f"  {label:<22} median {statistics.median(values):5.2f}s   "
        f"[{', '.join(f'{v:.2f}' for v in values)}]"
    )


async def main(snapshot: str, n: int, region: str, ephemeral: bool) -> None:
    print(f"snapshot={snapshot}  n={n}  region={region}  ephemeral={ephemeral}\n")

    print("without a warm pool")
    cold_seq = await sequential(snapshot, 3, ephemeral)
    show(f"sequential x3", cold_seq)
    cold_burst = await burst(snapshot, n, ephemeral)
    show(f"{n} at once", cold_burst)

    print("\ncreating warm pool...")
    filled = await make_pool(snapshot, n, region)
    if not filled:
        print("  pool never filled — aborting")
        await drop_pool(snapshot)
        return

    try:
        print("\nwith a warm pool")
        warm_seq = await sequential(snapshot, 3, ephemeral)
        show(f"sequential x3", warm_seq)
        warm_burst = await burst(snapshot, n, ephemeral)
        show(f"{n} at once", warm_burst)
    finally:
        await drop_pool(snapshot)
        print("\npool deleted")

    seq_gain = statistics.median(cold_seq) - statistics.median(warm_seq)
    burst_gain = statistics.median(cold_burst) - statistics.median(warm_burst)
    print(f"\nsequential improved by {seq_gain:+.2f}s, {n}-at-once improved by {burst_gain:+.2f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", default="daytonaio/sandbox:0.7.0")
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--region", default="eu")
    parser.add_argument("--ephemeral", action="store_true", default=True)
    parser.add_argument("--no-ephemeral", dest="ephemeral", action="store_false")
    args = parser.parse_args()
    asyncio.run(main(args.snapshot, args.n, args.region, args.ephemeral))
