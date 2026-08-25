"""Daytona warm pools: create one, wait until it is genuinely full, drop it.

Warm pools are absent from the high-level `daytona` SDK — they live in the
generated client (`WarmPoolsApi`). The subtlety worth knowing: a pool's own
`current_size` reports full roughly 12s before the sandboxes exist, so
`wait_full` counts real sandboxes instead. Counting them needs
`include_warm=True`; unclaimed pool members are excluded from `list_sandboxes`
by default, which makes a naive count return 0 forever.
"""

from __future__ import annotations

import asyncio
import os
import time

from daytona_api_client_async import ApiClient, Configuration, SandboxApi, WarmPoolsApi
from daytona_api_client_async.models.create_warm_pool import CreateWarmPool

HOST = "https://app.daytona.io/api"
LIVE_STATES = {"started", "running"}


def _api() -> ApiClient:
    return ApiClient(Configuration(host=HOST, access_token=os.environ["DAYTONA_API_KEY"]))


def _state(sandbox) -> str:
    return str(getattr(sandbox.state, "value", sandbox.state)).lower()


async def _live_ids(snapshot: str, *, include_warm: bool) -> set[str]:
    """Live sandbox ids on *snapshot*, paginated (`limit` caps at 200)."""
    ids: set[str] = set()
    cursor = None
    async with _api() as api:
        sandboxes = SandboxApi(api)
        while True:
            page = await sandboxes.list_sandboxes(
                include_warm=include_warm, snapshots=[snapshot], limit=200, cursor=cursor
            )
            items = page.items if hasattr(page, "items") else page
            ids.update(i.id for i in items if _state(i) in LIVE_STATES)
            cursor = getattr(page, "next_cursor", None)
            if not cursor or not items:
                return ids


async def unclaimed(snapshot: str) -> int:
    """Warm sandboxes nobody has claimed yet, as a set difference so it cannot
    overshoot when a sandbox is deleted between the two listings."""
    warm = await _live_ids(snapshot, include_warm=True)
    claimed = await _live_ids(snapshot, include_warm=False)
    return len(warm - claimed)


async def region_of(sandbox_id: str) -> str:
    """The region a sandbox actually landed in.

    DaytonaRuntime creates sandboxes without a target, so they go to the org
    default. Daytona only serves a pooled sandbox when the create request's
    region matches the pool's, so a hardcoded region silently bills for a pool
    nobody claims. Read it off a real sandbox instead.
    """
    async with _api() as api:
        sandbox = await SandboxApi(api).get_sandbox(sandbox_id)
    return str(sandbox.target)


async def create(snapshot: str, size: int, target: str) -> None:
    async with _api() as api:
        await WarmPoolsApi(api).create_warm_pool(
            CreateWarmPool(snapshot=snapshot, pool=size, target=target)
        )
    print(f"  pool created: {snapshot} size={size} target={target}", flush=True)


async def drop(snapshot: str) -> None:
    async with _api() as api:
        pools = WarmPoolsApi(api)
        for pool in await pools.list_warm_pools():
            if pool.snapshot == snapshot:
                await pools.delete_warm_pool(pool.id)
                print(f"  deleted warm pool {pool.id}", flush=True)


async def wait_full(snapshot: str, size: int, timeout_s: float = 1800, poll_s: float = 5) -> int:
    """Block until *size* sandboxes really exist. Returns how many."""
    started = time.perf_counter()
    last = -1
    while time.perf_counter() - started < timeout_s:
        live = await unclaimed(snapshot)
        if live != last:
            print(f"  pool: {live}/{size} @ {time.perf_counter() - started:.0f}s", flush=True)
            last = live
        if live >= size:
            return live
        await asyncio.sleep(poll_s)
    return last
