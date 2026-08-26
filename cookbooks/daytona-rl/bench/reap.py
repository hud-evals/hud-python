"""Delete leftover Daytona sandboxes and stale snapshots.

    uv run reap.py                        # list what would be deleted
    uv run reap.py --delete               # delete stranded sandboxes
    uv run reap.py --snapshots --delete   # also delete snapshots except the current hash

Sandboxes: DaytonaRuntime sets auto_stop_interval=0 (never auto-stop) and relies
on its own teardown, wrapped in contextlib.suppress. A killed client or a failed
delete strands them, billing until someone notices.

Snapshots: content-addressed names (see snapshot.py) mean every edit to env.py or
Dockerfile.hud mints a new one, and nothing reaps the old.
"""

from __future__ import annotations

import argparse
import asyncio

from daytona import AsyncDaytona

from snapshot import snapshot_name

PREFIX = "hud-smoke"


async def reap_sandboxes(daytona: AsyncDaytona, *, delete: bool, prefix: str) -> None:
    sandboxes = [s async for s in daytona.list()]
    ours = [s for s in sandboxes if str(getattr(s, "snapshot", "") or "").startswith(prefix)]

    print(f"{len(sandboxes)} sandboxes total, {len(ours)} matching {prefix!r}")
    for s in ours:
        print(f"  {s.id} state={s.state} snapshot={s.snapshot} created={s.created_at}")
    if not ours or not delete:
        if ours:
            print("  dry run — pass --delete to remove these")
        return

    results = await asyncio.gather(*(daytona.delete(s) for s in ours), return_exceptions=True)
    failed = [(s.id, r) for s, r in zip(ours, results) if isinstance(r, Exception)]
    print(f"  deleted {len(ours) - len(failed)}/{len(ours)}")
    for sid, err in failed:
        print(f"    FAILED {sid}: {type(err).__name__}: {err}")


async def reap_snapshots(daytona: AsyncDaytona, *, delete: bool, prefix: str) -> None:
    keep = snapshot_name()
    page = await daytona.snapshot.list()
    ours = [s for s in page.items if s.name.startswith(prefix) and s.name != keep]

    print(f"\n{len(ours)} stale snapshots matching {prefix!r} (keeping {keep})")
    for s in ours:
        print(f"  {s.name} {s.state} size={getattr(s, 'size', None)}")
    if not ours or not delete:
        if ours:
            print("  dry run — pass --delete to remove these")
        return

    results = await asyncio.gather(
        *(daytona.snapshot.delete(s) for s in ours), return_exceptions=True
    )
    failed = [(s.name, r) for s, r in zip(ours, results) if isinstance(r, Exception)]
    print(f"  deleted {len(ours) - len(failed)}/{len(ours)}")
    for name, err in failed:
        print(f"    FAILED {name}: {type(err).__name__}: {err}")


async def main(*, delete: bool, prefix: str, snapshots: bool) -> None:
    async with AsyncDaytona() as daytona:
        await reap_sandboxes(daytona, delete=delete, prefix=prefix)
        if snapshots:
            await reap_snapshots(daytona, delete=delete, prefix=prefix)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--delete", action="store_true")
    p.add_argument("--snapshots", action="store_true")
    p.add_argument("--prefix", default=PREFIX)
    a = p.parse_args()
    asyncio.run(main(delete=a.delete, prefix=a.prefix, snapshots=a.snapshots))
