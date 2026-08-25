"""One rollout of the smoke task on a Daytona sandbox, with timings.

    uv run run_daytona.py

Separates the two costs the guide needs to quote: sandbox spin-up (snapshot
resolve + create + serve + ssh forward) and agent time.
"""

from __future__ import annotations

import asyncio
import time

from daytona import Image
from hud.agents import create_agent
from hud.eval import DaytonaRuntime

from env import fix_calc
from snapshot import snapshot_name
from timing import TimedProvider

SNAPSHOT = snapshot_name()

# The model this repo trains. Any gateway model works — swap it for "claude" or
# a base model to see the untrained behaviour.
MODEL = "daytona-calc-3"


async def main() -> None:
    provider = TimedProvider(
        DaytonaRuntime(SNAPSHOT, image=Image.from_dockerfile("Dockerfile.hud"))
    )

    t0 = time.perf_counter()
    job = await fix_calc().run(create_agent(MODEL), runtime=provider)
    total = time.perf_counter() - t0

    spin_up = provider.spin_ups[0] if provider.spin_ups else float("nan")
    print(f"\n[result]  reward={job.reward}")
    print(f"[timing]  spin_up={spin_up:.1f}s  agent={total - spin_up:.1f}s  total={total:.1f}s")
    if provider.failures:
        print(f"[failures] {provider.failures}")


if __name__ == "__main__":
    asyncio.run(main())
