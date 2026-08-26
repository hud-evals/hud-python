"""Wrap any HUD Provider to record how long acquiring its sandbox took.

A Provider is just ``__call__(task) -> AsyncContextManager[Runtime]``
(hud/eval/runtime.py:118), so timing one needs no SDK changes.
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TimedProvider:
    """Delegates to *inner*, appending each spin-up duration to :attr:`spin_ups`."""

    inner: Any
    spin_ups: list[float] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)

    @asynccontextmanager
    async def __call__(self, task: Any):
        t0 = time.perf_counter()
        try:
            async with self.inner(task) as rt:
                self.spin_ups.append(time.perf_counter() - t0)
                yield rt
        except Exception as exc:
            self.failures.append(f"{type(exc).__name__}: {exc}")
            raise
