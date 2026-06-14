"""Sim execution strategies: *which thread* runs the (thread-affine) simulator.

A sim (MuJoCo/EGL, a hardware SDK) is usually thread-affine — every touch must run
on the thread that created it — but the bridge's asyncio loop can't be stalled by a
blocking step. A :class:`SimRunner` hides that choice behind one :meth:`~SimRunner.call`
verb:

- :class:`InlineSimRunner` — runs on the loop thread. Default; for cheap/CPU sims + tests.
- :class:`ThreadSimRunner` — sim on a dedicated worker thread, loop kept free. For
  heavy/blocking sims; used by the realtime bridges.
"""

from __future__ import annotations

import asyncio
import threading
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable


class SimRunner(ABC):
    """Strategy for *which thread* runs the (thread-affine) sim; bridges route every
    sim touch through :meth:`call`, so the choice is a one-line injection."""

    @abstractmethod
    async def call(self, fn: Callable[..., Any], *args: Any) -> Any:
        """Run ``fn(*args)`` on the sim thread, awaited on the loop."""

    def shutdown(self) -> None:  # noqa: B027  # optional hook: default no-op, subclasses override if they own threads
        """Release any owned thread(s). Idempotent."""


class InlineSimRunner(SimRunner):
    """Run sim work inline on the caller's (loop) thread. The default; for cheap/CPU
    sims and tests."""

    async def call(self, fn: Callable[..., Any], *args: Any) -> Any:
        return fn(*args)


class ThreadSimRunner(SimRunner):
    """Sim on a dedicated worker thread: the GL/device context binds to the worker,
    leaving the loop free during a blocking step. Used by the realtime bridges."""

    def __init__(self, *, thread_name_prefix: str = "sim") -> None:
        self._worker_ident: int | None = None
        # max_workers=1 -> the worker spawns lazily on first submit; its initializer
        # records the ident so re-entrant calls (already on the sim thread) run inline.
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix=thread_name_prefix, initializer=self._record_ident
        )

    def _record_ident(self) -> None:
        self._worker_ident = threading.get_ident()

    async def call(self, fn: Callable[..., Any], *args: Any) -> Any:
        if threading.get_ident() == self._worker_ident:  # avoid self-dispatch deadlock
            return fn(*args)
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, lambda: fn(*args))

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False)


__all__ = ["InlineSimRunner", "SimRunner", "ThreadSimRunner"]
