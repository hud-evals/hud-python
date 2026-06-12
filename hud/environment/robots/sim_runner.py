"""Sim execution strategies: *which thread* runs the (thread-affine) simulator.

A sim (MuJoCo/EGL, Isaac, a hardware SDK) is usually thread-affine — every touch must
happen on the thread that created it — yet the bridge serves on an asyncio loop a
blocking step must not stall. A ``SimRunner`` owns that "which thread, dispatched how"
decision behind one :meth:`SimRunner.call` verb, so bridge code is topology-agnostic:

- :class:`InlineSimRunner` — run on the caller (loop) thread. The default; for cheap/CPU
  sims and tests.
- :class:`ThreadSimRunner` — sim on a dedicated worker thread, HUD loop on main (launch
  with ``asyncio.run``). For render-heavy/blocking sims; used by the realtime bridges.
- :class:`MainThreadSimRunner` — sim on main, HUD loop on a worker. The inversion for
  runtimes that must own the main thread (Isaac/Omniverse); the main thread calls
  :meth:`serve_forever` to pump sim work.

Note: ``call`` dispatches arbitrary callables, so this is strictly in-process — crossing
a process boundary is a higher-layer concern (see ``notes/unified_framework.md``).
"""

from __future__ import annotations

import asyncio
import queue
import threading
from abc import ABC, abstractmethod
from concurrent.futures import Future
from typing import Any, Callable


class SimRunner(ABC):
    """Strategy for running thread-affine sim work off (or on) the loop thread.

    Subclasses decide which thread owns the sim; bridges funnel every sim touch through
    :meth:`call`.
    """

    @abstractmethod
    async def call(self, fn: Callable[..., Any], *args: Any) -> Any:
        """Run ``fn(*args)`` on the sim thread, awaited on the loop (inline if already
        on the sim thread, to avoid self-dispatch deadlock)."""

    def on_sim_thread(self) -> bool:
        """True if the caller is already on the sim thread (avoid self-dispatch)."""
        return False

    def serve_forever(self) -> None:
        """Pump submitted sim work until :meth:`shutdown` (only :class:`MainThreadSimRunner`
        uses this; it must run on the main thread)."""

    def shutdown(self) -> None:
        """Release any owned thread(s). Idempotent."""


class InlineSimRunner(SimRunner):
    """Run sim work inline on the caller's (loop) thread. The default; for cheap/CPU
    sims and tests."""

    async def call(self, fn: Callable[..., Any], *args: Any) -> Any:
        return fn(*args)

    def on_sim_thread(self) -> bool:
        return True


class ThreadSimRunner(SimRunner):
    """Run sim work on a dedicated worker thread (HUD loop keeps main).

    The sim's GL/device context binds to the worker, leaving the loop free during a
    blocking step. Launch with ``asyncio.run(...)``.
    """

    def __init__(self, *, thread_name_prefix: str = "sim") -> None:
        # Lazily created so the worker thread (and any per-thread context it owns) is
        # spawned by whatever event loop ends up driving us, not at construction time.
        self._loop_executor = None  # concurrent.futures.ThreadPoolExecutor (created on first use)
        self._thread_name_prefix = thread_name_prefix
        self._worker_ident: int | None = None

    def _ensure_executor(self):
        if self._loop_executor is None:
            from concurrent.futures import ThreadPoolExecutor

            self._loop_executor = ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix=self._thread_name_prefix,
                initializer=self._record_ident,
            )
        return self._loop_executor

    def _record_ident(self) -> None:
        # Runs once, on the worker thread, when the pool spins it up.
        self._worker_ident = threading.get_ident()

    async def call(self, fn: Callable[..., Any], *args: Any) -> Any:
        if self.on_sim_thread():
            return fn(*args)
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._ensure_executor(), lambda: fn(*args))

    def on_sim_thread(self) -> bool:
        return self._worker_ident is not None and threading.get_ident() == self._worker_ident

    def shutdown(self) -> None:
        if self._loop_executor is not None:
            self._loop_executor.shutdown(wait=False)
            self._loop_executor = None


class MainThreadSimRunner(SimRunner):
    """Run sim work on the main thread (HUD loop on a worker).

    The inversion for runtimes that must own the main thread (Isaac/Omniverse): boot the
    sim at import on main, run the HUD server on a daemon worker, then call
    :meth:`serve_forever` on main. :meth:`call` enqueues from the loop and awaits the result.
    """

    def __init__(self) -> None:
        self._q: queue.Queue[tuple[Callable[[], Any], Future] | None] = queue.Queue()
        self._stop = threading.Event()
        self._thread_ident: int | None = None

    def _submit(self, fn: Callable[[], Any]) -> Future:
        fut: Future = Future()
        self._q.put((fn, fut))
        return fut

    async def call(self, fn: Callable[..., Any], *args: Any) -> Any:
        if self.on_sim_thread():
            return fn(*args)
        return await asyncio.wrap_future(self._submit(lambda: fn(*args)))

    def on_sim_thread(self) -> bool:
        return self._thread_ident is not None and threading.get_ident() == self._thread_ident

    def serve_forever(self) -> None:
        """Execute submitted callables on this (main) thread until :meth:`shutdown`."""
        self._thread_ident = threading.get_ident()
        while not self._stop.is_set():
            try:
                item = self._q.get(timeout=0.1)
            except queue.Empty:
                continue
            if item is None:  # poison pill from shutdown()
                break
            fn, fut = item
            if not fut.set_running_or_notify_cancel():
                continue
            try:
                fut.set_result(fn())
            except BaseException as exc:  # noqa: BLE001 — propagate to the awaiting caller
                fut.set_exception(exc)

    def shutdown(self) -> None:
        self._stop.set()
        self._q.put(None)  # wake the pump if it is blocked on get()


__all__ = [
    "InlineSimRunner",
    "MainThreadSimRunner",
    "SimRunner",
    "ThreadSimRunner",
]
