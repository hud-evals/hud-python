"""Sim execution strategies: *which thread* runs the (thread-affine) simulator.

A sim (MuJoCo/EGL, Isaac, a hardware SDK) is usually thread-affine — every touch must
run on the thread that created it — but the bridge's asyncio loop can't be stalled by a
blocking step. A ``SimRunner`` hides that "which thread, dispatched how" choice behind
one :meth:`SimRunner.call` verb, keeping bridge code identical across all three:

- :class:`InlineSimRunner` — runs on the loop thread. Default; for cheap/CPU sims + tests.
- :class:`ThreadSimRunner` — sim on a worker thread, loop on main. For heavy/blocking
  sims; used by the realtime bridges.
- :class:`MainThreadSimRunner` — sim on main, loop on a worker. For runtimes that must
  own main (Isaac/Omniverse); main calls :meth:`serve_forever` to pump work.
"""

from __future__ import annotations

import asyncio
import queue
import threading
from abc import ABC, abstractmethod
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Callable


class SimRunner(ABC):
    """Strategy for running thread-affine sim work; bridges route every sim touch
    through :meth:`call`."""

    @abstractmethod
    async def call(self, fn: Callable[..., Any], *args: Any) -> Any:
        """Run ``fn(*args)`` on the sim thread, awaited on the loop (inline if already
        on the sim thread, to avoid self-dispatch deadlock)."""

    def on_sim_thread(self) -> bool:
        """True if the caller is already on the sim thread (avoid self-dispatch)."""
        return False

    def serve_forever(self) -> None:
        """Pump submitted work until :meth:`shutdown` (only MainThreadSimRunner; on main)."""

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
    """Sim on a dedicated worker thread (HUD loop keeps main): the GL/device context
    binds to the worker, leaving the loop free during a blocking step. ``asyncio.run``."""

    def __init__(self, *, thread_name_prefix: str = "sim") -> None:
        self._worker_ident: int | None = None
        # max_workers=1 -> the worker spawns lazily on first submit; its initializer
        # records the ident so on_sim_thread() can detect re-entrant calls.
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix=thread_name_prefix, initializer=self._record_ident
        )

    def _record_ident(self) -> None:
        self._worker_ident = threading.get_ident()

    async def call(self, fn: Callable[..., Any], *args: Any) -> Any:
        if self.on_sim_thread():
            return fn(*args)
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, lambda: fn(*args))

    def on_sim_thread(self) -> bool:
        return self._worker_ident is not None and threading.get_ident() == self._worker_ident

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False)


class MainThreadSimRunner(SimRunner):
    """Sim on the main thread (HUD loop on a worker): the inversion for runtimes that must
    own main (Isaac/Omniverse). Boot the sim on main, run HUD on a daemon worker, then call
    :meth:`serve_forever` on main; :meth:`call` enqueues from the loop and awaits."""

    def __init__(self) -> None:
        self._q: queue.Queue[tuple[Callable[[], Any], Future] | None] = queue.Queue()
        self._stop = threading.Event()
        self._thread_ident: int | None = None

    async def call(self, fn: Callable[..., Any], *args: Any) -> Any:
        if self.on_sim_thread():
            return fn(*args)
        fut: Future = Future()
        self._q.put((lambda: fn(*args), fut))
        return await asyncio.wrap_future(fut)

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
