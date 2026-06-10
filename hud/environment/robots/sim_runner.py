"""Sim execution strategies: *which thread* runs the (thread-affine) simulator.

A robot env's simulator — a MuJoCo/EGL render context, an Isaac/Omniverse app, or a
hardware SDK — is almost always **thread-affine**: every touch (create / reset / step /
render / close) must happen on the one thread that created it. Meanwhile the HUD
:class:`~hud.environment.robots.bridge.RobotBridge` serves its channels on an asyncio
event loop, and a blocking, often render-heavy sim step must not stall that loop.

A ``SimRunner`` captures the single decision *"which thread owns the sim, and how do I
dispatch work onto it"*, so the bridge code stays identical regardless of topology.
There are three strategies:

- :class:`InlineSimRunner` — no extra thread; run on the caller (event-loop) thread.
  For trivial/CPU sims and tests, where a step is cheap and there is no GL context to
  keep thread-affine. This is the default, so a plain ``RobotBridge`` behaves exactly as
  it did before this abstraction existed.

- :class:`ThreadSimRunner` — the sim runs on a dedicated **worker** thread; the HUD loop
  keeps the **main** thread. Launch with a plain ``asyncio.run(...)``. This is the right
  choice for render-heavy / blocking sims (and real robots): the GL/EGL context binds to
  the worker, and the loop stays free to stream observations / receive actions while a
  step runs. It is what the realtime bridges use.

- :class:`MainThreadSimRunner` — the sim runs on the **main** thread; the HUD loop runs on
  a **worker** thread. This is the inversion required by runtimes that *must* own the main
  thread — notably Isaac Lab / Omniverse, which boots at import time, pins its GL context
  and a private asyncio loop to that thread, and cannot share a thread with the HUD loop
  (two asyncio loops can't run on one thread). The process runs the HUD loop on a worker
  and calls :meth:`MainThreadSimRunner.serve_forever` on the main thread to pump sim work.

All three expose the same :meth:`SimRunner.call` dispatch verb, so a bridge says
``await self._sim_runner.call(self.step, action)`` and never has to know which thread (or
even which strategy) is in play.

.. note::
   A ``SimRunner`` dispatches *arbitrary Python callables*, so it is strictly an
   **in-process** concept — you cannot ship a closure across a process boundary. Crossing
   processes (a sim hosted in its own process) is a separate, future concern handled at a
   higher layer; see ``notes/unified_framework.md``.
"""

from __future__ import annotations

import asyncio
import queue
import threading
from abc import ABC, abstractmethod
from concurrent.futures import Future
from typing import Any, Callable


class SimRunner(ABC):
    """Strategy for running thread-affine simulator work off (or on) the loop thread.

    Subclasses decide *which* thread owns the sim. Bridges funnel every simulator touch
    through :meth:`call` so the dispatch is uniform across strategies.
    """

    @abstractmethod
    async def call(self, fn: Callable[..., Any], *args: Any) -> Any:
        """Run ``fn(*args)`` on the sim thread and await its result on the loop.

        Implementations must not block the event loop while the sim work runs (except
        :class:`InlineSimRunner`, which has no other thread to offload to). If the caller
        is already on the sim thread, the call runs inline to avoid self-dispatch deadlock.
        """

    def on_sim_thread(self) -> bool:
        """True if the caller is already running on the sim thread (avoid self-dispatch)."""
        return False

    def serve_forever(self) -> None:
        """Pump submitted sim work until :meth:`shutdown`. Blocks the calling thread.

        Only :class:`MainThreadSimRunner` does real work here — it must be called on the
        process main thread. The others are launched via ``asyncio.run`` and never use it.
        """

    def shutdown(self) -> None:
        """Release any owned thread(s). Idempotent."""


class InlineSimRunner(SimRunner):
    """Run sim work on the caller's thread — no extra thread, no offload.

    The default. A step runs inline on the event loop, exactly as a bare ``RobotBridge``
    behaved before ``SimRunner`` existed. Suitable for cheap/CPU sims and tests.
    """

    async def call(self, fn: Callable[..., Any], *args: Any) -> Any:
        return fn(*args)

    def on_sim_thread(self) -> bool:
        return True


class ThreadSimRunner(SimRunner):
    """Run sim work on a single dedicated worker thread; the HUD loop owns the main thread.

    The sim's GL/EGL/device context binds to the worker (the first thread to touch it),
    and the event loop stays free to service the control / data channels while a
    (blocking, GIL-releasing) step runs. Launch the process with ``asyncio.run(...)``.
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
    """Run sim work on the **main** thread; the HUD loop runs on a worker thread.

    The inversion required by runtimes that must own the main thread (Isaac/Omniverse).
    Wiring: boot the sim at import on the main thread, start the HUD asyncio server on a
    daemon worker thread, then call :meth:`serve_forever` on the main thread to execute
    every submitted sim callable there. :meth:`call` (invoked from the HUD loop on the
    worker) enqueues work and awaits the result without blocking the loop.
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
