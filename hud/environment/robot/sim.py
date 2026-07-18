"""The sim-process shape every sim program runs: the sim owns the main thread.

There is one way to serve a simulator (see :mod:`~.bridge`), whatever the sim:
serving — the robot WebSocket and the control side channel — runs on a
background loop thread, and every sim touch is queued to the process main
thread through the shared :class:`SimThread`. One shape because the hardest
sims demand it: a simulator is usually *thread-affine* (every touch must run
on the thread that created its GL/device context), and Isaac/Omniverse must
own the process main thread outright — Kit drives its own main-thread loop and
``env.reset()`` nests ``run_until_complete``, which cannot run inside an
asyncio task. Cheap CPU envs pay ~nothing: the main thread just blocks on the
queue.

Before :meth:`SimThread.run` starts (tests, in-loop use) calls execute inline
on the caller — the degenerate single-thread case.
"""

from __future__ import annotations

import asyncio
import contextlib
import queue
import signal
import sys
import threading
from concurrent.futures import Future
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable, Coroutine


class SimThread:
    """Executes sim touches on the thread that owns the simulator.

    ``call`` is awaitable from any event loop and routes the touch to the
    owning thread; :meth:`run` is the blocking loop that thread drives.
    Re-entrant calls (a sim touch calling back in) and calls made before
    :meth:`run` starts execute inline.
    """

    _shared: SimThread | None = None

    def __init__(self) -> None:
        self._q: queue.Queue[tuple[Callable[[], Any], Future]] = queue.Queue()
        self._ident: int | None = None

    @classmethod
    def shared(cls) -> SimThread:
        """The process-wide instance (one main thread, one sim thread)."""
        if cls._shared is None:
            cls._shared = cls()
        return cls._shared

    async def call(self, fn: Callable[..., Any], *args: Any) -> Any:
        """Run ``fn(*args)`` on the sim thread, awaited on the caller's loop."""
        if self._ident is None or threading.get_ident() == self._ident:
            return fn(*args)  # not serving yet, or already on the sim thread
        fut: Future = Future()
        self._q.put((lambda: fn(*args), fut))
        return await asyncio.wrap_future(fut)

    def bind(self) -> None:
        """Claim the calling thread as the sim thread (before serving starts,
        so no touch can slip through inline on the wrong thread)."""
        self._ident = threading.get_ident()

    def run(self, until: Callable[[], bool]) -> None:
        """Own the calling thread: execute queued touches until ``until()``.

        Each pass drains the queue, then runs the idle hook — a Kit update when
        Omniverse is loaded (it needs continuous pumping), else nothing (the
        queue ``get`` timeout is the wait).
        """
        self.bind()
        while not until():
            kit = sys.modules.get("omni.kit.app")
            try:
                item = self._q.get(timeout=0.002 if kit else 0.05)
            except queue.Empty:
                pass
            else:
                self._execute(*item)
                with contextlib.suppress(queue.Empty):  # drain the backlog
                    while True:
                        self._execute(*self._q.get_nowait())
            if kit:
                kit.get_app().update()

    @staticmethod
    def _execute(fn: Callable[[], Any], fut: Future) -> None:
        if not fut.set_running_or_notify_cancel():
            return
        try:
            fut.set_result(fn())
        except BaseException as exc:  # propagate to the awaiting caller
            fut.set_exception(exc)


def run_with_sim(serve: Callable[[], Coroutine[Any, Any, Any]]) -> None:
    """THE process shape for serving a sim, blocking for the serve's lifetime.

    ``await serve()`` runs on a background loop thread while the shared
    :class:`SimThread` owns the calling (main) thread. SIGTERM and Ctrl-C
    cancel the serve coroutine; the sim keeps draining through its teardown
    (``env.stop()`` touches the sim too), then this returns.
    """
    sim = SimThread.shared()
    sim.bind()  # claim main before serving starts, so no touch runs inline elsewhere
    loop = asyncio.new_event_loop()
    done = threading.Event()
    task_box: dict[str, asyncio.Task[Any]] = {}

    async def _main() -> None:
        task_box["task"] = asyncio.current_task()  # type: ignore[assignment]
        with contextlib.suppress(asyncio.CancelledError):
            await serve()

    def _thread() -> None:
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(_main())
        finally:
            done.set()

    def _cancel(*_: object) -> None:
        task = task_box.get("task")
        if task is not None:
            loop.call_soon_threadsafe(task.cancel)

    with contextlib.suppress(ValueError):  # signals are main-thread-only; tests may not be
        signal.signal(signal.SIGTERM, _cancel)

    thread = threading.Thread(target=_thread, name="hud-serve", daemon=True)
    thread.start()
    try:
        sim.run(until=done.is_set)
    except KeyboardInterrupt:
        _cancel()
        sim.run(until=done.is_set)  # keep draining through teardown
    thread.join(timeout=30)


__all__ = ["SimThread", "run_with_sim"]
