"""Batched inference for concurrent robot rollouts.

- BatchedModel: stacks concurrent ainfer calls into one infer
- BatchedAgent: gives each rollout its own state, shares one batched model
"""

from __future__ import annotations

import asyncio
import copy
import importlib
from typing import TYPE_CHECKING, Any

from hud.agents.base import Agent

from .model import Model

if TYPE_CHECKING:
    from hud.eval.run import Run

    from .adapter import ActionArray
    from .agent import RobotAgent


class BatchedModel(Model):
    """Coalesce concurrent ``ainfer`` calls into one stacked ``inner.infer``.

    Waits up to ``max_wait_s`` for callers, stacks to ``[N, ...]``, one forward,
    scatters ``[N, T, A]`` rows back. Omit ``batch_size`` to size to in-flight
    callers (``max_concurrent`` already bounds that); set it only to cap below
    concurrency (e.g. VRAM) - that also flushes early when full.

    ``inner`` must batch the leading ``N`` in one in-process forward
    (e.g. :class:`~hud.agents.robot.model.LeRobotModel`). Not
    :class:`~hud.agents.robot.model.RemoteModel` (OpenPI has no batched request;
    use one agent per rollout).
    """

    def __init__(
        self, inner: Model, *, batch_size: int | None = None, max_wait_s: float = 0.05
    ) -> None:
        self.inner = inner
        self.batch_size = None if batch_size is None else int(batch_size)
        self.max_wait_s = float(max_wait_s)
        # Bound to the running loop on first ainfer (the harness owns the loop).
        self._queue: asyncio.Queue[tuple[Any, asyncio.Future[ActionArray]]] | None = None
        self._worker: asyncio.Task[None] | None = None

    def infer(self, batch: Any) -> ActionArray:
        return self.inner.infer(batch)

    async def ainfer(self, batch: Any) -> ActionArray:
        loop = asyncio.get_running_loop()
        if self._worker is None:
            self._queue = asyncio.Queue()
            self._worker = loop.create_task(self._batch_loop())
        assert self._queue is not None
        fut: asyncio.Future[ActionArray] = loop.create_future()
        await self._queue.put((batch, fut))
        return await fut

    async def _batch_loop(self) -> None:
        assert self._queue is not None
        loop = asyncio.get_running_loop()
        while True:
            items = [await self._queue.get()]  # block for the first caller
            deadline = loop.time() + self.max_wait_s
            while self.batch_size is None or len(items) < self.batch_size:
                timeout = deadline - loop.time()
                if timeout <= 0:
                    break
                try:
                    items.append(await asyncio.wait_for(self._queue.get(), timeout))
                except TimeoutError:
                    break
            samples = [b for b, _ in items]
            try:
                torch: Any = importlib.import_module("torch")

                # Collate N raw observations into one [N, ...] batch: stack tensor
                # fields on a new leading dim, gather scalars/strings into a list.
                stacked: dict[str, Any] = {
                    k: torch.stack([s[k] for s in samples])
                    if torch.is_tensor(samples[0][k])
                    else [s[k] for s in samples]
                    for k in samples[0]
                }
                arr = await asyncio.to_thread(self.inner.infer, stacked)  # [N, T, A]
                for (_, fut), chunk in zip(items, arr, strict=True):
                    if not fut.done():
                        fut.set_result(chunk)
            except Exception as exc:  # isolate: a bad batch fails only its own callers
                for _, fut in items:
                    if not fut.done():
                        fut.set_exception(exc)


class BatchedAgent(Agent):
    """Drive many rollouts concurrently against one shared, batched model.

    Per run: shallow-clone ``agent`` with a per-run adapter copy and the shared
    :class:`BatchedModel` (stateless by contract; adapter copy keeps env bindings
    isolated). Not for :class:`~hud.agents.robot.model.RemoteModel`.

    Takes ownership: wraps ``agent.model`` in place with :class:`BatchedModel`,
    shared by every clone. Pass a dedicated agent - do not also use that instance
    for unbatched :class:`RobotAgent` rollouts.
    """

    def __init__(
        self, agent: RobotAgent, *, batch_size: int | None = None, max_wait_s: float = 0.05
    ) -> None:
        if agent.model is None:
            raise RuntimeError("BatchedAgent needs agent.model set")
        self._template = agent
        # Wrap once, in place: the passed-in agent is now permanently batched (see class doc).
        # Every per-run clone shares this batcher by reference.
        agent.model = BatchedModel(agent.model, batch_size=batch_size, max_wait_s=max_wait_s)

    async def __call__(self, run: Run, **kwargs: Any) -> None:
        worker = copy.copy(self._template)  # fresh __dict__; shares the batched model
        if worker.adapter is not None:  # defensive: a stateful custom adapter must be per-run
            worker.adapter = copy.copy(worker.adapter)
        await worker(run, **kwargs)


__all__ = ["BatchedAgent", "BatchedModel"]
