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

    from ._types import ActionArray
    from .agent import RobotAgent


class BatchedModel(Model):
    """Coalesce concurrent ``ainfer`` calls into one stacked ``inner.infer``.

    A lazily-started worker drains up to ``batch_size`` queued calls (or waits up to
    ``max_wait_s`` for stragglers — which avoids stalling when fewer rollouts are live,
    e.g. the tail of a suite), stacks them into one ``[N, ...]`` batch, runs a single
    forward, and scatters the ``[N, T, A]`` rows back to each caller.

    ``inner`` must be an in-process, stateless model whose :meth:`~Model.infer` runs the
    whole ``[N, ...]`` batch in one forward (e.g. :class:`~hud.agents.robot.model.LeRobotModel`).
    :class:`~hud.agents.robot.model.RemoteModel` is **not** supported: it does one WebSocket
    request per env and the OpenPI server protocol has no batched-request shape, so a stacked
    batch would be mis-sent as a single env. Run one agent per rollout against it instead.
    """

    def __init__(self, inner: Model, *, batch_size: int, max_wait_s: float = 0.05) -> None:
        self.inner = inner
        self.batch_size = int(batch_size)
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
            while len(items) < self.batch_size:
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

    Per run: a shallow clone of ``agent`` (its own episode state) sharing a per-run
    adapter copy and the single :class:`BatchedModel`, so concurrent ``ainfer`` calls
    coalesce into one forward. Relies on the agent keeping per-run state out of
    ``__init__`` (assigned in ``on_episode_start``) so the clones stay isolated, and on
    the model being stateless (no per-episode ``reset``) since it is shared across clones.

    Requires an in-process batchable model; :class:`~hud.agents.robot.model.RemoteModel`
    is not supported (the OpenPI server protocol has no batched-request shape).

    Takes ownership of ``agent``: it swaps ``agent.model`` for a :class:`BatchedModel` wrapper
    in place (so the wrapper is shared by every per-run clone). The passed-in instance is
    therefore permanently batched — hand :class:`BatchedAgent` a dedicated agent and don't
    also use that same instance for direct, unbatched :class:`RobotAgent` rollouts.
    """

    def __init__(self, agent: RobotAgent, *, batch_size: int, max_wait_s: float = 0.05) -> None:
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
