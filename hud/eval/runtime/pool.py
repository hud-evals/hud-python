"""Warm instance pool: many rollouts through a few long-lived substrates."""

from __future__ import annotations

import asyncio
import contextlib
import json
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Self

from .core import Runtime, resolve_runtime_config

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from hud.eval.task import Task

    from .core import Provider


@dataclass
class _Instance:
    context: contextlib.AsyncExitStack
    address: Runtime | None = None
    ready: asyncio.Event = field(default_factory=asyncio.Event)
    failed: BaseException | None = None
    active: int = 0
    leases: int = 0
    started_at: float = field(default_factory=time.monotonic)
    retired: bool = False


class _Pool:
    """The instances serving one ``(env, runtime_config)`` key."""

    def __init__(self, owner: InstancePool) -> None:
        self.owner = owner
        self.instances: list[_Instance] = []
        self.changed = asyncio.Condition()

    def _accepting(self, instance: _Instance) -> bool:
        if instance.retired or instance.failed is not None:
            return False
        owner = self.owner
        spent = owner.rows_per_instance is not None and instance.leases >= owner.rows_per_instance
        aged = owner.ttl_s is not None and time.monotonic() - instance.started_at >= owner.ttl_s
        instance.retired = spent or aged
        return not instance.retired

    async def _boot(self, instance: _Instance, task: Task) -> None:
        try:
            instance.address = await instance.context.enter_async_context(self.owner.inner(task))
        except BaseException as exc:
            instance.failed = exc
            raise
        finally:
            instance.ready.set()

    @asynccontextmanager
    async def lease(self, task: Task) -> AsyncIterator[Runtime]:
        owner = self.owner
        async with self.changed:
            while True:
                accepting = [i for i in self.instances if self._accepting(i)]
                open_slots = [i for i in accepting if i.active < owner.width]
                if open_slots:
                    instance = min(open_slots, key=lambda i: i.active)
                    boot = False
                    break
                if len(accepting) < owner.instances:
                    instance = _Instance(contextlib.AsyncExitStack())
                    self.instances.append(instance)
                    boot = True
                    break
                await self.changed.wait()
            instance.active += 1
            instance.leases += 1
        try:
            if boot:
                await self._boot(instance, task)
            else:
                await instance.ready.wait()
            if instance.failed is not None:
                # A failed boot fails only the leases that chose it; the next lease boots anew.
                raise instance.failed
            assert instance.address is not None
            yield instance.address
        finally:
            async with self.changed:
                instance.active -= 1
                finished = instance.failed is not None or (
                    instance.retired and instance.active == 0
                )
                if finished and instance in self.instances:
                    self.instances.remove(instance)
                self.changed.notify_all()
            if finished:
                await instance.context.aclose()

    async def close(self) -> None:
        instances, self.instances = self.instances, []
        for instance in instances:
            with contextlib.suppress(Exception):
                await instance.context.aclose()


class InstancePool:
    """Lease provider: up to ``instances`` warm substrates per task placement, each shared by
    at most ``width`` concurrent rollouts and retired after ``rows_per_instance`` leases or
    ``ttl_s`` seconds.

    The generalization of :class:`Shared` for batch work: environment startup is
    paid once per instance instead of once per row, and rotation bounds how long
    one instance lives (a hosted environment has a platform lifetime cap). A
    retired instance finishes the leases it holds, then stops; a replacement
    boots on the next lease. Boots run concurrently, and a failed boot fails only
    the lease that started it. Like :class:`Shared`, it lives for the enclosing
    ``async with`` scope (``Taskset.run`` opens one for a bare placement)::

        async with InstancePool(HUDRuntime(), instances=8, width=4, rows_per_instance=2000) as rt:
            await rollout(task, agent, runtime=rt, job_id=job.id)
    """

    def __init__(
        self,
        inner: Provider,
        *,
        instances: int = 1,
        width: int = 1,
        rows_per_instance: int | None = None,
        ttl_s: float | None = None,
    ) -> None:
        if instances < 1 or width < 1:
            raise ValueError("InstancePool instances and width must be >= 1")
        if rows_per_instance is not None and rows_per_instance < 1:
            raise ValueError("rows_per_instance must be >= 1")
        if ttl_s is not None and ttl_s <= 0:
            raise ValueError("ttl_s must be > 0")
        self.inner = inner
        self.instances = instances
        self.width = width
        self.rows_per_instance = rows_per_instance
        self.ttl_s = ttl_s
        self._pools: dict[tuple[str, str], _Pool] = {}
        self._opens = 0

    async def __aenter__(self) -> Self:
        self._opens += 1
        return self

    async def __aexit__(self, *exc: object) -> None:
        self._opens -= 1
        if self._opens == 0:
            pools, self._pools = self._pools, {}
            for pool in pools.values():
                await pool.close()

    @property
    def live(self) -> int:
        """Instances currently up (booting, serving, or draining)."""
        return sum(len(pool.instances) for pool in self._pools.values())

    @asynccontextmanager
    async def __call__(self, task: Task) -> AsyncIterator[Runtime]:
        if self._opens == 0:
            raise RuntimeError(
                "InstancePool substrates outlive single rollouts; lease inside the scope "
                "(Taskset.run opens it for you, or wrap calls in `async with InstancePool(...)`)"
            )
        config = resolve_runtime_config(self.inner, task)
        key = (
            task.env,
            json.dumps(config.model_dump(mode="python", exclude_unset=True), sort_keys=True)
            if config is not None
            else "",
        )
        pool = self._pools.get(key)
        if pool is None:
            pool = self._pools[key] = _Pool(self)
        async with pool.lease(task) as address:
            yield address


__all__ = ["InstancePool"]
