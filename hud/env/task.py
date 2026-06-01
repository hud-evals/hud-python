"""Task: async-generator that yields {"prompt": ...} then {"score": ...}.

A ``Task`` is the in-env challenge definition (formerly "scenario"): an async
generator that yields a prompt for the agent, then — once an answer is sent
back via ``asend`` — yields a score. ``TaskRunner`` drives one task through
its ``start -> evaluate`` lifecycle.
"""

from __future__ import annotations

import contextlib
import functools
import inspect
from collections.abc import AsyncGenerator, Callable
from typing import TYPE_CHECKING, Any, Generic, ParamSpec

if TYPE_CHECKING:
    from hud.client import Variant
    from hud.env.env import Env

TaskFn = Callable[..., AsyncGenerator[dict[str, Any], dict[str, Any]]]

P = ParamSpec("P")


class Task(Generic[P]):
    """A registered challenge — and a typed factory for runnable variants.

    Returned by ``@env.task``. Holds the async-generator ``func`` (prompt -> score),
    identity (``id`` / ``description``), and the owning ``env``. ``TaskRunner`` drives
    ``func`` server-side; calling the ``Task`` with the task's args binds a runnable
    :class:`~hud.client.Variant`, type-checked against the signature via ``ParamSpec``::

        @env.task(id="fix_bug")
        async def fix_bug(difficulty: int = 1, hint: str | None = None): ...

        variant_1 = fix_bug(difficulty=3, hint="line 42")   # -> Variant (type-checked)
        async with variant_1 as run:
            await run.rollout(agent)
    """

    def __init__(
        self,
        env: Env,
        id: str,
        description: str,
        func: Callable[P, AsyncGenerator[dict[str, Any], dict[str, Any]]],
    ) -> None:
        self.env = env
        self.id = id
        self.description = description
        self.func: TaskFn = func
        self._sig = inspect.signature(func)
        functools.update_wrapper(self, func)

    def manifest_entry(self) -> dict[str, Any]:
        return {"id": self.id, "description": self.description}

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> Variant:
        from hud.client import Variant  # local import: avoid env<->client cycle

        bound = self._sig.bind(*args, **kwargs)
        return Variant(env=self.env, task=self.id, args=dict(bound.arguments))


class TaskRunner:
    """Drives one task through prompt -> evaluate."""

    def __init__(self, task: Task[Any], args: dict[str, Any] | None = None) -> None:
        self.task = task
        self._args = args or {}
        self._gen: AsyncGenerator[dict[str, Any], dict[str, Any]] | None = None

        # Fail fast on bad args (TypeError before any side-effects run).
        try:
            inspect.signature(task.func).bind(**self._args)
        except TypeError as exc:
            raise TypeError(
                f"task {task.id!r}: bad args {sorted(self._args)}: {exc}",
            ) from exc

    async def start(self) -> dict[str, Any]:
        self._gen = self.task.func(**self._args)
        prompt = await self._gen.__anext__()
        if not isinstance(prompt, dict) or "prompt" not in prompt:
            raise RuntimeError(
                f"task {self.task.id!r}: first yield must be a dict with 'prompt'",
            )
        return prompt

    async def evaluate(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self._gen is None:
            raise RuntimeError("task not started")
        try:
            evaluation = await self._gen.asend(payload)
        except StopAsyncIteration as exc:
            raise RuntimeError(
                f"task {self.task.id!r}: ended without yielding an evaluation",
            ) from exc
        if not isinstance(evaluation, dict) or "score" not in evaluation:
            raise RuntimeError(
                f"task {self.task.id!r}: second yield must be a dict with 'score'",
            )
        with contextlib.suppress(Exception):
            await self._gen.aclose()
        return evaluation

    async def cancel(self) -> None:
        if self._gen is not None:
            with contextlib.suppress(Exception):
                await self._gen.aclose()
            self._gen = None


__all__ = ["Task", "TaskFn", "TaskRunner"]
