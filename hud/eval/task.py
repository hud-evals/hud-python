"""Task: a concrete runnable task bound to a specific env/sandbox.

``foo(x, y)`` (a task definition call) returns one of these. Entering it
launches the env and starts the task, yielding a live :class:`~hud.client.Run`.
"""

from __future__ import annotations

import hashlib
import json
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from .launch import launch

if TYPE_CHECKING:
    from types import TracebackType

    from hud.client import Run
    from hud.environment import Environment

    from .sandbox import Sandbox


@dataclass
class Task:
    """A concrete task on a specific env/sandbox. Enter it for a ``Run``."""

    env: Environment | Sandbox
    id: str
    args: dict[str, Any] = field(default_factory=dict)
    slug: str | None = None
    validation: list[dict[str, Any]] | None = None
    agent_config: dict[str, Any] | None = None
    columns: dict[str, Any] | None = None
    _stack: AsyncExitStack | None = field(default=None, init=False, repr=False)

    def default_slug(self) -> str:
        """A stable slug from the task id, disambiguated by an args hash when present."""
        if not self.args:
            return self.id
        digest = hashlib.sha1(  # noqa: S324 - non-crypto, stable disambiguator
            json.dumps(self.args, sort_keys=True, default=str).encode("utf-8"),
        ).hexdigest()[:8]
        return f"{self.id}-{digest}"

    @property
    def task(self) -> str:
        """Wire-compatible alias for the task id."""
        return self.id

    async def __aenter__(self) -> Run:
        self._stack = AsyncExitStack()
        try:
            client = await self._stack.enter_async_context(launch(self.env))
            return await self._stack.enter_async_context(client.task(self.id, **self.args))
        except BaseException:
            await self._stack.aclose()
            self._stack = None
            raise

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> bool:
        if self._stack is not None:
            await self._stack.aclose()
            self._stack = None
        return False

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Task:
        """Build a Task from a serialized ``{env, task, args}`` entry."""
        from .sandbox import sandbox_from_ref

        env_ref = data.get("env")
        if not isinstance(env_ref, dict):
            raise ValueError("task entry needs an 'env' object (a tagged env-ref)")
        task = data.get("task")
        if not isinstance(task, str):
            raise ValueError("task entry needs a string 'task' (the task id)")
        args = data.get("args") or {}
        if not isinstance(args, dict):
            raise ValueError("task 'args' must be an object")
        return cls(
            env=sandbox_from_ref(env_ref),
            id=task,
            args=args,
            slug=data.get("slug"),
            validation=data.get("validation"),
            agent_config=data.get("agent_config"),
            columns=data.get("columns"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to ``{env, task, args}`` with a portable env ref."""
        from hud.environment import Environment

        from .sandbox import HudSandbox, LocalSandbox, RemoteSandbox

        env = self.env
        if isinstance(env, LocalSandbox):
            env = env._env
        if isinstance(env, Environment):
            ref: dict[str, Any] = {"type": "hud", "name": env.name}
        elif isinstance(env, RemoteSandbox):
            ref = {"type": "url", "url": env._url, "params": env._params}
        elif isinstance(env, HudSandbox):
            ref = {"type": "hud", "name": env.image}
        else:
            raise TypeError(
                f"cannot serialize a {type(env).__name__} env-ref; "
                "use a live Environment, RemoteSandbox, or HudSandbox",
            )
        out: dict[str, Any] = {"env": ref, "task": self.id, "args": self.args}
        for key in ("slug", "validation", "agent_config", "columns"):
            value = getattr(self, key)
            if value is not None:
                out[key] = value
        return out


def task(
    env: Environment | Sandbox,
    id: str,
    *,
    slug: str | None = None,
    validation: list[dict[str, Any]] | None = None,
    agent_config: dict[str, Any] | None = None,
    columns: dict[str, Any] | None = None,
    **args: Any,
) -> Task:
    """Construct a concrete :class:`Task`: ``task(env, "id", arg=...)``."""
    return Task(
        env=env,
        id=id,
        args=args,
        slug=slug,
        validation=validation,
        agent_config=agent_config,
        columns=columns,
    )


__all__ = ["Task", "task"]
