"""Variant: a parameterized task bound to a specific env/sandbox.

``foo(x, y)`` (a :class:`~hud.env.task.Task` call) returns one of these. Entering
it launches the env and starts the task, yielding a live :class:`~hud.client.Run`.
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
class Variant:
    """A parameterized task on a specific env/sandbox. Enter it for a ``Run``.

    ``foo(x, y)`` (a ``Task`` call) returns one of these. Entering launches the
    env and starts the task::

        async with foo(difficulty=3) as run:        # launch(env) + client.task(...)
            await agent(run)                         # fills run.trace
        print(run.reward)
    """

    env: Environment | Sandbox
    task: str
    args: dict[str, Any] = field(default_factory=dict)
    #: Optional sync/registry metadata (used by ``hud sync``):
    slug: str | None = None
    validation: list[dict[str, Any]] | None = None
    agent_config: dict[str, Any] | None = None
    columns: dict[str, Any] | None = None
    _stack: AsyncExitStack | None = field(default=None, init=False, repr=False)

    def default_slug(self) -> str:
        """A stable slug from the task id, disambiguated by an args hash when present."""
        if not self.args:
            return self.task
        digest = hashlib.sha1(  # noqa: S324 - non-crypto, stable disambiguator
            json.dumps(self.args, sort_keys=True, default=str).encode("utf-8"),
        ).hexdigest()[:8]
        return f"{self.task}-{digest}"

    async def __aenter__(self) -> Run:
        self._stack = AsyncExitStack()
        try:
            client = await self._stack.enter_async_context(launch(self.env))
            return await self._stack.enter_async_context(client.task(self.task, **self.args))
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

    # ─── serialization ────────────────────────────────────────────────────

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Variant:
        """Build a Variant from a serialized ``{env, task, args}`` entry.

        ``env`` is a tagged env-ref resolved to a :class:`~hud.eval.sandbox.Sandbox`
        (see :func:`hud.eval.sandbox.sandbox_from_ref`). The task *code* is not in the
        data — it lives in the env the ref brings up.
        """
        from .sandbox import sandbox_from_ref

        env_ref = data.get("env")
        if not isinstance(env_ref, dict):
            raise ValueError("variant entry needs an 'env' object (a tagged env-ref)")
        task = data.get("task")
        if not isinstance(task, str):
            raise ValueError("variant entry needs a string 'task' (the task id)")
        args = data.get("args") or {}
        if not isinstance(args, dict):
            raise ValueError("variant 'args' must be an object")
        return cls(
            env=sandbox_from_ref(env_ref),
            task=task,
            args=args,
            slug=data.get("slug"),
            validation=data.get("validation"),
            agent_config=data.get("agent_config"),
            columns=data.get("columns"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to ``{env, task, args}``. The env-ref is its portable identity:

        a live ``Environment`` (or ``LocalSandbox``) → ``{"type": "hud", "name": ...}``;
        a ``RemoteSandbox`` → ``{"type": "url", ...}``; a ``HudSandbox`` →
        ``{"type": "hud", ...}``.
        """
        from hud.environment import Environment

        from .sandbox import HudSandbox, LocalSandbox, RemoteSandbox

        env = self.env
        if isinstance(env, LocalSandbox):
            env = env._env  # the wrapped live Environment
        if isinstance(env, Environment):
            ref: dict[str, Any] = {"type": "hud", "name": env.name}
        elif isinstance(env, RemoteSandbox):
            ref = {"type": "url", "url": env._url, "params": env._params}
        elif isinstance(env, HudSandbox):
            ref = {"type": "hud", "name": env.image}
        else:
            raise TypeError(
                f"cannot serialize a {type(env).__name__} env-ref; "
                "use a live Environment (→ hud name), RemoteSandbox (→ url), or HudSandbox",
            )
        out: dict[str, Any] = {"env": ref, "task": self.task, "args": self.args}
        for key in ("slug", "validation", "agent_config", "columns"):
            value = getattr(self, key)
            if value is not None:
                out[key] = value
        return out


def variant(
    env: Environment | Sandbox,
    task: str,
    *,
    slug: str | None = None,
    validation: list[dict[str, Any]] | None = None,
    agent_config: dict[str, Any] | None = None,
    columns: dict[str, Any] | None = None,
    **args: Any,
) -> Variant:
    """Construct a :class:`Variant`: ``variant(env, "task", arg=...)``.

    Optional ``slug``/``validation``/``agent_config``/``columns`` are sync/registry
    metadata consumed by ``hud sync``.
    """
    return Variant(
        env=env,
        task=task,
        args=args,
        slug=slug,
        validation=validation,
        agent_config=agent_config,
        columns=columns,
    )


__all__ = ["Variant", "variant"]
