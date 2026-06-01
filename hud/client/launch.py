"""launch + Variant: connect a ``HudClient`` to a spun-up ``Sandbox``.

These are client-side conveniences on top of the (decoupled) sandbox layer:
``launch`` brings up a sandbox and attaches a client to its runtime; ``Variant``
binds (env, task, args) into something you enter directly.
"""

from __future__ import annotations

import asyncio
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit

from hud.sandbox import as_sandbox

from .client import HudClient

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from types import TracebackType

    from hud.env import Env
    from hud.sandbox import Sandbox

    from .run import Run


async def _connect_ready(
    host: str,
    port: int,
    *,
    ready_timeout: float = 120.0,
    interval: float = 0.5,
) -> HudClient:
    """Connect to a control channel, retrying until it accepts or ``ready_timeout``.

    A freshly-spun sandbox may not be serving yet; the client owns waiting for
    readiness by retrying the connect (the sandbox just hands back a url).
    """
    loop = asyncio.get_event_loop()
    deadline = loop.time() + ready_timeout
    while True:
        try:
            return await HudClient.connect(host, port)
        except OSError:
            if loop.time() >= deadline:
                raise
            await asyncio.sleep(interval)


@asynccontextmanager
async def launch(ref: Sandbox | Env) -> AsyncIterator[HudClient]:
    """Bring up a substrate for ``ref``, attach a client, tear it down on exit.

    ``ref`` is a :class:`~hud.sandbox.Sandbox` (local, container, HUD-hosted, …)
    or a live ``Env`` (wrapped in a ``LocalSandbox``). ``launch`` *owns* what it
    spins up; the client connects to the sandbox's runtime url, retrying until the
    control channel is ready.
    """
    sandbox = as_sandbox(ref)
    async with sandbox as runtime:
        parts = urlsplit(runtime.url)
        if parts.scheme not in ("", "tcp"):
            raise NotImplementedError(
                f"control transport {parts.scheme!r} not supported yet (only tcp://)",
            )
        client = await _connect_ready(parts.hostname or "127.0.0.1", parts.port or 0)
        async with client:
            yield client


@dataclass
class Variant:
    """A parameterized task on a specific env/sandbox. Enter it for a ``Run``.

    ``foo(x, y)`` (a ``Task`` call) returns one of these. Entering launches the
    env and starts the task::

        async with foo(difficulty=3) as run:        # launch(env) + client.task(...)
            await agent(run)                         # fills run.trace
        print(run.trace.reward)
    """

    env: Env | Sandbox
    task: str
    args: dict[str, Any] = field(default_factory=dict)
    _stack: AsyncExitStack | None = field(default=None, init=False, repr=False)

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

        ``env`` is a tagged env-ref resolved to a :class:`~hud.sandbox.Sandbox`
        (see :func:`hud.sandbox.sandbox_from_ref`). The task *code* is not in the
        data — it lives in the env the ref brings up.
        """
        from hud.sandbox import sandbox_from_ref

        env_ref = data.get("env")
        if not isinstance(env_ref, dict):
            raise ValueError("variant entry needs an 'env' object (a tagged env-ref)")
        task = data.get("task")
        if not isinstance(task, str):
            raise ValueError("variant entry needs a string 'task' (the task id)")
        args = data.get("args") or {}
        if not isinstance(args, dict):
            raise ValueError("variant 'args' must be an object")
        return cls(env=sandbox_from_ref(env_ref), task=task, args=args)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to ``{env, task, args}``. The env-ref is its portable identity:

        a live ``Env`` (or ``LocalSandbox``) → ``{"type": "hud", "name": ...}``; a
        ``RemoteSandbox`` → ``{"type": "url", ...}``; a ``HudSandbox`` →
        ``{"type": "hud", ...}``.
        """
        from hud.env import Env
        from hud.sandbox import HudSandbox, LocalSandbox, RemoteSandbox

        env = self.env
        if isinstance(env, LocalSandbox):
            env = env._env  # the wrapped live Env
        if isinstance(env, Env):
            ref: dict[str, Any] = {"type": "hud", "name": env.name}
        elif isinstance(env, RemoteSandbox):
            ref = {"type": "url", "url": env._url, "params": env._params}
        elif isinstance(env, HudSandbox):
            ref = {"type": "hud", "name": env.image}
        else:
            raise TypeError(
                f"cannot serialize a {type(env).__name__} env-ref; "
                "use a live Env (→ hud name), RemoteSandbox (→ url), or HudSandbox",
            )
        return {"env": ref, "task": self.task, "args": self.args}


def variant(env: Env | Sandbox, task: str, **args: Any) -> Variant:
    """Construct a :class:`Variant`: ``variant(env, "task", arg=...)``."""
    return Variant(env=env, task=task, args=args)


__all__ = ["Variant", "launch", "variant"]
